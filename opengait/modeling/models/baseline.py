import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from utils import np2var, list2var, get_valid_args, ddp_all_gather
from data.transform import get_transform
from einops import rearrange
import torch.nn as nn
import numpy as np

class Baseline(BaseModel):
    def inputs_pretreament(self, inputs):
        if self.training:
            seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
            trf_cfgs = self.engine_cfg['transform']
            seq_trfs = get_transform(trf_cfgs)

            requires_grad = True if self.training else False
            seq_size = int(len(seqs_batch[0][0]) / 2)
            img_q = [np2var(np.asarray([trf(fra) for fra in np.asarray(seq)[:, :seq_size]]), requires_grad=requires_grad).float()  for trf, seq in zip(seq_trfs, seqs_batch)]
            img_k = [np2var(np.asarray([trf(fra) for fra in np.asarray(seq)[:, seq_size:]]), requires_grad=requires_grad).float()  for trf, seq in zip(seq_trfs, seqs_batch)]
            seqs = [torch.cat([img_q[0], img_k[0]], dim=0)]

            typs = typs_batch
            vies = vies_batch

            labs = list2var(labs_batch).long()
            labs = torch.cat([labs, labs], dim=0)

            if seqL_batch is not None:
                seqL_batch = np2var(seqL_batch).int()
            seqL = seqL_batch

            ipts = seqs
            del seqs

            return ipts, labs, typs, vies, seqL
        else:
            return super().inputs_pretreament(inputs)

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        # self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.alpha = nn.Parameter(torch.Tensor([0.9]))

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        # embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        weight = self.alpha
        head = embed_1[:, :, :4]
        body = embed_1[:, :, 4:-4]
        foot = embed_1[:, :, -4:]
        part1 = torch.cat([head,foot], dim=2)
        part2 = body
        embed = part1 * weight + part2 * (1 - weight)


        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                # 'softmax': {'embeddings': embed_1, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval