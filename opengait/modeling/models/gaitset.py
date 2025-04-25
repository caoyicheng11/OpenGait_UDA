import torch
import copy
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv2d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper
import numpy as np
from utils import np2var, list2var, get_valid_args, ddp_all_gather
from data.transform import get_transform

class GaitSet(BaseModel):
    """
        GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition
        Arxiv:  https://arxiv.org/abs/1811.06186
        Github: https://github.com/AbnerHqC/GaitSet
    """

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
        in_c = model_cfg['in_channels']
        self.set_block1 = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2 = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3 = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))

        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_pooling = PackSequenceWrapper(torch.max)

        self.Head = SeparateFCs(**model_cfg['SeparateFCs'])

        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.alpha = nn.Parameter(torch.Tensor([0.9]))

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  # [n, s, h, w]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        outs = self.set_block1(sils)
        gl = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        feature2 = self.HPP(gl)  # [n, c, p]
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs = self.Head(feature)

        weight = self.alpha
        head = embs[:, :, :15]
        body = embs[:, :, 15:-16]
        foot = embs[:, :, -16:]
        part1 = torch.cat([head,foot], dim=2)
        part2 = body
        embed = part1 * weight + part2 * (1 - weight)

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
