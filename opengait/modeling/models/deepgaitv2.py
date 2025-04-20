import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D

from utils import np2var, list2var, get_valid_args, ddp_all_gather
from data.transform import get_transform
from einops import rearrange

blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class DeepGaitV2(BaseModel):
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
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d': 
            strides = [
                [1, 1], 
                [1, 2, 2], 
                [1, 2, 2], 
                [1, 1, 1]
            ]
        else: 
            strides = [
                [1, 1], 
                [2, 2], 
                [2, 2], 
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1), 
            nn.BatchNorm2d(self.inplanes), 
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d': 
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        # self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        self.alpha = nn.Parameter(torch.Tensor([0.9]))


    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        
        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        # embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
                embed = embed_2
        else:
                embed = embed_1

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
                # 'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval
