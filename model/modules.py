import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from .common import SingleStageTCN
from .impl.asformer import MyTransformer
from .impl.gtad import GCNeXt
from .impl.actionformer import ConvTransformerBackbone, FPN1D, PtTransformerClsHead

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class FCPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class VideoClassifier(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        self._fc = nn.Linear(feat_dim, feat_dim)
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self._dropout = nn.Dropout()
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._dropout(self._relu(self._fc(x)))
        return self._fc_out(x)


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class GRU(nn.Module):

    def __init__(self, feat_dim, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, hidden_dim)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class TCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_stages=1, num_layers=5):
        super().__init__()

        self._tcn = SingleStageTCN(
            feat_dim, 256, num_classes, num_layers, True)
        self._stages = None
        if num_stages > 1:
            self._stages = nn.ModuleList([SingleStageTCN(
                num_classes, 256, num_classes, num_layers, True)
                for _ in range(num_stages - 1)])

    def forward(self, x):
        x = self._tcn(x)
        if self._stages is None:
            return x
        else:
            outputs = [x]
            for stage in self._stages:
                x = stage(F.softmax(x, dim=2))
                outputs.append(x)
            return torch.stack(outputs, dim=0)


class ASFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_decoders=3, num_layers=5):
        super().__init__()

        r1, r2 = 2, 2
        num_f_maps = 64
        self._net = MyTransformer(
            num_decoders, num_layers, r1, r2, num_f_maps, feat_dim,
            num_classes, channel_masking_rate=0.3)

    def forward(self, x):
        B, T, D = x.shape
        return self._net(
            x.permute(0, 2, 1), torch.ones((B, 1, T), device=x.device)
        ).permute(0, 1, 3, 2)
    
    
class GCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim=256, num_layers=2):
        super().__init__()

        self.idx_list = []
        self.fc_in = nn.Linear(feat_dim, hidden_dim)
        gcn_layers = [GCNeXt(hidden_dim, hidden_dim, k=3, groups=32,
                             idx=self.idx_list) for _ in range(num_layers)]
        self.backbone = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=4),
            nn.ReLU(inplace=True),
            *gcn_layers
        )
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        del self.idx_list[:]
        batch_size, clip_len, _ = x.shape
        x = self.fc_in(x.view(batch_size * clip_len, -1))
        x = F.relu(x).view(batch_size, clip_len, -1)
        x = self.backbone(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        return self.fc(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class ActionFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, kernal_size=3, d_model=256, n_head=4, max_len=128):
        super().__init__()

        self.backbone = ConvTransformerBackbone(feat_dim, d_model, n_head, 3, max_len)
        self.neck = FPN1D([d_model], d_model)
        self.cls_head = PtTransformerClsHead(d_model, d_model, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        B, C, T = x.size()
        batch_masks = torch.ones((B, 1, T), device=x.device, dtype=torch.bool)
        feats, masks = self.backbone(x, batch_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        return out_cls_logits[0].transpose(1, 2)
        