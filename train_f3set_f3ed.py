#!/usr/bin/env python3
""" Training for F3Set """
import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
import torch
import math
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from torch.utils.data import DataLoader
import torchvision
from itertools import groupby
import timm
from tqdm import tqdm

from model.common import step, BaseRGBModel
from model.shift import make_temporal_shift
from model.slowfast import ResNet3dSlowFast
from model.modules import *
from dataset.frame_process import ActionSeqDataset, ActionSeqVideoDataset
from util.eval import edit_score, non_maximum_suppression, non_maximum_suppression_np
from util.io import load_json, store_json, clear_files
from util.dataset import DATASETS, load_classes
import warnings
warnings.filterwarnings("ignore")

EPOCH_NUM_FRAMES = 500000
BASE_NUM_WORKERS = 4
BASE_NUM_VAL_EPOCHS = 20
INFERENCE_BATCH_SIZE = 4
HIDDEN_DIM = 768

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('frame_dir', type=str, help='Path to extracted frames')

    parser.add_argument(
        '-m', '--feature_arch', type=str, required=True, choices=[
            # From torchvision
            'rn50',
            'rn50_tsm',
            'rn50_gsm'
            'rny002',
            'rny002_tsm',
            'rny002_gsm',
            'rny008',
            'rny008_tsm',
            'rny008_gsm',
            'slowfast'
        ], help='architecture for feature extraction')
    parser.add_argument(
        '-t', '--temporal_arch', type=str, default='gru',
        choices=['gru', 'deeper_gru', 'mstcn', 'asformer', 'actionformer', 'gcn', 'tcn', 'fc'])
    parser.add_argument(
        '-ctx', '--use_ctx', type=bool, default=True,
        help='Whether include the contextual module; if not, just use multi-label classifier')

    parser.add_argument('--clip_len', type=int, default=96)
    parser.add_argument('--crop_dim', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--window', type=int, default=5,
                        help='Non maximun suppression window size')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')

    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-s', '--save_dir', type=str, required=True,
                        help='Dir to save checkpoints and predictions')

    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint in <save_dir>')

    parser.add_argument('--start_val_epoch', type=int, default=30)
    parser.add_argument('--criterion', choices=['edit', 'loss'], default='edit')

    parser.add_argument('-j', '--num_workers', type=int,
                        help='Base number of dataloader workers')

    parser.add_argument('-mgpu', '--gpu_parallel', action='store_true')
    return parser.parse_args()


class F3Set(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, step=1, window=5, use_ctx=True, device='cuda'):
            super().__init__()
            is_rgb = True
            self._device = device
            self._num_classes = num_classes
            self._window = window
            self._use_ctx = use_ctx

            if 'rn50' in feature_arch:
                resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
                glb_feat = getattr(torchvision.models, resnet_name)(pretrained=True)
                glb_feat_dim = features.fc.in_features
                glb_feat.fc = nn.Identity()


            elif feature_arch.startswith(('rny002', 'rny008')):
                glb_feat = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[feature_arch.rsplit('_', 1)[0]], pretrained=True)
                glb_feat_dim = glb_feat.head.fc.in_features
                glb_feat.head.fc = nn.Identity()

            elif 'slowfast' in feature_arch:
                glb_feat = ResNet3dSlowFast(None, slow_upsample=8)
                glb_feat.load_pretrained_weight()
                glb_feat_dim = 2304

            else:
                raise NotImplementedError(feature_arch)

            # Add Temporal Shift Modules
            self._require_clip_len = clip_len
            if feature_arch.endswith('_tsm'):
                make_temporal_shift(glb_feat, clip_len, is_gsm=False, step=step)
                self._require_clip_len = clip_len
            elif feature_arch.endswith('_gsm'):
                make_temporal_shift(glb_feat, clip_len, is_gsm=True)
                self._require_clip_len = clip_len

            self._glb_feat = glb_feat  # global feature extractor
            self._feat_dim = glb_feat_dim
            self._is_3d = 'slowfast' in feature_arch

            # head modules
            d_model = min(HIDDEN_DIM, self._feat_dim)
            if temporal_arch == 'gru':  # single layer GRU
                self._head = GRU(self._feat_dim, d_model, num_layers=1)
            elif temporal_arch == 'deeper_gru':  # deeper GRU
                self._head = GRU(self._feat_dim, d_model, num_layers=3)
            else:
                raise NotImplementedError(temporal_arch)

            # binary predictor, hit or not
            self._coarse_pred = nn.Linear(d_model, 2)

            # multi-label fine predictor
            self._fine_pred = nn.Linear(d_model, num_classes)

            # contextual module
            if use_ctx:
                self._ctx = GRUPrediction(num_classes + 1, num_classes + 1, d_model, num_layers=1)
            
        def forward(self, frame, coarse_label=None, fine_label=None, hand=None, max_seq_len=20):
            batch_size, true_clip_len, channels, height, width = frame.shape

            clip_len = true_clip_len
            if self._require_clip_len > 0:
                # TSM module requires clip len to be known
                assert true_clip_len <= self._require_clip_len, \
                    'Expected {}, got {}'.format(
                        self._require_clip_len, true_clip_len)
                if true_clip_len < self._require_clip_len:
                    frame = F.pad(
                        frame, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                    clip_len = self._require_clip_len

            # global visual embedding
            if self._is_3d:
                im_feat = self._glb_feat(frame.transpose(1, 2)).transpose(1, 2)
            else:
                im_feat = self._glb_feat(frame.view(-1, channels, height, width)).reshape(batch_size, clip_len, -1)

            # head module 
            enc_feat = self._head(im_feat)

            # coarse-grained prediction
            coarse_pred = self._coarse_pred(enc_feat)

            # fine-grained prediction
            fine_pred = self._fine_pred(enc_feat)

            # no contextual module
            if not self._use_ctx:
                return coarse_pred, fine_pred, None, None, None

            coarse_pred_score = torch.softmax(coarse_pred, axis=2)
            fine_pred_score = torch.sigmoid(fine_pred).to(dtype=fine_pred.dtype)

            if coarse_label is None:
                coarse_label = non_maximum_suppression(coarse_pred_score, self._window)
                coarse_label = torch.argmax(coarse_label, axis=2)
            else:
                coarse_label = coarse_pred_score * coarse_label.unsqueeze(-1)
                coarse_label = torch.argmax(coarse_label, axis=2)

            if fine_label is None:
                fine_label = fine_pred_score

            seq_pred = torch.zeros(batch_size, max_seq_len, self._num_classes + 1, dtype=fine_label.dtype, device=self._device)
            seq_label = torch.zeros(batch_size, max_seq_len, self._num_classes + 1, dtype=fine_label.dtype, device=self._device)
            seq_mask = torch.ones((batch_size, max_seq_len), dtype=torch.bool, device=self._device)

            for i in range(batch_size):
                # sequence of ground truth labels
                selected_label = fine_label[i, coarse_label[i].bool()]
                seq_label[i, :selected_label.shape[0], 1:] = selected_label

                # sequence of predicted classes
                selected_pred = fine_pred_score[i, coarse_label[i].bool()]
                seq_pred[i, :selected_label.shape[0], 1:] = selected_pred
                for j in range(selected_label.shape[0]):
                    seq_pred[i, j, 0] = hand[i, int(torch.round(selected_pred[j, 0]))]
                    seq_label[i, j, 0] = hand[i, int(torch.round(selected_label[j, 0]))]

                # sequence mask
                seq_mask[i, :selected_label.shape[0]] = False

            # contextual module refine sequence
            seq_pred_refined = self._ctx(seq_pred)
            return coarse_pred, fine_pred, seq_pred_refined, seq_label, seq_mask

    def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, step=1, window=5, use_ctx=True, device='cuda', multi_gpu=False):
        self._device = device
        self._multi_gpu = multi_gpu
        self._window = window
        self._use_ctx = use_ctx
        self._model = F3Set.Impl(num_classes, feature_arch, temporal_arch, clip_len, step=step, window=window, use_ctx=use_ctx)

        if multi_gpu:
            self._model = nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, acc_grad_iter=1, fg_weight=5, epoch=0):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        # coarse-grained frame binary classification weight
        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor([1, fg_weight]).to(self._device)

        epoch_loss = 0.
        with (torch.no_grad() if optimizer is None else nullcontext()):
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame, src_key_padding_mask = loader.dataset.load_frame_gpu(batch, self._device)

                coarse_label = batch['coarse_label'].to(self._device)
                fine_label = batch['fine_label'].float().to(self._device)
                hand = batch['hand'].float().to(self._device)

                with torch.cuda.amp.autocast():
                    coarse_pred, fine_pred, seq_pred, seq_label, seq_mask = self._model(frame, coarse_label, fine_label, hand=hand)

                    loss = 0.
                    # coarse-grained binary classification loss
                    coarse_loss = F.cross_entropy(coarse_pred.reshape(-1, 2), coarse_label.flatten(), **ce_kwargs)
                    if not math.isnan(coarse_loss.item()):
                        loss += coarse_loss

                    # fine-grained multi-label loss
                    fine_bce_loss = F.binary_cross_entropy_with_logits(fine_pred, fine_label.float(), reduction='none')
                    fine_mask = coarse_label.unsqueeze(2).expand_as(fine_pred)
                    masked_fine_loss = fine_bce_loss * fine_mask
                    fine_loss = masked_fine_loss.sum() / fine_mask.sum()
                    if not math.isnan(fine_loss.item()):
                        loss += fine_loss
                    
                    # contextual loss
                    ctx_loss = 0.
                    if self._use_ctx and (~seq_mask).any():
                        ctx_loss = F.binary_cross_entropy_with_logits(seq_pred[~seq_mask], seq_label[~seq_mask])
                        if not math.isnan(ctx_loss.item()):
                            loss += ctx_loss

                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter, lr_scheduler=lr_scheduler,
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()
        return epoch_loss / len(loader)     # Avg loss

    def predict(self, frame, hand, use_amp=True):
        if not isinstance(frame, torch.Tensor):
            frame = torch.FloatTensor(frame)
        if len(frame.shape) == 4:  # (L, C, H, W)
            frame = frame.unsqueeze(0)
        frame = frame.to(self._device)
        hand = hand.to(self._device)

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                coarse_pred, fine_pred, seq_pred, _, _ = self._model(frame, hand=hand)
            coarse_pred = torch.softmax(coarse_pred, axis=2)
            coarse_pred_cls = torch.argmax(non_maximum_suppression(coarse_pred, self._window), axis=2)

            fine_pred_refine = fine_pred 
            if self._use_ctx:
                for i in range(coarse_pred_cls.size(0)):
                    shot_id = 0
                    for j in range(coarse_pred_cls.size(1)):
                        if coarse_pred_cls[i, j] == 1:
                            fine_pred_refine[i, j] = seq_pred[i, shot_id, 1:]
                            shot_id += 1

            fine_pred = torch.sigmoid(fine_pred_refine)
            return coarse_pred_cls.cpu().numpy(), coarse_pred.cpu().numpy(), fine_pred.cpu().numpy()


def evaluate(model, dataset, classes, delta=0, window=5, device='cuda'):
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, 2), np.float32),
            np.zeros((video_len, len(classes)), np.float32),
            np.zeros(video_len, np.int32))

    classes_inv = {v: k for k, v in classes.items()}
    classes_inv[0] = 'NA'

    # Do not up the batch size if the dataset augments
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    for clip in tqdm(DataLoader(
            dataset, num_workers=BASE_NUM_WORKERS * 2, pin_memory=True,
            batch_size=batch_size
    )):

        if batch_size > 1:
            # Batched by dataloader
            _, batch_coarse_scores, batch_fine_scores = model.predict(clip['frame'], clip['hand'])
            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                coarse_scores, fine_scores, support = pred_dict[video]
                coarse_pred_scores = batch_coarse_scores[i]
                fine_pred_scores = batch_fine_scores[i]

                start = clip['start'][i].item()
                if start < 0:
                    coarse_pred_scores = coarse_pred_scores[-start:, :]
                    fine_pred_scores = fine_pred_scores[-start:, :]
                    start = 0
                end = start + coarse_pred_scores.shape[0]
                if end >= coarse_scores.shape[0]:
                    end = coarse_scores.shape[0]
                    coarse_pred_scores = coarse_pred_scores[:end - start, :]
                    fine_pred_scores = fine_pred_scores[:end - start, :]
                coarse_scores[start:end, :] += coarse_pred_scores
                fine_scores[start:end, :] += fine_pred_scores
                support[start:end] += 1

    # evaluation metrices
    f = open('error_sequences_f3ed.txt', 'w')
    edit_scores_high, edit_scores_mid, edit_scores_low = [], [], []
    f1_element = np.zeros((len(classes), 3), int)
    f1_event_high = dict() #np.zeros((len(classes), 3), int)
    f1_event_mid = dict() #np.zeros((len(classes), 3), int)
    f1_event_low = dict() #np.zeros((len(classes), 3), int)
    for video, (coarse_scores, fine_scores, support) in sorted(pred_dict.items()):
        coarse_label, fine_label = dataset.get_labels(video)
        coarse_scores /= support[:, None]
        fine_scores /= support[:, None]

        # argmax pred
        coarse_scores = non_maximum_suppression_np(coarse_scores, 5)
        coarse_pred = np.argmax(coarse_scores, axis=1)

        fine_pred = np.zeros_like(fine_scores, int)
        for i in range(len(fine_scores)):
            for start, end in [[0, 2], [2, 5], [5, 8], [16, 24], [25, 29]]:
                max_idx = np.argmax(fine_scores[i, start:end])
                fine_pred[i, start + max_idx] = 1
            if fine_scores[i, 24] > 0.5:  # approach
                fine_pred[i, 24] = 1
            if fine_pred[i, 5] != 1:  # not a serve
                for start, end in [[8, 10], [10, 16]]:
                    max_idx = np.argmax(fine_scores[i, start:end])
                    fine_pred[i, start + max_idx] = 1

        # fine_pred = fine_scores
        # fine_pred[fine_pred > 0.5] = 1
        # fine_pred[fine_pred <= 0.5] = 0
        # fine_pred = fine_pred.astype(int)

        fine_pred = coarse_pred[:, np.newaxis] * fine_pred

        # element F1 scores
        for i in range(len(fine_pred)):
            for j in range(len(fine_pred[0])):
                if fine_pred[i, j] == 1 and sum(
                        fine_label[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 1:
                    f1_element[j, 0] += 1  # tp
                if fine_pred[i, j] == 1 and sum(
                        fine_label[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 0:
                    f1_element[j, 1] += 1  # fp
                if fine_pred[i, j] == 0 and sum(
                        fine_label[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 1:
                    f1_element[j, 2] += 1  # fn

        print_preds, print_gts = [], []
        # tp, fp, fn
        for i in range(len(fine_pred)):
            if coarse_label[i] == 1:
                print_gt = []
                for j in range(len(fine_pred[0])):
                    if fine_label[i, j] == 1:
                        print_gt.append(classes_inv[j + 1])
                print_gts.append('_'.join(print_gt))
            if coarse_pred[i] == 1:
                print_pred = []
                for j in range(len(fine_pred[0])):
                    if fine_pred[i, j] == 1:
                        print_pred.append(classes_inv[j + 1])
                print_preds.append('_'.join(print_pred))

        labels_high = fine_label
        preds_high = fine_pred
        labels_mid = labels_high[:, :24]
        preds_mid = preds_high[:, :24]
        labels_low = labels_high[:, [0, 1, 5, 6, 7, 8, 9, 25, 26, 27, 28]]
        preds_low = preds_high[:, [0, 1, 5, 6, 7, 8, 9, 25, 26, 27, 28]]

        labels_high = [int(''.join(str(x) for x in row), 2) for row in labels_high]
        preds_high = [int(''.join(str(x) for x in row), 2) for row in preds_high]
        labels_mid = [int(''.join(str(x) for x in row), 2) for row in labels_mid]
        preds_mid = [int(''.join(str(x) for x in row), 2) for row in preds_mid]
        labels_low = [int(''.join(str(x) for x in row), 2) for row in labels_low]
        preds_low = [int(''.join(str(x) for x in row), 2) for row in preds_low]

        preds_high = coarse_pred * preds_high
        preds_mid = coarse_pred * preds_mid
        preds_low = coarse_pred * preds_low

        # event F1 scores
        for i in range(len(preds_high)):
            if preds_high[i] > 0 and preds_high[i] in labels_high[
                                                      max(0, i - delta):min(len(preds_high), i + delta + 1)]:
                if preds_high[i] not in f1_event_high:
                    f1_event_high[preds_high[i]] = [1, 0, 0]
                else:
                    f1_event_high[preds_high[i]][0] += 1
            if preds_high[i] > 0 and sum(labels_high[max(0, i - delta):min(len(preds_high), i + delta + 1)]) == 0:
                if preds_high[i] not in f1_event_high:
                    f1_event_high[preds_high[i]] = [0, 1, 0]
                else:
                    f1_event_high[preds_high[i]][1] += 1
            if labels_high[i] > 0 and labels_high[i] not in preds_high[
                                                            max(0, i - delta):min(len(preds_high), i + delta + 1)]:
                if labels_high[i] not in f1_event_high:
                    f1_event_high[labels_high[i]] = [0, 0, 1]
                else:
                    f1_event_high[labels_high[i]][2] += 1

            if preds_mid[i] > 0 and preds_mid[i] in labels_mid[max(0, i - delta):min(len(preds_high), i + delta + 1)]:
                if preds_mid[i] not in f1_event_mid:
                    f1_event_mid[preds_mid[i]] = [1, 0, 0]
                else:
                    f1_event_mid[preds_mid[i]][0] += 1
            if preds_mid[i] > 0 and sum(labels_mid[max(0, i - delta):min(len(preds_high), i + delta + 1)]) == 0:
                if preds_mid[i] not in f1_event_mid:
                    f1_event_mid[preds_mid[i]] = [0, 1, 0]
                else:
                    f1_event_mid[preds_mid[i]][1] += 1
            if labels_mid[i] > 0 and labels_mid[i] not in preds_mid[
                                                          max(0, i - delta):min(len(preds_high), i + delta + 1)]:
                if labels_mid[i] not in f1_event_mid:
                    f1_event_mid[labels_mid[i]] = [0, 0, 1]
                else:
                    f1_event_mid[labels_mid[i]][2] += 1

            if preds_low[i] > 0 and preds_low[i] in labels_low[max(0, i - delta):min(len(preds_high), i + delta + 1)]:
                if preds_low[i] not in f1_event_low:
                    f1_event_low[preds_low[i]] = [1, 0, 0]
                else:
                    f1_event_low[preds_low[i]][0] += 1
            if preds_low[i] > 0 and sum(labels_low[max(0, i - delta):min(len(preds_high), i + delta + 1)]) == 0:
                if preds_low[i] not in f1_event_low:
                    f1_event_low[preds_low[i]] = [0, 1, 0]
                else:
                    f1_event_low[preds_low[i]][1] += 1
            if labels_low[i] > 0 and labels_low[i] not in preds_low[
                                                          max(0, i - delta):min(len(preds_high), i + delta + 1)]:
                if labels_low[i] not in f1_event_low:
                    f1_event_low[labels_low[i]] = [0, 0, 1]
                else:
                    f1_event_low[labels_low[i]][2] += 1

        gt = [k for k, g in groupby(labels_high) if k != 0]
        pred = [k for k, g in groupby(preds_high) if k != 0]
        gt_mid = [k for k, g in groupby(labels_mid) if k != 0]
        pred_mid = [k for k, g in groupby(preds_mid) if k != 0]
        gt_low = [k for k, g in groupby(labels_low) if k != 0]
        pred_low = [k for k, g in groupby(preds_low) if k != 0]

        # record error sequences
        if len(pred) == len(gt):
            for j in range(len(pred)):
                if pred[j] != gt[j]:
                    f.write(video + '\n')
                    f.write('->'.join(print_preds) + '\n')
                    f.write('->'.join(print_gts) + '\n')
                    f.write('\n')
                    break
        else:
            f.write(video + '\n')
            f.write('->'.join(print_preds) + '\n')
            f.write('->'.join(print_gts) + '\n')
            f.write('\n')

        edit_scores_high.append(edit_score(pred, gt))
        edit_scores_mid.append(edit_score(pred_mid, gt_mid))
        edit_scores_low.append(edit_score(pred_low, gt_low))

    f.close()

    f1_high, count = 0, 0
    for value in f1_event_high.values():
        if sum(value) == 0:
            continue
        precision = value[0] / (value[0] + value[1] + 1e-10)
        recall = value[0] / (value[0] + value[2] + 1e-10)
        f1_high += 2 * precision * recall / (precision + recall + 1e-10)
        count += 1
    f1_high /= count

    f1_mid, count = 0, 0
    for value in f1_event_mid.values():
        if sum(value) == 0:
            continue
        precision = value[0] / (value[0] + value[1] + 1e-10)
        recall = value[0] / (value[0] + value[2] + 1e-10)
        f1_mid += 2 * precision * recall / (precision + recall + 1e-10)
        count += 1
    f1_mid /= count

    f1_low, count = 0, 0
    for value in f1_event_low.values():
        if sum(value) == 0:
            continue
        precision = value[0] / (value[0] + value[1] + 1e-10)
        recall = value[0] / (value[0] + value[2] + 1e-10)
        f1_low += 2 * precision * recall / (precision + recall + 1e-10)
        count += 1
    f1_low /= count

    print('Mean F1 (event) high:', np.mean(f1_high))
    print('Mean F1 (event) mid:', np.mean(f1_mid))
    print('Mean F1 (event) low:', np.mean(f1_low))
    print()

    precision = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 1] + 1e-10)
    recall = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 2] + 1e-10)
    f1_high = 2 * precision * recall / (precision + recall + 1e-10)
    f1_mid = f1_high[:24]
    f1_low = f1_high[[0, 1, 5, 6, 7, 8, 9, 25, 26, 27, 28]]

    print(f1_high)
    print()

    print('Mean F1 (element) high:', np.mean(f1_high))
    print('Mean F1 (element) mid:', np.mean(f1_mid))
    print('Mean F1 (element) low:', np.mean(f1_low))
    print()

    print('Edit high:', sum(edit_scores_high) / len(edit_scores_high))
    print('Edit mid:', sum(edit_scores_mid) / len(edit_scores_mid))
    print('Edit low:', sum(edit_scores_low) / len(edit_scores_low))
    return sum(edit_scores_high) / len(edit_scores_high)
    # edit_scores = []
    # f1_element = np.zeros((len(classes), 3), int)
    # f1_event = dict() 
    # for video, (coarse_scores, fine_scores, support) in sorted(pred_dict.items()):
    #     coarse_label, fine_label = dataset.get_labels(video)
    #     coarse_scores /= support[:, None]
    #     fine_scores /= support[:, None]

    #     # argmax pred
    #     coarse_scores = non_maximum_suppression_np(coarse_scores, 5)
    #     coarse_pred = np.argmax(coarse_scores, axis=1)

    #     # dataset specific
    #     fine_pred = np.zeros_like(fine_scores, int)
    #     for i in range(len(fine_scores)):
    #         for start, end in [[0, 2], [2, 5], [5, 8], [16, 24], [25, 29]]:
    #             max_idx = np.argmax(fine_scores[i, start:end])
    #             fine_pred[i, start + max_idx] = 1
    #         if fine_scores[i, 24] > 0.5:  # approach
    #             fine_pred[i, 24] = 1
    #         if fine_pred[i, 5] != 1:  # not a serve
    #             for start, end in [[8, 10], [10, 16]]:
    #                 max_idx = np.argmax(fine_scores[i, start:end])
    #                 fine_pred[i, start + max_idx] = 1

    #     fine_pred = coarse_pred[:, np.newaxis] * fine_pred

    #     # element F1 scores
    #     for i in range(len(fine_pred)):
    #         for j in range(len(fine_pred[0])):
    #             if fine_pred[i, j] == 1 and sum(
    #                     fine_label[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 1:
    #                 f1_element[j, 0] += 1  # tp
    #             if fine_pred[i, j] == 1 and sum(
    #                     fine_label[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 0:
    #                 f1_element[j, 1] += 1  # fp
    #             if fine_pred[i, j] == 0 and sum(
    #                     fine_label[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 1:
    #                 f1_element[j, 2] += 1  # fn

    #     print_preds, print_gts = [], []
    #     # tp, fp, fn
    #     for i in range(len(fine_pred)):
    #         if coarse_label[i] == 1:
    #             print_gt = []
    #             for j in range(len(fine_pred[0])):
    #                 if fine_label[i, j] == 1:
    #                     print_gt.append(classes_inv[j + 1])
    #             print_gts.append('_'.join(print_gt))
    #         if coarse_pred[i] == 1:
    #             print_pred = []
    #             for j in range(len(fine_pred[0])):
    #                 if fine_pred[i, j] == 1:
    #                     print_pred.append(classes_inv[j + 1])
    #             print_preds.append('_'.join(print_pred))

    #     labels = [int(''.join(str(x) for x in row), 2) for row in fine_label]
    #     preds = [int(''.join(str(x) for x in row), 2) for row in fine_pred]
    #     preds = coarse_pred * preds

    #     # event F1 scores
    #     for i in range(len(preds)):
    #         if preds[i] > 0 and preds[i] in labels[max(0, i - delta):min(len(preds), i + delta + 1)]:
    #             if preds[i] not in f1_event:
    #                 f1_event[preds[i]] = [1, 0, 0]
    #             else:
    #                 f1_event[preds[i]][0] += 1
    #         if preds[i] > 0 and sum(labels[max(0, i - delta):min(len(preds), i + delta + 1)]) == 0:
    #             if preds[i] not in f1_event:
    #                 f1_event[preds[i]] = [0, 1, 0]
    #             else:
    #                 f1_event[preds[i]][1] += 1
    #         if labels[i] > 0 and labels[i] not in preds[max(0, i - delta):min(len(preds), i + delta + 1)]:
    #             if labels[i] not in f1_event:
    #                 f1_event[labels[i]] = [0, 0, 1]
    #             else:
    #                 f1_event[labels[i]][2] += 1

    #     gt = [k for k, g in groupby(labels) if k != 0]
    #     pred = [k for k, g in groupby(preds) if k != 0]

    #     # record error sequence
    #     if len(pred) == len(gt):
    #         for j in range(len(pred)):
    #             if pred[j] != gt[j]:
    #                 f.write(video + '\n')
    #                 f.write('->'.join(print_preds) + '\n')
    #                 f.write('->'.join(print_gts) + '\n')
    #                 f.write('\n')
    #                 break
    #     else:
    #         f.write(video + '\n')
    #         f.write('->'.join(print_preds) + '\n')
    #         f.write('->'.join(print_gts) + '\n')
    #         f.write('\n')

    #     edit_scores.append(edit_score(pred, gt))

    # f.close()

    # f1, count = 0, 0
    # for value in f1_event.values():
    #     if sum(value) == 0:
    #         continue
    #     precision = value[0] / (value[0] + value[1] + 1e-10)
    #     recall = value[0] / (value[0] + value[2] + 1e-10)
    #     f1 += 2 * precision * recall / (precision + recall + 1e-10)
    #     count += 1
    # f1 /= count

    # print('Mean F1 (event):', np.mean(f1))
    # print()

    # precision = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 1] + 1e-10)
    # recall = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 2] + 1e-10)
    # f1 = 2 * precision * recall / (precision + recall + 1e-10)
    # print('Mean F1 (element):', np.mean(f1))
    # print()

    # print('Edit score:', sum(edit_scores) / len(edit_scores))
    # return sum(edit_scores) / len(edit_scores)


def get_last_epoch(save_dir):
    max_epoch = -1
    for file_name in os.listdir(save_dir):
        if not file_name.startswith('optim_'):
            continue
        epoch = int(os.path.splitext(file_name)[0].split('optim_')[1])
        if epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def get_best_epoch_and_history(save_dir, criterion):
    data = load_json(os.path.join(save_dir, 'loss.json'))
    if criterion == 'edit':
        key = 'val_edit'
        best = max(data, key=lambda x: x[key])
    else:
        key = 'val'
        best = min(data, key=lambda x: x[key])
    return data, best['epoch'], best[key]


def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'elements.txt'))

    dataset_len = EPOCH_NUM_FRAMES // (args.clip_len * args.stride)
    dataset_kwargs = {
        'crop_dim': args.crop_dim, 'stride': args.stride
    }

    print('Dataset size:', dataset_len)
    train_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.clip_len, dataset_len, is_eval=False,
        **dataset_kwargs)
    train_data.print_info()
    val_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.clip_len, dataset_len // 4,
        **dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    if args.criterion == 'edit':
        # Only perform edit score evaluation during training if criterion is edit
        val_data_frames = ActionSeqVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.clip_len, crop_dim=args.crop_dim, stride=args.stride, overlap_len=0)

    return classes, train_data, val_data, None, val_data_frames


def load_from_save(
        args, model, optimizer, scaler, lr_scheduler
):
    assert args.save_dir is not None
    epoch = get_last_epoch(args.save_dir)

    print('Loading from epoch {}'.format(epoch))
    model.load(torch.load(os.path.join(
        args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))))

    if args.resume:
        opt_data = torch.load(os.path.join(
            args.save_dir, 'optim_{:03d}.pt'.format(epoch)))
        optimizer.load_state_dict(opt_data['optimizer_state_dict'])
        scaler.load_state_dict(opt_data['scaler_state_dict'])
        lr_scheduler.load_state_dict(opt_data['lr_state_dict'])

    losses, best_epoch, best_criterion = get_best_epoch_and_history(
        args.save_dir, args.criterion)
    return epoch, losses, best_epoch, best_criterion


def store_config(file_path, args, num_epochs, classes):
    config = {
        'dataset': args.dataset,
        'num_classes': len(classes),
        'feature_arch': args.feature_arch,
        'temporal_arch': args.temporal_arch,
        'use_ctx': args.use_ctx,
        'clip_len': args.clip_len,
        'batch_size': args.batch_size,
        'crop_dim': args.crop_dim,
        'window': args.window,
        'stride': args.stride,
        'num_epochs': num_epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.start_val_epoch,
        'gpu_parallel': args.gpu_parallel,
        'epoch_num_frames': EPOCH_NUM_FRAMES
    }
    store_json(file_path, config, pretty=True)


def get_num_train_workers(args):
    n = BASE_NUM_WORKERS * 2
    return min(os.cpu_count(), n)


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    if args.num_workers is not None:
        global BASE_NUM_WORKERS
        BASE_NUM_WORKERS = args.num_workers

    assert args.batch_size % args.acc_grad_iter == 0
    if args.start_val_epoch is None:
        args.start_val_epoch = args.num_epochs - BASE_NUM_VAL_EPOCHS
    if args.crop_dim <= 0:
        args.crop_dim = None

    classes, train_data, val_data, train_data_frames, val_data_frames = get_datasets(args)

    def worker_init_fn(id):
        random.seed(id + epoch * 100)
    loader_batch_size = args.batch_size // args.acc_grad_iter
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=get_num_train_workers(args),
        prefetch_factor=1, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=BASE_NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    model = F3Set(len(classes), args.feature_arch, args.temporal_arch, clip_len=args.clip_len, step=args.stride, 
                  window=args.window, use_ctx=args.use_ctx, multi_gpu=args.gpu_parallel)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    # Warmup schedule
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    num_epochs, lr_scheduler = get_lr_scheduler(
        args, optimizer, num_steps_per_epoch)

    losses = []
    best_epoch = None
    best_criterion = 0 if args.criterion == 'edit' else float('inf')
    best_loss, stop_criterion = float('inf'), 0

    epoch = 0
    if args.resume:
        epoch, losses, best_epoch, best_criterion = load_from_save(args, model, optimizer, scaler, lr_scheduler)
        epoch += 1

    # Write it to console
    store_config('/dev/stdout', args, num_epochs, classes)

    for epoch in range(epoch, num_epochs):
        train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter, epoch=epoch)
        val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter, epoch=epoch)
        print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
            epoch, train_loss, val_loss))

        val_edit = 0
        if args.criterion == 'loss':
            if val_loss < best_criterion:
                best_criterion = val_loss
                best_epoch = epoch
                print('New best epoch!')
        elif args.criterion == 'edit':
            if epoch >= args.start_val_epoch:
                val_edit = evaluate(model, val_data_frames, classes, window=args.window)
                if args.criterion == 'edit' and val_edit > best_criterion:
                    best_criterion = val_edit
                    best_epoch = epoch
                    print('New best epoch!')
        else:
            print('Unknown criterion:', args.criterion)

        losses.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss, 'val_edit': val_edit})
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            store_json(os.path.join(args.save_dir, 'loss.json'), losses,
                        pretty=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir,
                    'checkpoint_{:03d}.pt'.format(epoch)))
            clear_files(args.save_dir, r'optim_\d+\.pt')
            torch.save(
                {'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'lr_state_dict': lr_scheduler.state_dict()},
                os.path.join(args.save_dir,
                                'optim_{:03d}.pt'.format(epoch)))
            store_config(os.path.join(args.save_dir, 'config.json'),
                            args, num_epochs, classes)

    print('Best epoch: {}\n'.format(best_epoch))

    if args.save_dir is not None:
        model.load(torch.load(os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

        # Evaluate on hold out splits
        eval_splits = ['test']
        for split in eval_splits:
            split_path = os.path.join(
                'data', args.dataset, '{}.json'.format(split))
            if os.path.exists(split_path):
                split_data = ActionSeqVideoDataset(classes, split_path, args.frame_dir, args.clip_len, overlap_len=args.clip_len//2,
                                                   crop_dim=args.crop_dim, stride=args.stride)
                split_data.print_info()
                evaluate(model, split_data, classes, window=args.window)


if __name__ == '__main__':
    main(get_args())
