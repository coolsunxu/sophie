import os
import math
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, Dataset

from constants import *

def read_file(path, delim='\t'):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def collate(data):
    obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, vgg_list = zip(*data)
    obs_seq = torch.cat(obs_seq, dim=0).permute(3, 0, 1, 2)
    pred_seq = torch.cat(pred_seq, dim=0).permute(2, 0, 1)
    obs_seq_rel = torch.cat(obs_seq_rel, dim=0).permute(3, 0, 1, 2)
    pred_seq_rel = torch.cat(pred_seq_rel, dim=0).permute(2, 0, 1)
    vgg_list = torch.cat(vgg_list, dim=0).repeat(obs_seq.size(1), 1, 1)
    return tuple([obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, vgg_list])

def data_loader(path):
    dset = TrajDataset(path)
    loader = DataLoader(dset, 1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate)
    return dset, loader

class TrajDataset(Dataset):
    def __init__(self, data_dir):

        super(TrajDataset, self).__init__()
        all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path[0] != "." and path.endswith(".txt")]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        seq_len = OBS_LEN + PRED_LEN
        fet_map = {}
        fet_list = []

        for path in all_files:
            data = read_file(path)
            frames = np.unique(data[:, 0]).tolist()

            hkl_path = os.path.splitext(path)[0] + ".pkl"
            with open(hkl_path, 'rb') as handle:
                new_fet = pickle.load(handle, encoding='bytes')
            fet_map[hkl_path] = torch.from_numpy(new_fet)

            frame_data = [data[frame == data[:, 0], :] for frame in frames]
            num_sequences = len(frames) - seq_len + 1
            for idx in range(0, num_sequences+1):

                curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))
                
                num_peds_considered = 0
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    curr_seq[num_peds_considered, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[num_peds_considered, :, pad_front:pad_end] = rel_curr_ped_seq
                    num_peds_considered += 1

                if num_peds_considered > 1:
                    num_peds_in_seq.append(num_peds_considered)

                    curr_seq_exp = np.zeros((num_peds_considered, MAX_PEDS, 2, seq_len))
                    curr_seq_rel_exp = np.zeros((num_peds_considered, MAX_PEDS, 2, seq_len))
                    for i in range(num_peds_considered):
                        curr_seq_exp[i, 0, :, :] = curr_seq[i]
                        curr_seq_exp[i, 1:i+1, :, :] = curr_seq[0:i]
                        curr_seq_exp[i, i+1:num_peds_considered, :, :] = curr_seq[i+1:num_peds_considered]

                        dists = (curr_seq_exp[i, :] - curr_seq_exp[i, 0]) ** 2
                        dists = np.sum(np.sum(dists, axis=2), axis=1)
                        idxs = np.argsort(dists)
                        curr_seq_exp[i, :] = curr_seq_exp[i, :][idxs]

                        curr_seq_rel_exp[i, 0, :, :] = curr_seq_rel[i]
                        curr_seq_rel_exp[i, 1:i+1, :, :] = curr_seq_rel[0:i]
                        curr_seq_rel_exp[i, i+1:num_peds_considered, :, :] = curr_seq_rel[i+1:num_peds_considered]
                        curr_seq_rel_exp[i, :] = curr_seq_rel_exp[i, :][idxs]

                    seq_list.append(curr_seq_exp[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel_exp[:num_peds_considered])
                    fet_list.append(hkl_path)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :, :OBS_LEN]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, 0, :, OBS_LEN:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :, :OBS_LEN]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, 0, :, OBS_LEN:]).type(torch.float)

        self.fet_map = fet_map
        self.fet_list = fet_list

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.fet_map[self.fet_list[index]]
        ]
        return out
