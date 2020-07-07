import os
import pickle
from pathlib import Path

import numpy as np

import torch
from torch._six import container_abcs, string_classes, int_classes
from torch_geometric.data import Data, Batch, Dataset

from mesh_utils import get_mesh_graph


class MeshAirfoilDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.mode = mode
        self.data_dir = Path(root) / ('outputs_' + mode)
        self.file_list = os.listdir(self.data_dir)
        self.len = len(self.file_list)

        self.mesh_graph = get_mesh_graph(Path(root) / 'mesh_fine.su2')

        # either [maxes, mins] or [means, stds] from data for normalization
        # with open(self.data_dir / 'train_mean_std.pkl', 'rb') as f:
        with open(self.data_dir.parent / 'train_max_min.pkl', 'rb') as f:
            self.normalization_factors = pickle.load(f)

        self.nodes = torch.from_numpy(self.mesh_graph[0])
        self.edges = torch.from_numpy(self.mesh_graph[1])
        self.elems_list = self.mesh_graph[2]
        self.marker_dict = self.mesh_graph[3]
        self.node_markers = self.nodes.new_full((self.nodes.shape[0], 1), fill_value=-1)
        for i, (marker_tag, marker_elems) in enumerate(self.marker_dict.items()):
            for elem in marker_elems:
                self.node_markers[elem[0]] = i
                self.node_markers[elem[1]] = i

        super().__init__(root)

    def __len__(self):
        return self.len

    def get(self, idx):
        with open(self.data_dir / self.file_list[idx], 'rb') as f:
            fields = pickle.load(f)
        fields = self.preprocess(fields)

        aoa, reynolds, mach = self.get_params_from_name(self.file_list[idx])
        aoa = aoa
        aoa = torch.from_numpy(aoa)
        mach_or_reynolds = mach if reynolds is None else reynolds
        mach_or_reynolds = torch.from_numpy(mach_or_reynolds)

        norm_aoa = aoa / 10
        norm_mach_or_reynolds = mach_or_reynolds if reynolds is None else (mach_or_reynolds - 1.5e6) / 1.5e6

        # add physics parameters to graph
        nodes = torch.cat([
            self.nodes,
            norm_aoa.unsqueeze(0).repeat(self.nodes.shape[0], 1),
            norm_mach_or_reynolds.unsqueeze(0).repeat(self.nodes.shape[0], 1),
            self.node_markers
        ], dim=-1)

        data = Data(x=nodes, y=fields, edge_index=self.edges)
        data.aoa = aoa
        data.norm_aoa = norm_aoa
        data.mach_or_reynolds = mach_or_reynolds
        data.norm_mach_or_reynolds = norm_mach_or_reynolds
        return data

    def preprocess(self, tensor_list, stack_output=True):
        # data_means, data_stds = self.normalization_factors
        data_max, data_min = self.normalization_factors
        normalized_tensors = []
        for i in range(len(tensor_list)):
            # tensor_list[i] = (tensor_list[i] - data_means[i]) / data_stds[i] / 10
            normalized = (tensor_list[i] - data_min[i]) / (data_max[i] - data_min[i]) * 2 - 1
            if type(normalized) is np.ndarray:
                normalized = torch.from_numpy(normalized)
            normalized_tensors.append(normalized)
        if stack_output:
            normalized_tensors = torch.stack(normalized_tensors, dim=1)
        return normalized_tensors

    def _download(self):
        pass

    def _process(self):
        pass

    @staticmethod
    def get_params_from_name(filename):
        s = filename.rsplit('.', 1)[0].split('_')
        aoa = np.array(s[s.index('aoa') + 1])[np.newaxis].astype(np.float32)
        reynolds = s[s.index('re') + 1]
        reynolds = np.array(reynolds)[np.newaxis].astype(np.float32) if reynolds != 'None' else None
        mach = np.array(s[s.index('mach') + 1])[np.newaxis].astype(np.float32)
        return aoa, reynolds, mach
