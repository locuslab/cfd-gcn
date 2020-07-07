import os
import logging
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.unpool import knn_interpolate

from su2torch import SU2Module
from mesh_utils import write_graph_mesh, quad2tri, get_mesh_graph, signed_dist_graph, is_cw


class MeshGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6, improved=False,
                 cached=False, bias=True, fine_marker_dict=None):
        super().__init__()
        self.fine_marker_dict = torch.tensor(fine_marker_dict['airfoil']).unique()
        self.sdf = None
        in_channels += 1  # account for sdf

        channels = [in_channels]
        channels += [hidden_channels] * (num_layers - 1)
        channels.append(out_channels)

        convs = []
        for i in range(num_layers):
            convs.append(GCNConv(channels[i], channels[i+1], improved=improved,
                                 cached=cached, bias=bias))
        self.convs = nn.ModuleList(convs)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch_size = data.aoa.shape[0]

        if self.sdf is None:
            with torch.no_grad():
                self.sdf = signed_dist_graph(x[data.batch == 0, :2],
                                             self.fine_marker_dict).unsqueeze(1)
        x = torch.cat([data.x, self.sdf.repeat(batch_size, 1)], dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index)
        return x


class CFDGCN(nn.Module):
    def __init__(self, config_file, coarse_mesh, fine_marker_dict, process_sim=lambda x, y: x,
                 freeze_mesh=False, num_convs=6, num_end_convs=3, hidden_channels=512,
                 out_channels=3, device='cuda'):
        super().__init__()
        meshes_temp_dir = 'temp_meshes'
        os.makedirs(meshes_temp_dir, exist_ok=True)
        self.mesh_file = meshes_temp_dir + '/' + str(os.getpid()) + '_mesh.su2'

        if not coarse_mesh:
            raise ValueError('Need to provide a coarse mesh for CFD-GCN.')
        nodes, edges, self.elems, self.marker_dict = get_mesh_graph(coarse_mesh)
        self.nodes = torch.from_numpy(nodes).to(device)
        if not freeze_mesh:
            self.nodes = nn.Parameter(self.nodes)
        self.elems, new_edges = quad2tri(sum(self.elems, []))
        self.elems = [self.elems]
        self.edges = torch.from_numpy(edges).to(device)
        print(self.edges.dtype, new_edges.dtype)
        self.edges = torch.cat([self.edges, new_edges.to(self.edges.device)], dim=1)
        self.marker_inds = torch.tensor(sum(self.marker_dict.values(), [])).unique()
        assert is_cw(self.nodes, self.elems[0]).nonzero().shape[0] == 0, 'Mesh has flipped elems'

        self.process_sim = process_sim
        self.su2 = SU2Module(config_file, mesh_file=self.mesh_file)
        logging.info(f'Mesh filename: {self.mesh_file.format(batch_index="*")}')

        self.fine_marker_dict = torch.tensor(fine_marker_dict['airfoil']).unique()
        self.sdf = None

        improved = False
        self.num_convs = num_end_convs
        self.convs = []
        if self.num_convs > 0:
            self.convs = nn.ModuleList()
            in_channels = out_channels + hidden_channels
            for i in range(self.num_convs - 1):
                self.convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
                in_channels = hidden_channels
            self.convs.append(GCNConv(in_channels, out_channels, improved=improved))

        self.num_pre_convs = num_convs - num_end_convs
        self.pre_convs = []
        if self.num_pre_convs > 0:
            in_channels = 5 + 1  # one extra channel for sdf
            self.pre_convs = nn.ModuleList()
            for i in range(self.num_pre_convs - 1):
                self.pre_convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
                in_channels = hidden_channels
            self.pre_convs.append(GCNConv(in_channels, hidden_channels, improved=improved))

        self.sim_info = {}  # store output of coarse simulation for logging / debugging

    def forward(self, batch):
        start = time.time()
        batch_size = batch.aoa.shape[0]

        if self.sdf is None:
            with torch.no_grad():
                self.sdf = signed_dist_graph(batch.x[batch.batch == 0, :2],
                                             self.fine_marker_dict).unsqueeze(1)
        fine_x = torch.cat([batch.x, self.sdf.repeat(batch_size, 1)], dim=1)

        for i, conv in enumerate(self.pre_convs):
            fine_x = F.relu(conv(fine_x, batch.edge_index))

        nodes = self.get_nodes()
        num_nodes = nodes.shape[0]
        self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
        batch_x = batch_x.to('cpu', non_blocking=True)
        batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
                           batch_aoa[..., None], batch_mach_or_reynolds[..., None])
        batch_y = [y.to(batch.x.device) for y in batch_y]
        batch_y = self.process_sim(batch_y, False)

        coarse_y = torch.stack([y.flatten() for y in batch_y], dim=1)
        coarse_x = nodes.repeat(batch_size, 1)[:, :2]
        zeros = batch.batch.new_zeros(num_nodes)
        coarse_batch = torch.cat([zeros + i for i in range(batch_size)])

        fine_y = self.upsample(coarse_y, coarse_x, coarse_batch, batch)
        fine_y = torch.cat([fine_y, fine_x], dim=1)

        for i, conv in enumerate(self.convs[:-1]):
            fine_y = F.relu(conv(fine_y, batch.edge_index))
        fine_y = self.convs[-1](fine_y, batch.edge_index)

        self.sim_info['nodes'] = coarse_x[:, :2]
        self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return fine_y

    def upsample(self, y, coarse_nodes, coarse_batch, fine):
        fine_nodes = fine.x[:, :2]
        y = knn_interpolate(y.cpu(), coarse_nodes[:, :2].cpu(), fine_nodes.cpu(),
                            coarse_batch.cpu(), fine.batch.cpu(), k=3).to(y.device)
        return y

    def get_nodes(self):
        # return torch.cat([self.marker_nodes, self.not_marker_nodes])
        return self.nodes

    @staticmethod
    def write_mesh_file(x, elems, marker_dict, filename='mesh.su2'):
        write_graph_mesh(filename, x[:, :2], elems, marker_dict)

    @staticmethod
    def contiguous_elems_list(elems, inds):
        # Hack to easily have compatibility with MeshEdgePool
        return elems


class UCM(CFDGCN):
    """Simply upsamples the coarse simulation without using any GCNs."""
    def __init__(self, config_file, coarse_mesh, fine_marker_dict, process_sim=lambda x, y: x,
                 freeze_mesh=False, device='cuda'):
        super().__init__(config_file, coarse_mesh, fine_marker_dict, process_sim=process_sim,
                         freeze_mesh=freeze_mesh, num_convs=0, num_end_convs=0, device=device)

    def forward(self, batch):
        batch_size = batch.aoa.shape[0]

        nodes = self.get_nodes()
        num_nodes = nodes.shape[0]
        self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
        batch_x = batch_x.to('cpu', non_blocking=True)
        batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
                           batch_aoa[..., None], batch_mach_or_reynolds[..., None])
        batch_y = [y.to(batch.x.device) for y in batch_y]
        batch_y = self.process_sim(batch_y, False)

        coarse_y = torch.stack([y.flatten() for y in batch_y], dim=1)
        coarse_x = nodes.repeat(batch_size, 1)[:, :2]
        zeros = batch.batch.new_zeros(num_nodes)
        coarse_batch = torch.cat([zeros + i for i in range(batch_size)])

        fine_y = self.upsample(coarse_y, coarse_x, coarse_batch, batch)

        self.sim_info['nodes'] = coarse_x[:, :2]
        self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return fine_y


class CFD(CFDGCN):
    """Simply outputs the results of the (fine) CFD simulation."""
    def __init__(self, config_file, mesh, fine_marker_dict, process_sim=lambda x, y: x,
                 freeze_mesh=False, device='cuda'):
        super().__init__(config_file, mesh, fine_marker_dict, process_sim=process_sim,
                         freeze_mesh=freeze_mesh, num_convs=0, num_end_convs=0, device=device)

    def forward(self, batch):
        batch_size = batch.aoa.shape[0]

        nodes = self.get_nodes()
        num_nodes = nodes.shape[0]
        self.write_mesh_file(nodes, self.elems, self.marker_dict, filename=self.mesh_file)

        params = torch.stack([batch.aoa, batch.mach_or_reynolds], dim=1)
        batch_aoa = params[:, 0].to('cpu', non_blocking=True)
        batch_mach_or_reynolds = params[:, 1].to('cpu', non_blocking=True)

        batch_x = nodes.unsqueeze(0).expand(batch_size, -1, -1)
        batch_x = batch_x.to('cpu', non_blocking=True)
        batch_y = self.su2(batch_x[..., 0], batch_x[..., 1],
                           batch_aoa[..., None], batch_mach_or_reynolds[..., None])
        batch_y = [y.to(batch.x.device) for y in batch_y]
        batch_y = self.process_sim(batch_y, False)

        coarse_y = torch.stack([y.flatten() for y in batch_y], dim=1)
        coarse_x = nodes.repeat(batch_size, 1)[:, :2]
        zeros = batch.batch.new_zeros(num_nodes)
        coarse_batch = torch.cat([zeros + i for i in range(batch_size)])

        fine_y = coarse_y

        self.sim_info['nodes'] = coarse_x[:, :2]
        self.sim_info['elems'] = [self.elems] * batch_size
        self.sim_info['batch'] = coarse_batch
        self.sim_info['output'] = coarse_y

        return fine_y

