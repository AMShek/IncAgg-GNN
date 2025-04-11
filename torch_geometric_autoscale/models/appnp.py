from typing import Optional

import ipdb.stdout
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_sparse import SparseTensor

from torch_geometric_autoscale.models import ScalableGNN
from torch_geometric_autoscale import SubgraphLoader

import logging
log = logging.getLogger(__name__)

import ipdb


class APPNP(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float,
                 dropout: float = 0.0, pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, out_channels, num_layers, pool_size,
                         buffer_size, device, in_channels=in_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.dropout = dropout

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.reg_modules = self.lins[:1]
        self.nonreg_modules = self.lins[1:]

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, drift_norm: int, aggregate_combined: bool = True, use_aggregation=True, *args) -> Tensor:
        time_push_and_pull_all = 0 

        n_id = args[1]
        n_id = n_id.to(x.device)
        batch_size = args[0]

        if use_aggregation:
            in_batch_mask = (adj_t.storage.row() < batch_size) & (adj_t.storage.col() < batch_size)
            out_batch_mask = ~in_batch_mask
            combined_mask = in_batch_mask | out_batch_mask

            in_batch_adj = SparseTensor(
                row=adj_t.storage.row()[in_batch_mask],
                col=adj_t.storage.col()[in_batch_mask],
                value=adj_t.storage.value()[in_batch_mask] if adj_t.storage.value() is not None else None,
                sparse_sizes=(adj_t.size(0), adj_t.size(1))
            )

            combined_adj = SparseTensor(
                row=adj_t.storage.row()[combined_mask],
                col=adj_t.storage.col()[combined_mask],
                value=adj_t.storage.value()[combined_mask] if adj_t.storage.value() is not None else None,
                sparse_sizes=adj_t.sparse_sizes(),
                is_sorted=True,  # Use True to prevent re-ordering in SparseTensor constructor
                trust_data=True
            )

            if aggregate_combined:
                adj_t = combined_adj
            else:
                adj_t = in_batch_adj

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[0](x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[1](x)
            x_0 = x[:adj_t.size(0)]

            for history in self.histories:
                x = (1 - self.alpha) * (adj_t @ x) + self.alpha * x_0
                x, time_push_and_pull = self.push_and_pull(history, x, batch_size=args[0], n_id=args[1].to(history.emb.device), offset=args[2], count=args[3])
                time_push_and_pull_all += time_push_and_pull

            x = (1 - self.alpha) * (adj_t @ x) + self.alpha * x_0
        else:
            x = x[:batch_size]
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[0](x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[1](x)
            x_0 = x[:adj_t.size(0)]

            for history in self.histories:
                x = (1 - self.alpha) * x + self.alpha * x_0
                x, time_push_and_pull = self.push_and_pull(history, x, batch_size=args[0], n_id=args[1].to(history.emb.device)[:batch_size], offset=args[2], count=args[3])
                time_push_and_pull_all += time_push_and_pull

            x = (1 - self.alpha) * x + self.alpha * x_0

        return x, time_push_and_pull_all

    def VR_forward(self, x: Tensor, adj_t: SparseTensor, drift_norm: int, epoch: int, batch_idx: int, *args) -> Tensor:
        batch_size = args[0]
        x = x[:batch_size]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[0](x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[1](x)
        x_0 = x[:adj_t.size(0)]

        for i in range(self.num_layers-1):
            period_hist_selected_new = self.pool.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
            period_hist_aggr_selected_new = self.pool_ag.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()            

            x_vr = adj_t@(x-period_hist_selected_new) + period_hist_aggr_selected_new
            x = (1 - self.alpha) * x_vr + self.alpha * x_0

            self.pool.free_pull()
            self.pool_ag.free_pull()

        period_hist_selected_new = self.pool.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
        period_hist_aggr_selected_new = self.pool_ag.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
        x_vr = adj_t@(x-period_hist_selected_new) + period_hist_aggr_selected_new
        x = (1 - self.alpha) * x_vr + self.alpha * x_0

        ### free the buffer space for current batch and layer of M_in and M_ag
        self.pool.free_pull()
        self.pool_ag.free_pull()
        
        return x, 0, 0, 0


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state, use_aggregation=True):
        if use_aggregation: 
            if layer == 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                # ipdb.set_trace()
                x = self.lins[0](x)
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x_0 = self.lins[1](x)
                state['x_0'] = x_0[:adj_t.size(0)]

            x = (1 - self.alpha) * (adj_t @ x) + self.alpha * state['x_0']
            return x
        else: 
            x = x[:adj_t.size(0)]

            if layer == 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.lins[0](x)
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x_0 = self.lins[1](x)
                state['x_0'] = x_0[:adj_t.size(0)]
            
            x = (1 - self.alpha) * x + self.alpha * state['x_0']
            return x

    @torch.no_grad()
    def initialize_period_hist(self, A, node_features):
        log.info(f'Initializing M_in and M_ag')
        hist_device = self.histories[0].pull().device
        A = A.to(hist_device)

        # Initialize period_hist and period_hist_aggr as lists to allow different dimensions
        self.period_hist = []
        self.period_hist_aggr = []

        # Assign raw node features to period_hist[0]
        raw_hist = node_features.to(hist_device).clone().detach()
        raw_hist = self.lins[0](raw_hist)
        raw_hist = raw_hist.relu()
        raw_hist = self.lins[1](raw_hist)

        aggr_hist_0 = A @ raw_hist
        aggr_hist_0 = aggr_hist_0.clone().detach()
        # ipdb.set_trace()

        self.period_hist.append(raw_hist)  # Add raw features
        self.period_hist_aggr.append(aggr_hist_0)  # Add aggregated raw features

        # Add features from histories with varying dimensions
        for i, history in enumerate(self.histories):
            hist = history.pull().to(hist_device).clone().detach()
            aggr_hist = A @ hist
            aggr_hist = aggr_hist.clone().detach()

            self.period_hist.append(hist)
            self.period_hist_aggr.append(aggr_hist)


    @torch.no_grad()
    def update_period_hist(self, A, node_features=None):
        log.info(f'Updating M_in and calculating M_ag')
        hist_device = self.histories[0].pull().device
        A = A.to(hist_device)

        # if raw features are inputed, update layer[0] periodical memory
        if node_features != None:
            raw_hist = node_features.to(hist_device).clone().detach()
            raw_hist = self.lins[0](raw_hist)
            raw_hist = raw_hist.relu()
            raw_hist = self.lins[1](raw_hist)
            aggr_hist_0 = A @ raw_hist
            aggr_hist_0 = aggr_hist_0.clone().detach()
            self.period_hist[0] = raw_hist
            self.period_hist_aggr[0] = aggr_hist_0

        # Update the histories with varying dimensions while preserving the raw features
        for i, history in enumerate(self.histories):
            hist = history.pull().to(hist_device).clone().detach()
            aggr_hist = A @ hist
            aggr_hist = aggr_hist.clone().detach()

            self.period_hist[i + 1] = hist
            self.period_hist_aggr[i + 1] = aggr_hist


    @torch.no_grad()
    def mini_inference_vr(self, loader: SubgraphLoader, use_aggregation = True) -> Tensor:    
        r"""An implementation of layer-wise evaluation of GNNs.
        For each individual layer and mini-batch, :meth:`forward_layer` takes
        care of computing the next state of node embeddings.
        Additional state (such as residual connections) can be stored in
        a `state` directory."""

        # We iterate over the loader in a layer-wise fashsion.
        # In order to re-use some intermediate representations, we maintain a
        # `state` dictionary for each individual mini-batch.
        # full_loader = loader
        loader = [sub_data + ({}, ) for sub_data in loader]

        # We push the outputs of the first layer to the history:
        for data, batch_size, n_id, offset, count, state in loader:
            x = data.x.to(self.device)
            adj_t = data.adj_t.to(self.device)
            out = self.forward_layer(0, x, adj_t, state, use_aggregation)[:batch_size]
            self.pool.async_push(out, offset, count, self.histories[1].emb) # index change

            M_in_layer0 = self.lins[0](x)
            M_in_layer0 = M_in_layer0.relu()
            M_in_layer0 = self.lins[1](M_in_layer0)

            M_ag_layer0 = adj_t @ M_in_layer0 # shape from (IB+OB) to IB
            if M_ag_layer0.size(1) < self.hidden_channels:
                M_ag_layer0_expanded = torch.zeros((M_ag_layer0.size(0), self.hidden_channels), device=self.device)
                M_ag_layer0_expanded[:, :M_ag_layer0.size(1)] = M_ag_layer0
                M_ag_layer0 = M_ag_layer0_expanded
            # ipdb.set_trace()
            self.pool_ag.async_push(M_ag_layer0, offset, count, self.histories_ag[0].emb)

            M_in_layer0 = M_in_layer0[:batch_size] # only keep the batch_size rows
            if M_in_layer0.size(1) < self.hidden_channels:
                M_in_layer0_expanded = torch.zeros((M_in_layer0.size(0), self.hidden_channels), device=self.device)
                M_in_layer0_expanded[:, :M_in_layer0.size(1)] = M_in_layer0
                M_in_layer0 = M_in_layer0_expanded
            self.pool.async_push(M_in_layer0, offset, count, self.histories[0].emb) # push x to history


        self.pool.synchronize_push()
        self.pool_ag.synchronize_push()

        for i in range(1, len(self.histories)-1): # index change
            # Pull the complete layer-wise history:
            for _, batch_size, n_id, offset, count, _ in loader:
                self.pool.async_pull(self.histories[i].emb, offset, count, n_id[batch_size:]) # index change
                # self.pool_ag.async_pull(self.histories_ag[i].emb, None, None, torch.empty(0)) # for debug

            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for batch, batch_size, n_id, offset, count, state in loader:
                adj_t = batch.adj_t.to(self.device)
                x = self.pool.synchronize_pull()[:n_id.numel()]

                M_ag = adj_t @ x
                self.pool_ag.async_push(M_ag, offset, count, self.histories_ag[i].emb)

                out = self.forward_layer(i, x, adj_t, state, use_aggregation)[:batch_size]
                self.pool.async_push(out, offset, count, self.histories[i+1].emb) # index change
                
                self.pool.free_pull()
            self.pool.synchronize_push()
            self.pool_ag.synchronize_push()

        # We pull the histories from the last layer:
        for _, batch_size, n_id, offset, count, _ in loader:
            self.pool.async_pull(self.histories[-1].emb, offset, count, n_id[batch_size:])
            # self.pool_ag.async_pull(self.histories_ag[-1].emb, None, None, torch.empty(0)) # for debug

        # And compute final output embeddings, which we write into a private
        # output embedding matrix:
        for batch, batch_size, n_id, offset, count, state in loader:
            adj_t = batch.adj_t.to(self.device)
            x = self.pool.synchronize_pull()[:n_id.numel()]
            # x_useless = self.pool_ag.synchronize_pull() # for debug
            out = self.forward_layer(self.num_layers - 1, x, adj_t, state, use_aggregation)[:batch_size]
            M_ag = adj_t @ x
            self.pool_ag.async_push(M_ag, offset, count, self.histories_ag[-1].emb)
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
            # self.pool_ag.free_pull() # for debug
        self.pool_ag.synchronize_push()
        self.pool.synchronize_push()

        return self._out