from typing import Optional

import torch
# torch.set_default_dtype(torch.float64)


from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv

from torch_geometric_autoscale.models import ScalableGNN
from torch_geometric_autoscale import SubgraphLoader


# ambershhek for logging
import logging
# A logger for this file
log = logging.getLogger(__name__)

import ipdb
import matplotlib.pyplot as plt
import time

from torch_sparse import masked_select

from torch.autograd.profiler import _disable_profiler


class GCN(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, linear: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        # ipdb.set_trace()
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device, in_channels=in_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual
        self.linear = linear
    

        self.lins = ModuleList()
        if linear:
            self.lins.append(Linear(in_channels, hidden_channels))
            self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0 and not linear:
                in_dim = in_channels
            if i == num_layers - 1 and not linear:
                out_dim = out_channels
            conv = GCNConv(in_dim, out_dim, normalize=False)
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)
            

        self.absolute_drift_values = [[] for _ in range(num_layers)]  # To store absolute drift values for each layer
        self.relative_drift_values = [[] for _ in range(num_layers)]  # To store relative drift values for each layer
        self.absolute_approx_error_values = [[] for _ in range(num_layers)]  # To store absolute approximation error values AFTER each layer
        self.relative_approx_error_values = [[] for _ in range(num_layers)]  # To store relative approximation error values AFTER each layer

    @property
    def reg_modules(self):
        if self.linear:
            return ModuleList(list(self.convs) + list(self.bns))
        else:
            return ModuleList(list(self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins if self.linear else self.convs[-1:]

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor,  drift_norm: int, aggregate_combined: bool = True, use_aggregation=True, *args) -> Tensor:

        if args[1].shape[0] == x.shape[0]:
            n_id = args[1]
        batch_size = args[0]


        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # linear layer
        if self.linear:
            x = self.lins[0](x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        time_push_and_pull_all = 0 

        # use original GCNConv layger to do aggregation forwarding
        if use_aggregation:
            
            in_batch_mask = (adj_t.storage.row() < batch_size) & (adj_t.storage.col() < batch_size)
            out_batch_mask = ~in_batch_mask
            combined_mask = in_batch_mask | out_batch_mask

            in_batch_adj_t = SparseTensor(
                row=adj_t.storage.row()[in_batch_mask],
                col=adj_t.storage.col()[in_batch_mask],
                value=adj_t.storage.value()[in_batch_mask] if adj_t.storage.value() is not None else None,
                sparse_sizes=(adj_t.size(0), adj_t.size(1))
            )

            combined_adj_t = SparseTensor(
                row=adj_t.storage.row()[combined_mask],
                col=adj_t.storage.col()[combined_mask],
                value=adj_t.storage.value()[combined_mask] if adj_t.storage.value() is not None else None,
                sparse_sizes=adj_t.sparse_sizes(),
                is_sorted=True,  # Use True to prevent re-ordering in SparseTensor constructor
                trust_data=True
            )
            

            if aggregate_combined:
                adj_t = combined_adj_t
            else:
                adj_t = in_batch_adj_t
            for layer_idx, (conv, bn, hist) in enumerate(zip(self.convs[:-1], self.bns, self.histories)):
                h = conv(x, adj_t)
                if self.batch_norm:
                    h = bn(h)
                if self.residual and h.size(-1) == x.size(-1):
                    h += x[:h.size(0)]
                x = h.relu_()

                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                x, returned_time = self.push_and_pull(self.histories[layer_idx+1], x, batch_size=args[0], n_id=args[1].to(hist.emb.device), offset=args[2], count=args[3])
                end_time.record()

                torch.cuda.synchronize()
                time_push_and_pull = start_time.elapsed_time(end_time) / 1000  # Convert milliseconds to seconds

                time_push_and_pull_all += time_push_and_pull
                               
                x = F.dropout(x, p=self.dropout, training=self.training)
  
            h = self.convs[-1](x, adj_t)


        # not use the aggregation function in GCNConv, degrade to MLP
        else:
            log.info(f'Not using aggregation... GCNConv degraded to MLP!')
            # only keep current batch
            x = x[:batch_size]
            
            for layer_idx, (conv, bn, hist) in enumerate(zip(self.convs[:-1], self.bns, self.histories)):

                h = conv.lin(x)

                if conv.bias is not None:
                    h = h + conv.bias
                
                if self.batch_norm:
                    h = bn(h)
                if self.residual and h.size(-1) == x.size(-1):
                    h += x[:h.size(0)]
                x = h.relu_()

                x, time_push_and_pull = self.push_and_pull(hist, x, batch_size=args[0], n_id=args[1].to(hist.emb.device)[:batch_size], offset=args[2], count=args[3])
                time_push_and_pull_all += time_push_and_pull
                
                x = F.dropout(x, p=self.dropout, training=self.training)

            h = self.convs[-1].lin(x)
            if self.convs[-1].bias is not None:
                h = h + self.convs[-1].bias
            

        if not self.linear:
            return h, time_push_and_pull_all

        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lins[1](h), time_push_and_pull_all


    ### used for efficiecy test 1230
    def VR_forward(self, x: Tensor, adj_t: SparseTensor, drift_norm: int, epoch: int, batch_idx: int,  *args) -> Tensor:
        """
        Perform variance reduction forward pass.

        Args:
            x (Tensor): Input node features.
            adj_t (SparseTensor): Sparse adjacency tensor.
            drift_norm (int): Norm type for drift calculation.
            *args: Additional arguments.

        Returns:
            Tensor: Output tensor after forward pass.
        """
        batch_size = args[0]   

        ### dropout
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        ### linear layer
        if self.linear:
            x = self.lins[0](x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        time_data_movement = 0

        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = x[:batch_size]

            period_hist_selected_new = self.pool.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
            period_hist_aggr_selected_new = self.pool_ag.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()

            h = adj_t @ (x - period_hist_selected_new) + period_hist_aggr_selected_new
            h = conv.lin(h)
            if conv.bias is not None:
                h = h + conv.bias
            self.pool.free_pull()
            self.pool_ag.free_pull()
            if self.batch_norm: # True in default settings of GCN on arxiv
                h = bn(h) 
            if self.residual and h.size(-1) == x.size(-1): # False in default settings of GCN on arxiv
                h += x[:h.size(0)] 
            x = h.relu_() 

            time_data_movement += 0

            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x[:batch_size]

        period_hist_selected_new = self.pool.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
        period_hist_aggr_selected_new = self.pool_ag.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()

        h = adj_t @ (x - period_hist_selected_new) + period_hist_aggr_selected_new
        h = self.convs[-1].lin(h) # 使用VR得到的embeddings
        h = h + self.convs[-1].bias

        self.pool.free_pull()
        self.pool_ag.free_pull()

        if not self.linear:
            return h, time_data_movement, 0, 0

        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        # return self.lins[1](h)
        return self.lins[1](h), time_data_movement


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state, use_aggregation=True):
    # def forward_layer(self, layer, x, adj_t): # remove unused arg 'state'
        if use_aggregation: # GCN
            if layer == 0:
                if self.drop_input:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                if self.linear:
                    x = self.lins[0](x).relu_()
                    x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)

            # ipdb.set_trace()
            h = self.convs[layer](x, adj_t)

            if layer < self.num_layers - 1 or self.linear:
                if self.batch_norm:
                    h = self.bns[layer](h)
                if self.residual and h.size(-1) == x.size(-1):
                    h += x[:h.size(0)]
                h = h.relu_()

            if self.linear:
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = self.lins[1](h)
        else: # do NOT use aggregation, degrade to MLP
            log.info(f'[forward_layer] Not using aggregation... GCNConv degraded to MLP!')
            if layer == 0:
                if self.drop_input:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                if self.linear:
                    x = self.lins[0](x).relu_()
                    x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)

            h = self.convs[layer].lin(x)

            if layer < self.num_layers - 1 or self.linear:
                if self.batch_norm:
                    h = self.bns[layer](h)
                if self.residual and h.size(-1) == x.size(-1):
                    h += x[:h.size(0)]
                h = h.relu_()

            if self.linear:
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = self.lins[1](h)

        return h


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

            M_in_layer0 = x[:batch_size] # only keep the batch_size rows
            if M_in_layer0.size(1) < self.hidden_channels:
                M_in_layer0_expanded = torch.zeros((M_in_layer0.size(0), self.hidden_channels), device=self.device)
                M_in_layer0_expanded[:, :M_in_layer0.size(1)] = M_in_layer0
                M_in_layer0 = M_in_layer0_expanded

            M_ag_layer0 = adj_t @ x # shape from (IB+OB) to IB, reddit报错的地方
            if M_ag_layer0.size(1) < self.hidden_channels:
                M_ag_layer0_expanded = torch.zeros((M_ag_layer0.size(0), self.hidden_channels), device=self.device)
                M_ag_layer0_expanded[:, :M_ag_layer0.size(1)] = M_ag_layer0
                M_ag_layer0 = M_ag_layer0_expanded

            self.pool.async_push(M_in_layer0, offset, count, self.histories[0].emb) # push x to history
            self.pool_ag.async_push(M_ag_layer0, offset, count, self.histories_ag[0].emb)
            self.pool.async_push(out, offset, count, self.histories[1].emb) # index change

            torch.cuda.synchronize()

        self.pool.synchronize_push()
        self.pool_ag.synchronize_push()

        for i in range(1, len(self.histories)-1): # index change
            # Pull the complete layer-wise history:
            for _, batch_size, n_id, offset, count, _ in loader:
                self.pool.async_pull(self.histories[i].emb, offset, count, n_id[batch_size:])

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

        for batch, batch_size, n_id, offset, count, state in loader:
            adj_t = batch.adj_t.to(self.device)
            x = self.pool.synchronize_pull()[:n_id.numel()]
            out = self.forward_layer(self.num_layers - 1, x, adj_t, state, use_aggregation)[:batch_size]
            M_ag = adj_t @ x
            self.pool_ag.async_push(M_ag, offset, count, self.histories_ag[-1].emb)
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
        self.pool_ag.synchronize_push()
        self.pool.synchronize_push()

        return self._out
    
