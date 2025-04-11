from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCN2Conv

from torch_geometric_autoscale.models import ScalableGNN

import ipdb
import matplotlib.pyplot as plt

from torch_geometric_autoscale import SubgraphLoader

# ambershhek for logging
import logging
# A logger for this file
log = logging.getLogger(__name__)

class GCN2(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int, alpha: float,
                 theta: float, shared_weights: bool = True,
                 dropout: float = 0.0, drop_input: bool = True,
                 batch_norm: bool = False, residual: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device, in_channels=in_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            conv = GCN2Conv(hidden_channels, alpha=alpha, theta=theta,
                            layer=i + 1, shared_weights=shared_weights,
                            normalize=False)
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
        return ModuleList(list(self.convs) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, drift_norm: int, aggregate_combined: bool = True, use_aggregation=True, *args) -> Tensor:
        if args[1].shape[0] == x.shape[0]:
            n_id = args[1]
        n_id = n_id.to(x.device)
        batch_size = args[0]

        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x_0 = self.lins[0](x).relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)


        time_push_and_pull_all = 0

        # use original GCNConv layger to do aggregation forwarding
        if use_aggregation:
            # masks for IB and OB
            in_batch_mask = (adj_t.storage.row() < batch_size) & (adj_t.storage.col() < batch_size)
            out_batch_mask = ~in_batch_mask
            # Combine in_batch_mask and out_batch_mask directly
            combined_mask = in_batch_mask | out_batch_mask

            # Count in-batch and out-of-batch neighbors
            # num_in_batch_neighbors = in_batch_mask.sum().item()
            # num_out_batch_neighbors = out_batch_mask.sum().item()
            # log.info(f'Number of in-batch neighbors: {num_in_batch_neighbors}, Number of out-of-batch neighbors: {num_out_batch_neighbors}')

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
                # log.info(f'Use ib & ob neighbors!')
            else:
                adj_t = in_batch_adj
                # log.info(f'Discard out-of-batch neighbors!')

            for conv, bn, hist in zip(self.convs[:-1], self.bns[:-1], self.histories):
                h = conv(x, x_0, adj_t)
                if self.batch_norm:
                    h = bn(h)
                if self.residual:
                    h += x[:h.size(0)]
                x = h.relu_()
                # x = self.push_and_pull(hist, x, *args)
                # ipdb.set_trace()
                x, time_push_and_pull = self.push_and_pull(hist, x, batch_size=args[0], n_id=args[1].to(hist.emb.device), offset=args[2], count=args[3])
                time_push_and_pull_all += time_push_and_pull
                x = F.dropout(x, p=self.dropout, training=self.training)

            h = self.convs[-1](x, x_0, adj_t)
            if self.batch_norm:
                h = self.bns[-1](h)
            if self.residual:
                h += x[:h.size(0)]
            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[1](x)
        else:
            log.info(f'Not using aggregation... no neighbor information')
            # only keep current batch
            x = x[:batch_size]
            x_0 = x_0[:batch_size]
            
            for conv, bn, hist in zip(self.convs[:-1], self.bns[:-1], self.histories):
                # h = conv(x, x_0, adj_t)
                # ipdb.set_trace()
                h = conv.forward_no_neighbor(x, x_0)
                if self.batch_norm:
                    h = bn(h)
                if self.residual:
                    h += x[:h.size(0)]
                x = h.relu_()
                # x = self.push_and_pull(hist, x, *args)
                # ipdb.set_trace()

                ### no need to do push_and_pull for this setting
                # x, time_push_and_pull = self.push_and_pull(hist, x, batch_size=args[0], n_id=args[1].to(hist.emb.device), offset=args[2], count=args[3])
                # time_push_and_pull_all += time_push_and_pull

                x = F.dropout(x, p=self.dropout, training=self.training)

            # h = self.convs[-1](x, x_0, adj_t)
            h = self.convs[-1].forward_no_neighbor(x, x_0)
            if self.batch_norm:
                h = self.bns[-1](h)
            if self.residual:
                h += x[:h.size(0)]
            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[1](x)

        # return x
        return x, time_push_and_pull_all

    def VR_forward(self, x: Tensor, adj_t: SparseTensor, drift_norm: int,  epoch: int, batch_idx: int,  *args) -> Tensor:
        if args[1].shape[0] == x.shape[0]:
            n_id = args[1]
        n_id = n_id.to(x.device)
        batch_size = args[0]

        """
        # Modified part: Aggregate only in-batch neighbors
        in_batch_mask = (adj_t.storage.row() < batch_size) & (adj_t.storage.col() < batch_size)
        out_batch_mask = ~in_batch_mask

        # Count in-batch and out-of-batch neighbors
        num_in_batch_neighbors = in_batch_mask.sum().item()
        num_out_batch_neighbors = out_batch_mask.sum().item()
        # log.info(f'Number of in-batch neighbors: {num_in_batch_neighbors}, Number of out-of-batch neighbors: {num_out_batch_neighbors}')
        
        in_batch_adj = SparseTensor(
            row=adj_t.storage.row()[in_batch_mask],
            col=adj_t.storage.col()[in_batch_mask],
            value=adj_t.storage.value()[in_batch_mask] if adj_t.storage.value() is not None else None,
            sparse_sizes=(adj_t.size(0), adj_t.size(1))
        )

        adj_t = in_batch_adj
        # log.info(f'Discard out-of-batch neighbors!')

        """

        # input dropout
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # linear layer at the beginning, embedding dim 128 --> 256
        x = x_0 = self.lins[0](x).relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)

        time_push_and_pull_all = 0
        # for conv, bn, hist in zip(self.convs[:-1], self.bns[:-1], self.histories):
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns[:-1])):
            """
            ### get M_in and M_aggr
            period_hist_selected = self.period_hist[i].index_select(0, n_id)
            period_hist_aggr_selected = self.period_hist_aggr[i].to(x.device).index_select(0, n_id[:batch_size])

            
            # absolute embedding drift 
            diff = x - period_hist_selected
            absolute_drift = torch.norm(diff, dim=1, p=drift_norm)
            absolute_drift = absolute_drift.mean()
            self.absolute_drift_values[i].append(absolute_drift.item())
            
            # relative embedding drift
            relative_drift = absolute_drift / torch.norm(x, dim=1, p=drift_norm)
            relative_drift = relative_drift.mean()
            self.relative_drift_values[i].append(relative_drift.item())
            """

            
            ### new implementation, get M_in and M_ag from CPU using asynchronic pools
            period_hist_selected_new = self.pool.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
            period_hist_aggr_selected_new = self.pool_ag.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
            
            
            # h = conv(x, x_0, adj_t) # original gcn2_conv layer forwarding
            # vr-aggregation here
            # edge_index, edge_weight = conv.forward_before_propagate(x, x_0, adj_t)
            # ipdb.set_trace()
            # h = adj_t @ x - (adj_t @ period_hist_selected - period_hist_aggr_selected)
            h = adj_t @ (x - period_hist_selected_new) + period_hist_aggr_selected_new
            # h = edge_index@(x-period_hist_selected_new) + period_hist_aggr_selected_new
            h = conv.forward_after_propagate(h, x_0)

            
            ### free the buffer space for current batch and layer of M_in and M_ag
            self.pool.free_pull()
            self.pool_ag.free_pull()
            

            # ipdb.set_trace()
            if self.batch_norm:
                h = bn(h)
            if self.residual:
                h += x[:h.size(0)]
            x = h.relu_()
            # x = self.push_and_pull(hist, x, *args)
            # ipdb.set_trace()
            # x, time_push_and_pull = self.push_and_pull(hist, x, batch_size=args[0], n_id=args[1].to(hist.emb.device), offset=args[2], count=args[3])
            # time_push_and_pull_all += time_push_and_pull
            time_push_and_pull_all += 0
            x = F.dropout(x, p=self.dropout, training=self.training)

        """
        # get M_in and M_aggr
        period_hist_selected = self.period_hist[-1].index_select(0, n_id)
        period_hist_aggr_selected = self.period_hist_aggr[-1].to(x.device).index_select(0, n_id[:batch_size])
        
        # absolute embedding drift 
        diff = x - period_hist_selected
        absolute_drift = torch.norm(diff, dim=1, p=drift_norm)
        absolute_drift = absolute_drift.mean()
        self.absolute_drift_values[-1].append(absolute_drift.item())

        # relative embedding drift
        relative_drift = absolute_drift / torch.norm(x, dim=1, p=drift_norm)
        relative_drift = relative_drift.mean()
        self.relative_drift_values[-1].append(relative_drift.item())        
        """

        # only keep current batch
        x = x[:batch_size]

        # new implementation, get M_in and M_ag from CPU using asynchronic pools
        period_hist_selected_new = self.pool.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
        period_hist_aggr_selected_new = self.pool_ag.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()

        # last GCN2Conv layer
        # h = self.convs[-1](x, x_0, adj_t) # original gcn2_conv layer forwarding
        # h = adj_t @ x - (adj_t @ period_hist_selected - period_hist_aggr_selected)
        h = adj_t @ (x - period_hist_selected_new) + period_hist_aggr_selected_new
        h = self.convs[-1].forward_after_propagate(h, x_0)

        ### free the buffer space for current batch and layer of M_in and M_ag
        self.pool.free_pull()   
        self.pool_ag.free_pull()

        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual:
            h += x[:h.size(0)]
        x = h.relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)

        # linear layer at the end, embedding dim 256 --> 40
        x = self.lins[1](x)

        # return x
        return x, time_push_and_pull_all, 0, 0

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state, use_aggregation=True):
        if use_aggregation: # GCN2
            if layer == 0:
                if self.drop_input:
                    x = F.dropout(x, p=self.dropout, training=self.training)

                x = x_0 = self.lins[0](x).relu_()
                state['x_0'] = x_0[:adj_t.size(0)]

            x = F.dropout(x, p=self.dropout, training=self.training)
            h = self.convs[layer](x, state['x_0'], adj_t)
            # ipdb.set_trace()
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = h.relu_()

            if layer == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.lins[1](x)

        else: # remove aggregation
            x = x[:adj_t.size(0)] # only cope with in-batch target nodes, no neighbors
            if layer == 0:
                if self.drop_input:
                    x = F.dropout(x, p=self.dropout, training=self.training)

                x = x_0 = self.lins[0](x).relu_()
                state['x_0'] = x_0[:adj_t.size(0)]

            # print(f'x.shape: {x.shape}')
            # ipdb.set_trace()

            x = F.dropout(x, p=self.dropout, training=self.training)
            # h = self.convs[layer](x, state['x_0'], adj_t)
            h = self.convs[layer].forward_no_neighbor(x, state['x_0'])
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = h.relu_()

            if layer == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.lins[1](x)


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
        raw_hist = self.lins[0](raw_hist).relu_()
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
            # ipdb.set_trace()

            self.period_hist.append(hist)
            self.period_hist_aggr.append(aggr_hist)
        # ipdb.set_trace()

    @torch.no_grad()
    def update_period_hist(self, A, node_features=None):
        log.info(f'Updating M_in and calculating M_ag')
        hist_device = self.histories[0].pull().device
        A = A.to(hist_device)

        # if raw features are inputed, update layer[0] periodical memory
        if node_features != None:
            raw_hist = node_features.to(hist_device).clone().detach()
            raw_hist = self.lins[0](raw_hist).relu_()
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

            M_in_layer0 = self.lins[0](x).relu_()

            M_ag_layer0 = adj_t @ M_in_layer0 
            self.pool_ag.async_push(M_ag_layer0, offset, count, self.histories_ag[0].emb)

            M_in_layer0 = M_in_layer0[:batch_size] # only keep the batch_size rows
            self.pool.async_push(M_in_layer0, offset, count, self.histories[0].emb) # push x to history
            self.pool.async_push(out, offset, count, self.histories[1].emb) # index change


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
                # x_useless = self.pool_ag.synchronize_pull() # for debug
                M_ag = adj_t @ x
                self.pool_ag.async_push(M_ag, offset, count, self.histories_ag[i].emb)

                out = self.forward_layer(i, x, adj_t, state, use_aggregation)[:batch_size]
                self.pool.async_push(out, offset, count, self.histories[i+1].emb) # index change
                # self.pool_ag.free_pull() # for debug
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

    @torch.no_grad()
    def plot_absolute_drift(self):
        plt.figure()
        for i, drift_values in enumerate(self.absolute_drift_values):
            plt.plot(drift_values, label=f'Layer {i}')
        plt.xlabel('Training Step')
        plt.ylabel('Absolute Drift')
        plt.title('Absolute Drift over Training Steps for Each Layer')
        plt.legend()
        plt.savefig('absolute_drift_plot.png')
        plt.close()

    @torch.no_grad()
    def plot_absolute_approximation_error(self):
        plt.figure()
        for i, error_values in enumerate(self.absolute_approx_error_values):
            plt.plot(error_values, label=f'After layer {i}')
        plt.xlabel('Training Step')
        plt.ylabel('Absolute approximation error')
        plt.title('Absolute approximation error over Training Steps for Each Layer')
        plt.legend()
        plt.savefig('absolute_approximation_error_plot.png')
        plt.close()