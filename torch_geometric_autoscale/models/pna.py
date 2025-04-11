from itertools import product
from typing import Optional, List

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing

from torch_geometric_autoscale.models import ScalableGNN

EPS = 1e-5

import ipdb
import matplotlib.pyplot as plt

# ambershhek for logging
import logging
# A logger for this file
log = logging.getLogger(__name__)


class PNAConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 **kwargs):
        super().__init__(aggr=None, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers

        deg = deg.to(torch.float)
        self.avg_deg = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
        }

        self.pre_lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels)
            for _ in range(len(aggregators) * len(scalers))
        ])
        self.post_lins = torch.nn.ModuleList([
            Linear(out_channels, out_channels)
            for _ in range(len(aggregators) * len(scalers))
        ])

        self.lin = Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.pre_lins:
            lin.reset_parameters()
        for lin in self.post_lins:
            lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, adj_t):
        out = self.propagate(adj_t, x=x)
        out += self.lin(x)[:out.size(0)]
        return out

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        deg = adj_t.storage.rowcount().to(x.dtype).view(-1, 1)

        out = 0
        for (aggr, scaler), pre_lin, post_lin in zip(
                product(self.aggregators, self.scalers), self.pre_lins,
                self.post_lins):
            # ipdb.set_trace()
            h = pre_lin(x).relu_() 
            h = adj_t.matmul(h, reduce=aggr)
            h = post_lin(h)
            if scaler == 'amplification':
                h *= (deg + 1).log() / self.avg_deg['log']
            elif scaler == 'attenuation':
                h *= self.avg_deg['log'] / ((deg + 1).log() + EPS)

            out += h

        return out


class PNA(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, aggregators: List[int],
                 scalers: List[int], deg: Tensor, dropout: float = 0.0,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None):
        super().__init__(num_nodes, hidden_channels, num_layers, pool_size,
                         buffer_size, device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual

        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            conv = PNAConv(in_dim, out_dim, aggregators=aggregators,
                           scalers=scalers, deg=deg)
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers - 1):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)

        self.absolute_drift_values = [[] for _ in range(num_layers)]  # To store absolute drift values for each layer
        self.relative_drift_values = [[] for _ in range(num_layers)]  # To store relative drift values for each layer
        self.absolute_approx_error_values = [[] for _ in range(num_layers)]  # To store absolute approximation error values AFTER each layer
        self.relative_approx_error_values = [[] for _ in range(num_layers)]  # To store relative approximation error values AFTER each layer


    @property
    def reg_modules(self):
        return ModuleList(list(self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.convs[-1:]

    def reset_parameters(self):
        super().reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        time_push_and_pull_all = 0 
        
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn, hist in zip(self.convs[:-1], self.bns, self.histories):
            h = conv(x, adj_t)
            if self.batch_norm:
                h = bn(h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = h.relu_()
            # ipdb.set_trace()
            # x = self.push_and_pull(hist, x, *args)
            x, time_push_and_pull = self.push_and_pull(hist, x, batch_size=args[3], n_id=args[4].to(hist.emb.device), offset=args[5], count=args[6])
            time_push_and_pull_all += time_push_and_pull
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, adj_t)
        return x, time_push_and_pull_all


    # how called: out, time_push_and_pull_all, out_no_VR, drift = self.VR_forward(x, adj_t, drift_norm, batch_size, n_id, offset, count, **kwargs)
    def VR_forward(self, x: Tensor, adj_t: SparseTensor, drift_norm: int, *args) -> Tensor:
        # ipdb.set_trace()
        if args[1].shape[0] == x.shape[0]:
            n_id = args[1]
        batch_size = args[0]
        offset = args[2]
        count = args[3]

        # Modified part: Aggregate only in-batch neighbors
        in_batch_mask = (adj_t.storage.row() < batch_size) & (adj_t.storage.col() < batch_size)
        out_batch_mask = ~in_batch_mask

        # Count in-batch and out-of-batch neighbors
        num_in_batch_neighbors = in_batch_mask.sum().item()
        num_out_batch_neighbors = out_batch_mask.sum().item()
        log.info(f'Number of in-batch neighbors: {num_in_batch_neighbors}, Number of out-of-batch neighbors: {num_out_batch_neighbors}')
        
        in_batch_adj = SparseTensor(
            row=adj_t.storage.row()[in_batch_mask],
            col=adj_t.storage.col()[in_batch_mask],
            value=adj_t.storage.value()[in_batch_mask] if adj_t.storage.value() is not None else None,
            sparse_sizes=(adj_t.size(0), adj_t.size(1))
        )

        # Combine in_batch_mask and out_batch_mask directly
        combined_mask = in_batch_mask | out_batch_mask

        # Create the combined adjacency using the combined mask
        combined_adj = SparseTensor(
            row=adj_t.storage.row()[combined_mask],
            col=adj_t.storage.col()[combined_mask],
            value=adj_t.storage.value()[combined_mask] if adj_t.storage.value() is not None else None,
            sparse_sizes=adj_t.sparse_sizes(),
            is_sorted=True,  # Use True to prevent re-ordering in SparseTensor constructor
            trust_data=True
        )

        adj_t = in_batch_adj
        log.info(f'Discard out-of-batch neighbors!')
        # adj_t = combined_adj
        # log.info(f'Use IB+OB neighbors!')

        time_push_and_pull_all = 0 

        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i, (conv, bn, hist) in enumerate(zip(self.convs[:-1], self.bns, self.histories)):
            n_id = n_id.to(x.device)
            """ PNAConv .foward()
            def forward(self, x: Tensor, adj_t):
                out = self.propagate(adj_t, x=x)
                out += self.lin(x)[:out.size(0)]
                return out
            """
            
            period_hist_selected = self.period_hist[i].index_select(0, n_id)
            period_hist_aggr_selected = self.period_hist_aggr[i].to(x.device).index_select(0, n_id[:batch_size])
            
            diff = x - period_hist_selected
            absolute_drift = torch.norm(diff, dim=1, p=drift_norm)
            absolute_drift = absolute_drift.mean()
            self.absolute_drift_values[i].append(absolute_drift.item())
            relative_drift = absolute_drift / torch.norm(x, dim=1, p=drift_norm)
            relative_drift = relative_drift.mean()
            self.relative_drift_values[i].append(relative_drift.item())
            
            # h = conv(x, adj_t) # vanilla forward
            online_period_hist_aggr = conv.propagate(combined_adj, x=period_hist_selected).detach()
            log.info(f'Distance between online & periodical M_ag = {torch.norm(conv(period_hist_selected, combined_adj) - period_hist_aggr_selected)}')

            # implementation 1 - vr on only propagate
            # h = conv.propagate(adj_t, x=x-period_hist_selected) + online_period_hist_aggr
            h = conv.propagate(adj_t, x=x)  # mock for debug
            h += conv.lin(x)[:h.size(0)]  

            # implementation 2 - vr on entire PNAConv layer
            # h = conv(x, adj_t) - conv(period_hist_selected, adj_t) + conv(period_hist_selected, combined_adj) # online Mag, not detached
            # h = conv(x, adj_t) - conv(period_hist_selected, adj_t) + period_hist_aggr_selected  # stored Mag

            if self.batch_norm:
                h = bn(h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = h.relu_()
            # x = self.push_and_pull(hist, x, *args)
            x, time_push_and_pull = self.push_and_pull(hist, x, batch_size=batch_size, n_id=n_id.to(hist.emb.device), offset=offset, count=count)
            time_push_and_pull_all += time_push_and_pull
            x = F.dropout(x, p=self.dropout, training=self.training)


        period_hist_selected = self.period_hist[-1].index_select(0, n_id)
        period_hist_aggr_selected = self.period_hist_aggr[-1].to(x.device).index_select(0, n_id[:batch_size])
        # log.info(f'[Before VR] L2 distance between x[{-1}] and M_in[{-1}] = {torch.norm(x - period_hist_selected, p=2)}')
        diff = x - period_hist_selected
        absolute_drift = torch.norm(diff, dim=1, p=drift_norm)
        absolute_drift = absolute_drift.mean()
        self.absolute_drift_values[-1].append(absolute_drift.item())
        relative_drift = absolute_drift / torch.norm(x, dim=1, p=drift_norm)
        relative_drift = relative_drift.mean()
        self.relative_drift_values[-1].append(relative_drift.item())

        # x = self.convs[-1](x, adj_t) # vanilla forward
        online_period_hist_aggr = self.convs[-1](period_hist_selected, combined_adj).detach()
        log.info(f'Distance between online & periodical M_ag = {torch.norm(self.convs[-1](period_hist_selected, combined_adj) - period_hist_aggr_selected).mean().item()}')
        
        # implementation 1 - vr on only propagate
        # h = self.convs[-1].propagate(adj_t, x=x-period_hist_selected) + online_period_hist_aggr
        h = self.convs[-1].propagate(adj_t, x=x)  # mock for debug
        h += self.convs[-1].lin(x)[:h.size(0)]
        
        # implementation 2 - vr on entire PNAConv layer
        # h = self.convs[-1](x, adj_t) - self.convs[-1](period_hist_selected, adj_t) + self.convs[-1](period_hist_selected, combined_adj) # online Mag, not deatach
        # h = self.convs[-1](x, adj_t) - self.convs[-1](period_hist_selected, adj_t) + period_hist_aggr_selected # stored Mag
        # h = self.convs[-1](x, adj_t) - self.convs[-1](period_hist_selected, adj_t) + online_period_hist_aggr # online Mag, detached

        return h, time_push_and_pull_all, None, relative_drift


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state):
        if layer == 0 and self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[layer](x, adj_t)
        if layer < self.num_layers - 1:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = h.relu_()
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h
    

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
        # aggr_hist_0 = A @ raw_hist
        # aggr_hist_0 =  self.convs[0].propagate(A, x=raw_hist) # imple 1
        aggr_hist_0 =  self.convs[0](raw_hist, A) # imple 2
        aggr_hist_0 = aggr_hist_0.clone().detach()
        # ipdb.set_trace()

        self.period_hist.append(raw_hist)  # Add raw features
        self.period_hist_aggr.append(aggr_hist_0)  # Add aggregated raw features

        # Add features from histories with varying dimensions
        for i, history in enumerate(self.histories):
            hist = history.pull().to(hist_device).clone().detach()
            # aggr_hist = A @ hist
            # ipdb.set_trace()
            # aggr_hist  = self.convs[i+1].propagate(A, x=hist) # imple 1
            aggr_hist  = self.convs[i+1](hist, A) # imple 2
            aggr_hist = aggr_hist.clone().detach()
            # ipdb.set_trace()

            self.period_hist.append(hist)
            self.period_hist_aggr.append(aggr_hist)
        # ipdb.set_trace()

    @torch.no_grad()
    def update_period_hist(self, A):
        log.info(f'Updating M_in and calculating M_ag')
        hist_device = self.histories[0].pull().device
        A = A.to(hist_device)

        # Update the histories with varying dimensions while preserving the raw features
        for i, history in enumerate(self.histories):
            hist = history.pull().to(hist_device).clone().detach()
            # aggr_hist = A @ hist
            aggr_hist  = self.convs[i+1].propagate(A, x=hist) # imple 1
            # aggr_hist  = self.convs[i+1](hist, A) # imple 2
            aggr_hist = aggr_hist.clone().detach()

            self.period_hist[i + 1] = hist
            self.period_hist_aggr[i + 1] = aggr_hist

    @torch.no_grad()
    def update_period_hist_momentum(self, A, alpha=0.1):
        log.info(f'Updating M_in and calculating M_ag with momentum (alpha={alpha})')
        hist_device = self.histories[0].pull().device
        A = A.to(hist_device)

        # Update the histories with momentum while preserving the raw features
        for i, history in enumerate(self.histories):
            hist = history.pull().to(hist_device).clone().detach()
            # aggr_hist = A @ hist
            aggr_hist  = self.convs[i].propagate(A, x=hist)
            aggr_hist = aggr_hist.clone().detach()

            self.period_hist[i + 1] = (1 - alpha) * self.period_hist[i + 1] + alpha * hist
            self.period_hist_aggr[i + 1] = (1 - alpha) * self.period_hist_aggr[i + 1] + alpha * aggr_hist

    @torch.no_grad()
    def plot_relative_drift(self):
        plt.figure()
        for i, drift_values in enumerate(self.relative_drift_values):
            plt.plot(drift_values, label=f'Layer {i}')
        plt.xlabel('Training Step')
        plt.ylabel('Relative Drift')
        plt.title('Relative Drift over Training Steps for Each Layer')
        plt.legend()
        plt.savefig('relative_drift_plot.png')
        plt.close()

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


    @torch.no_grad()
    def plot_relative_approximation_error(self):
        plt.figure()
        for i, error_values in enumerate(self.relative_approx_error_values):
            plt.plot(error_values, label=f'After layer {i}')
        plt.xlabel('Training Step')
        plt.ylabel('Relative approximation error')
        plt.title('Relative approximation error over Training Steps for Each Layer')
        plt.legend()
        plt.savefig('relative_approximation_error_plot.png')
        plt.close()