from typing import Optional

import torch
# torch.set_default_dtype(torch.float64)


from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv

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

from torch_geometric.utils import spmm


class GraphSAGE(ScalableGNN):
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
        
        # self.in_aggr_mem = [torch.zeros(num_nodes, out_channels)]
        # self.in_aggr_mem = [torch.zeros(num_nodes, out_channels) for _ in range(num_layers - 1)]
        # 20240905 to test momentum update
        # self.in_aggr_mem = [torch.zeros(num_nodes, dim) for dim in (256, 256)]

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
            conv = SAGEConv(in_dim, out_dim, normalize=False)
            self.convs.append(conv)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = BatchNorm1d(hidden_channels)
            self.bns.append(bn)
            
        # ipdb.set_trace()
        # change the dtype of bn
        # for bn in self.bns:
        #     bn.weight = torch.nn.Parameter(bn.weight.double())
        #     bn.bias = torch.nn.Parameter(bn.bias.double())
        #     bn.running_mean = bn.running_mean.double()
        #     bn.running_var = bn.running_var.double()
        
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
            # Combine in_batch_mask and out_batch_mask directly
            combined_mask = in_batch_mask | out_batch_mask

            ### Count in-batch and out-of-batch neighbors
            # num_in_batch_neighbors = in_batch_mask.sum().item()
            # num_out_batch_neighbors = out_batch_mask.sum().item()
            # log.info(f'Number of in-batch neighbors: {num_in_batch_neighbors}, Number of out-of-batch neighbors: {num_out_batch_neighbors}')

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
                # log.info(f'Use ib & ob neighbors!')
            else:
                adj_t = in_batch_adj_t
                # log.info(f'Discard out-of-batch neighbors!')

            # for conv, bn, hist in zip(self.convs[:-1], self.bns, self.histories):
            for layer_idx, (conv, bn, hist) in enumerate(zip(self.convs[:-1], self.bns, self.histories)):
                """
                ### record x before this layer to calculate h_VR
                x_before = x.clone().detach() 
                ### track drift for runs without VR
                n_id = n_id.to(x.device)
                period_hist_selected = self.period_hist[layer_idx].index_select(0, n_id)
                period_hist_aggr_selected = self.period_hist_aggr[layer_idx].to(x.device).index_select(0, n_id[:batch_size])
                diff = x - period_hist_selected
                absolute_drift = torch.norm(diff, dim=1, p=drift_norm)
                absolute_drift = absolute_drift.mean()
                self.absolute_drift_values[layer_idx].append(absolute_drift.item())
                relative_drift = absolute_drift / torch.norm(x, dim=1, p=drift_norm)
                relative_drift = relative_drift.mean()
                self.relative_drift_values[layer_idx].append(relative_drift.item())
                """

                # start_time_layer = torch.cuda.Event(enable_timing=True)
                # end_time_layer = torch.cuda.Event(enable_timing=True)
                # start_time_layer.record()

                # ipdb.set_trace()
                h = conv(x, adj_t)
                # h_copy = h.clone().detach()
                if self.batch_norm:
                    # pass
                    h = bn(h)
                    # h_copy_bn = bn(h_copy) # go through train bn
                if self.residual and h.size(-1) == x.size(-1):
                    h += x[:h.size(0)]
                x = h.relu_()

                # x, time_push_and_pull = self.push_and_pull(hist, x, batch_size=args[0], n_id=args[1].to(hist.emb.device), offset=args[2], count=args[3])
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                x, returned_time = self.push_and_pull(self.histories[layer_idx+1], x, batch_size=args[0], n_id=args[1].to(hist.emb.device), offset=args[2], count=args[3])
                end_time.record()

                torch.cuda.synchronize()
                time_push_and_pull = start_time.elapsed_time(end_time) / 1000  # Convert milliseconds to seconds
                # ipdb.set_trace()

                time_push_and_pull_all += time_push_and_pull
                               
                x = F.dropout(x, p=self.dropout, training=self.training)

                # end_time_layer.record()
                # torch.cuda.synchronize()
                # time_layer = start_time_layer.elapsed_time(end_time_layer) / 1000  # Convert milliseconds to seconds
                # log.info(f'Time taken for layer {layer_idx}: {time_layer} seconds')

                """
                # 1023 if updating with h_full, then calculate the approximation error of h_VR AFTER this layer 
                if aggregate_combined:
                    with torch.no_grad():  # Disable gradient tracking for this comparison
                        self.eval()
                        n_id = n_id.to(x.device)
                        h_VR = conv.lin(x_before)

                        period_hist_selected = self.period_hist[layer_idx].index_select(0, n_id).clone().detach()
                        period_hist_aggr_selected = self.period_hist_aggr[layer_idx].to(x.device).index_select(0, n_id[:batch_size]).clone().detach()
                        h_VR = in_batch_adj @ h_VR - conv.lin(in_batch_adj @ period_hist_selected - period_hist_aggr_selected)  
                        # log.info(f'[After VR] L2 distance between h[{i}] and M_in[{i}] = {torch.norm(h - period_hist_selected, p=2)}')
                        # h_VR = conv(x_before, in_batch_adj)
                        if conv.bias is not None:
                            h_VR = h_VR + conv.bias
                        # ipdb.set_trace() # check h after GCNConv layer
                        # log.info(f'After only GCNConv layer, torch.norm(h_copy - h_VR) = {torch.norm(h_copy - h_VR)}')

                        # operations after the GCNConv layer
                        if self.batch_norm: # True in default settings of GCN on arxiv
                            # pass
                            # log.info(f'Before BatchNorm, torch.norm(h_VR - h_copy) = {torch.norm(h_VR - h_copy)}')
                            h_VR = bn(h_VR) 
                            # h_copy = bn(h_copy) # go through eval bn
                            # log.info(f'After BatchNorm, torch.norm(h_VR - h_copy) = {torch.norm(h_VR - h_copy)}')
                            # log.info(f'After BatchNorm, torch.norm(h_VR - h_copy_bn) = {torch.norm(h_VR - h_copy_bn)}')
                        # ipdb.set_trace() # check h after bn
                        if self.residual and h_VR.size(-1) == x_before.size(-1): # False in default settings of GCN on arxiv
                            # log.info(f'Before Residual, torch.norm(h_VR - h_copy) = {torch.norm(h_VR - h_copy)}')
                            h_VR += x_before[:h.size(0)] 
                            # h_copy += x_before[:h.size(0)]
                            # log.info(f'After Residual, torch.norm(h_VR - h_copy) = {torch.norm(h_VR - h_copy)}')
                        # log.info(f'Before ReLU, torch.norm(h_VR - h_copy) = {torch.norm(h_VR - h_copy)}')
                        x_VR = h_VR.relu_()
                        # x_copy = h_copy.relu_()
                        # log.info(f'After ReLU, torch.norm(x_VR - x_copy) = {torch.norm(x_VR - x_copy)}')

                        # log.info(f'To check x_copy go through correct forwarding, torch.norm(x_after, x_copy) = {torch.norm(x_after - x_copy)}')

                        x_VR = F.dropout(x_VR, p=self.dropout, training=self.training)
                        approx_error = x_after - x_VR
                        approx_error_absolute = torch.norm(approx_error).mean()
                        # log.info(f'approx_error_absolute.item() = {approx_error_absolute.item()}')
                        self.absolute_approx_error_values[layer_idx].append(approx_error_absolute.item())

                        approx_error_relative = approx_error_absolute / torch.norm(x_after)
                        # log.info(f'torch.norm(x_after) = {torch.norm(x_after)}')
                        approx_error_relative = approx_error_relative.mean()
                        # log.info(f'approx_error_relative.item() = {approx_error_relative.item()}')
                        self.relative_approx_error_values[layer_idx].append(approx_error_relative.item())
                        self.train()

            """
  
            h = self.convs[-1](x, adj_t)

            """
            # record x before this layer to calculate h_VR
            x_before = x.clone().detach() 
            h = self.convs[-1](x, adj_t)
            h_after = h.clone().detach()

            # track drift for runs without VR
            n_id = n_id.to(x.device)
            period_hist_selected = self.period_hist[-1].index_select(0, n_id)
            period_hist_aggr_selected = self.period_hist_aggr[-1].to(x.device).index_select(0, n_id[:batch_size])
            diff = x - period_hist_selected
            absolute_drift = torch.norm(diff, dim=1, p=drift_norm)
            absolute_drift = absolute_drift.mean()
            self.absolute_drift_values[-1].append(absolute_drift.item())
            relative_drift = absolute_drift / torch.norm(x, dim=1, p=drift_norm)
            relative_drift = relative_drift.mean()
            self.relative_drift_values[-1].append(relative_drift.item())

            # 1023 if updating with h_full, then calculate the approximation error of h_VR AFTER this layer 
            if aggregate_combined:
                self.eval()
                n_id = n_id.to(x.device)
                h_VR = self.convs[-1].lin(x_before)
                period_hist_selected = self.period_hist[-1].index_select(0, n_id).clone().detach()
                period_hist_aggr_selected = self.period_hist_aggr[-1].to(x.device).index_select(0, n_id[:batch_size]).clone().detach()
                h_VR = in_batch_adj @ h_VR - self.convs[-1].lin(in_batch_adj @ period_hist_selected - period_hist_aggr_selected)  
                # h_VR = self.convs[-1](x_before, in_batch_adj)
                if self.convs[-1].bias is not None:
                    h_VR = h_VR + self.convs[-1].bias
                approx_error = h_after - h_VR
                approx_error_absolute = torch.norm(approx_error).mean()
                self.absolute_approx_error_values[-1].append(approx_error_absolute.item())
                approx_error_relative = approx_error_absolute / torch.norm(h_after)
                approx_error_relative = approx_error_relative.mean()
                self.relative_approx_error_values[-1].append(approx_error_relative.item())
                self.train()
            """


        # not use the aggregation function in GCNConv, degrade to MLP
        else:
            log.info(f'Not using aggregation... GCNConv degraded to MLP!')
            # only keep current batch
            x = x[:batch_size]
            
            for layer_idx, (conv, bn, hist) in enumerate(zip(self.convs[:-1], self.bns, self.histories)):

                h = conv.lin(x)
                # not use aggregation in GCNConv
                # h = adj_t @ h
                if conv.bias is not None:
                    h = h + conv.bias
                
                if self.batch_norm:
                    h = bn(h)
                if self.residual and h.size(-1) == x.size(-1):
                    h += x[:h.size(0)]
                x = h.relu_()

                """
                # Track relative drift for non-VR
                n_id = n_id.to(x.device)
                # ipdb.set_trace()
                diff = x - self.period_hist[layer_idx+1].index_select(0, n_id[:batch_size])
                relative_drift = torch.norm(diff, dim=1, p=drift_norm) / torch.norm(x, dim=1, p=drift_norm)
                relative_drift = relative_drift.mean()
                log.info(f'[No VR] Relative drift between x_{layer_idx+1} and M_in_{layer_idx+1} = {relative_drift}')
                self.relative_drift_values[layer_idx+1].append(relative_drift.item())  # Store relative drift value for each layer
                """

                # use only the in-batch n_id parts
                x, time_push_and_pull = self.push_and_pull(hist, x, batch_size=args[0], n_id=args[1].to(hist.emb.device)[:batch_size], offset=args[2], count=args[3])
                time_push_and_pull_all += time_push_and_pull
                
                x = F.dropout(x, p=self.dropout, training=self.training)

            h = self.convs[-1].lin(x)
            if self.convs[-1].bias is not None:
                h = h + self.convs[-1].bias
            

        if not self.linear:
            # return h
            return h, time_push_and_pull_all

        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        # return self.lins[1](h)
        return self.lins[1](h), time_push_and_pull_all

    ### VR_forward used in main results
    def VR_forward_archive(self, x: Tensor, adj_t: SparseTensor, drift_norm: int, epoch: int, *args) -> Tensor:
        # find specific arguments in *args
        if args[1].shape[0] == x.shape[0]:
            n_id = args[1]
        batch_size = args[0]
        
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.linear:
            x = self.lins[0](x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        time_data_movement = 0
        # drift = 0

        # only keep current batch
        x = x[:batch_size]
        
        # Modified part: Aggregate only in-batch neighbors
        in_batch_mask = (adj_t.storage.row() < batch_size) & (adj_t.storage.col() < batch_size)
        out_batch_mask = ~in_batch_mask

        ### Count in-batch and out-of-batch neighbors
        num_in_batch_neighbors = in_batch_mask.sum().item()
        num_out_batch_neighbors = out_batch_mask.sum().item()
        # log.info(f'Number of in-batch neighbors: {num_in_batch_neighbors}, Number of out-of-batch neighbors: {num_out_batch_neighbors}')
        
        in_batch_adj = SparseTensor(
            row=adj_t.storage.row()[in_batch_mask],
            col=adj_t.storage.col()[in_batch_mask],
            value=adj_t.storage.value()[in_batch_mask] if adj_t.storage.value() is not None else None,
            sparse_sizes=(adj_t.size(0), adj_t.size(0))
        )

        out_batch_adj = SparseTensor(
            row=adj_t.storage.row()[out_batch_mask],
            col=adj_t.storage.col()[out_batch_mask],
            value=adj_t.storage.value()[out_batch_mask] if adj_t.storage.value() is not None else None,
            sparse_sizes=adj_t.sparse_sizes()
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

        # ipdb.set_trace()
        adj_t = in_batch_adj
        # log.info(f'Discard out-of-batch neighbors!')
        # adj_t = combined_adj
        # log.info(f'Use ib & ob neighbors!')

        for i, (conv, bn, hist) in enumerate(zip(self.convs[:-1], self.bns, self.histories)):
            n_id = n_id.to(x.device)

            # period_hist_selected = self.period_hist[i].index_select(0, n_id)
            period_hist_selected = self.period_hist[i].index_select(0, n_id[:batch_size])
            period_hist_aggr_selected = self.period_hist_aggr[i].to(x.device).index_select(0, n_id[:batch_size])

            # ipdb.set_trace()
            diff = x - period_hist_selected
            absolute_drift = torch.norm(diff, dim=1, p=drift_norm)
            absolute_drift = absolute_drift.mean()
            self.absolute_drift_values[i].append(absolute_drift.item())

            relative_drift = absolute_drift / torch.norm(x, dim=1, p=drift_norm)
            relative_drift = relative_drift.mean()
            self.relative_drift_values[i].append(relative_drift.item())

            # linear transformation in GCNConv layer
            h = conv.lin(x)

            # log.info(f'[Before VR] L2 distance between x[{i}] and M_in[{i}] = {}')
            h = adj_t @ h - conv.lin(adj_t @ period_hist_selected - period_hist_aggr_selected)  # TODO 1125 h-M_in first, then @ (to reduce calculation)
            # log.info(f'[After VR] L2 distance between h[{i}] and M_in[{i}] = {torch.norm(h - period_hist_selected, p=2)}')
            if conv.bias is not None:
                h = h + conv.bias
            # ipdb.set_trace() # check h after GCNConv layer

            # operations after the GCNConv layer
            if self.batch_norm: # True in default settings of GCN on arxiv
                h = bn(h) 
            # ipdb.set_trace() # check h after bn
            if self.residual and h.size(-1) == x.size(-1): # False in default settings of GCN on arxiv
                h += x[:h.size(0)] 
            x = h.relu_() 
            
            # diff = x - self.period_hist[i+1].index_select(0, n_id[:batch_size])
            # relative_drift = torch.norm(diff, dim=1, p=drift_norm) / torch.norm(x, dim=1, p=drift_norm)
            # relative_drift = relative_drift.mean()
            # log.info(f'Relative drift between x_{i} and M_in_{i} = {relative_drift}')
            # self.relative_drift_values[i+1].append(relative_drift.item())  # Store relative drift value for each layer


            
            ### history push and pull
            # ipdb.set_trace()  # before push_and_pull
            # x, time_push_and_pull = self.push_and_pull(hist, x, batch_size=args[0], n_id=args[1].to(hist.emb.device), offset=args[2] , count=args[3] ) #更新hist, original
            x, time_push = self.push_only(hist, x, batch_size=args[0], n_id=args[1].to(hist.emb.device), offset=args[2] , count=args[3] )
            # ipdb.set_trace()            

            time_data_movement += time_push
            # ipdb.set_trace()

            x = F.dropout(x, p=self.dropout, training=self.training)
            # ipdb.set_trace() # check x after dropout    

            # ipdb.set_trace() # check embeddings after the entire layer

        # tmp = self.convs[-1](x, adj_t)
        # tmp = self.convs[-1].lin(adj_t@x)
        # out_no_VR = tmp.clone().detach()
        # out_no_VR.requires_grad_(False) # 不使用VR得到的embeddings      
        out_no_VR = None
        
        n_id = n_id.to(x.device)
        # period_hist_selected = self.period_hist[-1].index_select(0, n_id)
        period_hist_selected = self.period_hist[-1].index_select(0, n_id[:batch_size])
        period_hist_aggr_selected = self.period_hist_aggr[-1].to(x.device).index_select(0, n_id[:batch_size])
        
        # log.info(f'[Before VR] L2 distance between x[{-1}] and M_in[{-1}] = {torch.norm(x - period_hist_selected, p=2)}')
        diff = x - period_hist_selected
        absolute_drift = torch.norm(diff, dim=1, p=drift_norm)
        absolute_drift = absolute_drift.mean()
        self.absolute_drift_values[-1].append(absolute_drift.item())

        relative_drift = absolute_drift / torch.norm(x, dim=1, p=drift_norm)
        relative_drift = relative_drift.mean()
        self.relative_drift_values[-1].append(relative_drift.item())
        
        ### VR update的最后一层聚合
        # ## 先propagate再lin
        # n_id = n_id.to(x.device)
        # # h1 = adj_t @ (x - self.period_hist[-1].index_select(0, n_id)) + self.period_hist_aggr[-1].to(x.device).index_select(0, n_id[:batch_size])
        # h1 = adj_t @ x - (adj_t @ self.period_hist[-1].index_select(0, n_id) - self.period_hist_aggr[-1].to(x.device).index_select(0, n_id[:batch_size]))
        # h = self.convs[-1].lin(h1) # 使用VR得到的embeddings
        # h = h + self.convs[-1].bias

        ### 先lin()再@
        h = self.convs[-1].lin(x)
        h = adj_t @ h - self.convs[-1].lin(adj_t @ period_hist_selected - period_hist_aggr_selected)  
        # log.info(f'compensation term = {self.convs[-1].lin(adj_t @ self.period_hist[-1].index_select(0, n_id) - self.period_hist_aggr[-1].to(x.device).index_select(0, n_id[:batch_size]))}')
        h = h + self.convs[-1].bias

        if not self.linear:
            # return h
            # return h, time_push_and_pull_all, m_in_aggr
            # return h, time_push_and_pull_all
            return h, time_data_movement, num_in_batch_neighbors, num_out_batch_neighbors, out_no_VR, relative_drift

        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        # return self.lins[1](h)
        return self.lins[1](h), time_data_movement
    
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

        # ipdb.set_trace()

        # if args[1].shape[0] == x.shape[0]:
        #     n_id = args[1]
        batch_size = args[0]

        # drift = 0     

        ### dropout
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        ### linear layer
        if self.linear:
            x = self.lins[0](x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        time_data_movement = 0

        """
        in_batch_mask = (adj_t.storage.row() < batch_size) & (adj_t.storage.col() < batch_size)
        # out_batch_mask = ~in_batch_mask

        ### Count in-batch and out-of-batch neighbors
        # num_in_batch_neighbors = in_batch_mask.sum().item()
        # num_out_batch_neighbors = out_batch_mask.sum().item()
        # log.info(f'Number of in-batch neighbors: {num_in_batch_neighbors}, Number of out-of-batch neighbors: {num_out_batch_neighbors}')


        in_batch_adj_t_1 = SparseTensor(
            row=adj_t.storage.row()[in_batch_mask],
            col=adj_t.storage.col()[in_batch_mask],
            value=adj_t.storage.value()[in_batch_mask] if adj_t.storage.value() is not None else None,
            # sparse_sizes=(adj_t.size(0), adj_t.size(1))
            sparse_sizes=(adj_t.size(0), adj_t.size(0)), # only keep in-batch neighbors
        )
        """
        
        """
        # Exclude this section from profiling
        with torch.profiler.record_function("Excluded: in_batch_adj"):
            mask = torch.zeros(n_id.numel(), dtype=torch.bool, device=adj_t.device())
            mask[:batch_size] = True
            adj_t = masked_select(src=adj_t, dim=1, mask=mask)
        """
        

        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            # start_time_layer = torch.cuda.Event(enable_timing=True)
            # end_time_layer = torch.cuda.Event(enable_timing=True)
            # start_time_layer.record()
        
            # n_id = n_id.to(x.device)

            # only keep current batch
            x = x[:batch_size]
            
            ### out-of-dated implementation, get M_in and M_ag from GPU
            # period_hist_selected = self.period_hist[i].index_select(0, n_id[:batch_size])
            # period_hist_aggr_selected = self.period_hist_aggr[i].to(x.device).index_select(0, n_id[:batch_size])
            
            
            ### new implementation, get M_in and M_ag from CPU using asynchronic pools
            period_hist_selected_new = self.pool.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
            period_hist_aggr_selected_new = self.pool_ag.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()


            # h = conv(x, adj_t) # vanilla forward, for debugging
            
            # DONE VR update here. 根据SAGEConv.forward()改
            if isinstance(x, Tensor):
                x = (x, x)
            if conv.project and hasattr(conv, 'lin'):
                x = (conv.lin(x[0]).relu(), x[1])

            # propagate_type: (x: OptPairTensor)
            # h = conv.propagate(adj_t, x=x, size=None)
            if isinstance(adj_t, SparseTensor):
                adj_t = adj_t.set_value(None, layout=None)
            # ipdb.set_trace()
            # h = spmm(adj_t, x[0], reduce=conv.aggr)
            h = spmm(adj_t, (x[0] - period_hist_selected_new), reduce=conv.aggr) + period_hist_aggr_selected_new
            
            # linear
            h = conv.lin_l(h)
            
            # residual
            x_r = x[1][:adj_t.size(0)] # ambershek only keep the in-batch part
            if conv.root_weight and x_r is not None:
                h = h + conv.lin_r(x_r)
            if conv.normalize:
                h = F.normalize(h, p=2., dim=-1)
                
            
            ## free the buffer space for current batch and layer of M_in and M_ag
            self.pool.free_pull()
            self.pool_ag.free_pull()
            

            # operations after the GCNConv layer
            if self.batch_norm: # True in default settings of GCN on arxiv
                h = bn(h) 
            if self.residual and h.size(-1) == x.size(-1): # False in default settings of GCN on arxiv
                h += x[:h.size(0)] 
            x = h.relu_() 

            ### do not update history (no push or pull)
            time_data_movement += 0

            x = F.dropout(x, p=self.dropout, training=self.training)

        # only keep current batch
        x = x[:batch_size]

        # new implementation, get M_in and M_ag from CPU using asynchronic pools
        period_hist_selected_new = self.pool.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()
        period_hist_aggr_selected_new = self.pool_ag.synchronize_pull()[:batch_size, :x.shape[1]].clone().detach()

        # h = self.convs[-1](x, adj_t) # vanilla forward, for debugging
        
        if isinstance(x, Tensor):
            x = (x, x)
        if self.convs[-1].project and hasattr(self.convs[-1], 'lin'):
            x = (self.convs[-1].lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        # h = self.convs[-1].propagate(adj_t, x=x, size=None)
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        # h = spmm(adj_t, x[0], reduce=self.convs[-1].aggr)
        h = spmm(adj_t, (x[0] - period_hist_selected_new), reduce=self.convs[-1].aggr) + period_hist_aggr_selected_new

        h = self.convs[-1].lin_l(h)
        x_r = x[1][:adj_t.size(0)] # ambershek only keep the in-batch part
        if self.convs[-1].root_weight and x_r is not None:
            h = h + self.convs[-1].lin_r(x_r)
        if self.convs[-1].normalize:
            h = F.normalize(h, p=2., dim=-1)        

        ### free the buffer space for current batch and layer of M_in and M_ag
        self.pool.free_pull()
        self.pool_ag.free_pull()

        if not self.linear:
            # return h, time_data_movement, num_in_batch_neighbors, num_out_batch_neighbors
            return h, time_data_movement, 0, 0

        if self.batch_norm:
            h = self.bns[-1](h)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        # return self.lins[1](h)
        return self.lins[1](h), time_data_movement

    def VR_forward_debug(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        raise NotImplementedError


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
            log.info(f'[forward_layer] Not using aggregation... ConvGNN degraded to MLP!')
            if layer == 0:
                if self.drop_input:
                    x = F.dropout(x, p=self.dropout, training=self.training)
                if self.linear:
                    x = self.lins[0](x).relu_()
                    x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)

            # ipdb.set_trace()
            # h = self.convs[layer](x, adj_t)
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
    def initialize_period_hist(self, A, node_features):
        log.info(f'Initializing M_in and M_ag')
        hist_device = self.histories[0].pull().device
        A = A.to(hist_device)

        # Initialize period_hist and period_hist_aggr as lists to allow different dimensions
        self.period_hist = []
        self.period_hist_aggr = []

        # Assign raw node features to period_hist[0]
        raw_hist = node_features.to(hist_device).clone().detach()
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

    # update period_hist using History classs
    # unfinished
    @torch.no_grad()
    def initialize_period_hist_History(self, A, node_features):
        log.info(f'Initializing M_in and M_ag')
        hist_device = self.histories[0].pull().device
        A = A.to(hist_device)

        # Initialize period_hist and period_hist_aggr as lists to allow different dimensions
        self.period_hist = []
        self.period_hist_aggr = []

        # Assign raw node features to period_hist[0]
        raw_hist = node_features.to(hist_device).clone().detach()
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
        # ipdb.set_trace()
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
        # for i, history in enumerate(self.histories):
        for i, (history, history_ag) in enumerate(zip(self.histories[1:], self.histories_ag[1:]), start=1):
            # ipdb.set_trace()
            hist = history.pull().to(hist_device).clone().detach()
            # aggr_hist = A @ hist
            # aggr_hist = aggr_hist.clone().detach()
            aggr_hist = history_ag.pull().to(hist_device).clone().detach() # directly use the aggregated history

            # self.period_hist[i + 1] = hist
            # self.period_hist_aggr[i + 1] = aggr_hist

            # index change after adding one layer of histories for raw feature (or transformed in APPNP)
            self.period_hist[i] = hist
            self.period_hist_aggr[i] = aggr_hist

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
            
            # debugging
            # print(f'data.x: {data.x}') 
            # print(f'data.adj_t: {data.adj_t}')
            # print(f'self.device: {self.device}')

            x = data.x.to(self.device)
            adj_t = data.adj_t.to(self.device)
            out = self.forward_layer(0, x, adj_t, state, use_aggregation)[:batch_size]

            M_in_layer0 = x[:batch_size] # only keep the batch_size rows
            if M_in_layer0.size(1) < self.hidden_channels:
                M_in_layer0_expanded = torch.zeros((M_in_layer0.size(0), self.hidden_channels), device=self.device)
                M_in_layer0_expanded[:, :M_in_layer0.size(1)] = M_in_layer0
                M_in_layer0 = M_in_layer0_expanded
            
            # ipdb.set_trace() # check adj_t

            # M_ag_layer0 = adj_t @ x # shape from (IB+OB) to IB, reddit报错的地方
            adj_t_none = adj_t.set_value(None, layout=None)
            M_ag_layer0 = spmm(adj_t_none, x, reduce=self.convs[0].aggr) # shape from (IB+OB) to IB
            if M_ag_layer0.size(1) < self.hidden_channels:
                M_ag_layer0_expanded = torch.zeros((M_ag_layer0.size(0), self.hidden_channels), device=self.device)
                M_ag_layer0_expanded[:, :M_ag_layer0.size(1)] = M_ag_layer0
                M_ag_layer0 = M_ag_layer0_expanded

            # ipdb.set_trace() # check M_in_layer0 and M_ag_layer0
            self.pool.async_push(M_in_layer0, offset, count, self.histories[0].emb) # push x to history
            self.pool_ag.async_push(M_ag_layer0, offset, count, self.histories_ag[0].emb)
            self.pool.async_push(out, offset, count, self.histories[1].emb) # index change

            torch.cuda.synchronize()

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
                # M_ag = adj_t @ x
                adj_t_none = adj_t.set_value(None, layout=None)
                M_ag = spmm(adj_t_none, x, reduce=self.convs[i].aggr) # shape from (IB+OB) to IB
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
            # M_ag = adj_t @ x
            adj_t_none = adj_t.set_value(None, layout=None)
            M_ag = spmm(adj_t_none, x, reduce=self.convs[-1].aggr) # shape from (IB+OB) to IB
            self.pool_ag.async_push(M_ag, offset, count, self.histories_ag[-1].emb)
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
            # self.pool_ag.free_pull() # for debug
        self.pool_ag.synchronize_push()
        self.pool.synchronize_push()

        return self._out
    

    @torch.no_grad()
    def update_period_hist_momentum(self, A, alpha=0.1):
        log.info(f'Updating M_in and calculating M_ag with momentum (alpha={alpha})')
        hist_device = self.histories[0].pull().device
        A = A.to(hist_device)

        # Update the histories with momentum while preserving the raw features
        for i, history in enumerate(self.histories):
            hist = history.pull().to(hist_device).clone().detach()
            aggr_hist = A @ hist
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