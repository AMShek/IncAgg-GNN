from typing import Optional, Callable, Dict, Any

import warnings

import torch
# torch.set_default_dtype(torch.float64)

from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric_autoscale import History, AsyncIOPool
from torch_geometric_autoscale import SubgraphLoader, EvalSubgraphLoader

# ambershek record time consumed for push and pull
import time
import ipdb

import logging
log = logging.getLogger(__name__)

import os
import torch.profiler



class ScalableGNN(torch.nn.Module):
    r"""An abstract class for implementing scalable GNNs via historical
    embeddings.
    This class will take care of initializing :obj:`num_layers - 1` historical
    embeddings, and provides a convenient interface to push recent node
    embeddings to the history, and to pull previous embeddings from the
    history.
    In case historical embeddings are stored on the CPU, they will reside
    inside pinned memory, which allows for asynchronous memory transfers of
    historical embeddings.
    For this, this class maintains a :class:`AsyncIOPool` object that
    implements the underlying mechanisms of asynchronous memory transfers as
    described in our paper.

    Args:
        num_nodes (int): The number of nodes in the graph.
        hidden_channels (int): The number of hidden channels of the model.
            As a current restriction, all intermediate node embeddings need to
            utilize the same number of features.
        num_layers (int): The number of layers of the model.
        pool_size (int, optional): The number of pinned CPU buffers for pulling
            histories and transfering them to GPU.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj:`None`)
        buffer_size (int, optional): The size of pinned CPU buffers, i.e. the
            maximum number of out-of-mini-batch nodes pulled at once.
            Needs to be set in order to make use of asynchronous memory
            transfers. (default: :obj=`None`)
    """
    def __init__(self, num_nodes: int, hidden_channels: int, num_layers: int,
                 pool_size: Optional[int] = None,
                 buffer_size: Optional[int] = None, device=None, in_channels=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers - 1 if pool_size is None else pool_size
        self.buffer_size = buffer_size

        # set histories dimension all with hidden featuer dimension
        self.histories = torch.nn.ModuleList([
            # History(num_nodes, max(in_channels, hidden_channels), device)
            History(num_nodes, hidden_channels, device)
            for i in range(num_layers)
            # for _ in range(num_layers - 1)
        ])
        self.pool: Optional[AsyncIOPool] = None

        
        self.histories_ag = torch.nn.ModuleList([
            # History(num_nodes, max(in_channels, hidden_channels), device)
            History(num_nodes, hidden_channels, device)
            for i in range(num_layers)
        ])
        self.pool_ag: Optional[AsyncIOPool] = None
        
        
        
        self._async = False
        self.__out: Optional[Tensor] = None

    @property
    def emb_device(self):
        return self.histories[0].emb.device

    @property
    def device(self):
        return self.histories[0]._device

    def _apply(self, fn: Callable) -> None:
        super()._apply(fn)
        # We only initialize the AsyncIOPool in case histories are on CPU:
        # print('We only initialize the AsyncIOPool in case histories are on CPU:') #as
        if (str(self.emb_device) == 'cpu' and str(self.device)[:4] == 'cuda'
                and self.pool_size is not None
                and self.buffer_size is not None):
            self.pool = AsyncIOPool(self.pool_size, self.buffer_size,
                                    self.histories[0].embedding_dim) # original, when histories[0].dim is hidden_channels
                                    # self.histories[1].embedding_dim) # use hidden_channels
                                    # max(self.histories[0].embedding_dim, self.histories[1].embedding_dim) )
            self.pool.to(self.device)

            
            # 20250112: add pool_ag for M_ag
            self.pool_ag = AsyncIOPool(self.pool_size, self.buffer_size,
                                    self.histories_ag[0].embedding_dim) # original,  when histories[0].dim is hidden_channels
                                    # self.histories_ag[1].embedding_dim) # use hidden_channels
                                    # max(self.histories[0].embedding_dim, self.histories[1].embedding_dim) )
            self.pool_ag.to(self.device)
            
            log.info(f'Initialized AsyncIOPool with pool_size={self.pool_size}, buffer_size={self.buffer_size}, embedding_dim={self.histories[0].embedding_dim}')
            log.info(f'Initialized AsyncIOPool_ag with pool_size={self.pool_size}, buffer_size={self.buffer_size}, embedding_dim={self.histories_ag[0].embedding_dim}')
            
        return self

    def reset_parameters(self): 
        for history in self.histories:
            history.reset_parameters()

    def __call__(
        self,
        x: Optional[Tensor] = None,
        adj_t: Optional[SparseTensor] = None,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
        loader: EvalSubgraphLoader = None,
        drift_norm: int = 2,
        aggregate_combined: bool = True,
        use_aggregation: bool = True,
        **kwargs,
    ) -> Tensor:
        r"""Enhances the call of forward propagation by immediately start
        pulling historical embeddings for all layers asynchronously.
        After forward propogation is completed, the push of node embeddings to
        the histories will be synchronized.

        For example, given a mini-batch with node indices
        :obj:`n_id = [0, 1, 5, 6, 7, 3, 4]`, where the first 5 nodes
        represent the mini-batched nodes, and nodes :obj=`3` and :obj=`4`
        denote out-of-mini-batched nodes (i.e. the 1-hop neighbors of the
        mini-batch that are not included in the current mini-batch), then
        other input arguments should be given as:

        .. code-block:: python

            batch_size = 5
            offset = [0, 5]
            count = [2, 3]

        Args:
            x (Tensor, optional): Node feature matrix. (default: :obj=`None`)
            adj_t (SparseTensor, optional) The sparse adjacency matrix.
                (default: :obj=`None`)
            batch_size (int, optional): The in-mini-batch size of nodes.
                (default: :obj=`None`)
            n_id (Tensor, optional): The global indices of mini-batched and
                out-of-mini-batched nodes. (default: :obj=`None`)
            offset (Tensor, optional): The offset of mini-batched nodes inside
                a utilize a contiguous memory layout. (default: :obj=`None`)
            count (Tensor, optional): The number of mini-batched nodes inside a
                contiguous memory layout. (default: :obj=`None`)
            loader (EvalSubgraphLoader, optional): A subgraph loader used for
                evaluating the given GNN in a layer-wise fashsion.
            drift_norm (int, optional): The norm type for drift calculation.
                (default: :obj=`2`)
            aggregate_combined (bool, optional): Whether to aggregate combined
                embeddings. (default: :obj=`True`)
            use_aggregation (bool, optional): Whether to use aggregation in
                forward propagation. (default: :obj=`True`)
        """

        if loader is not None:
            # print("mini_inference called")
            return self.mini_inference(loader, use_aggregation)

        # We only perform asynchronous history transfer in case the following
        # conditions are met:
        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)
        
        if (batch_size is not None and not self._async
                and str(self.emb_device) == 'cpu'
                and str(self.device)[:4] == 'cuda'):
            warnings.warn('Asynchronous I/O disabled, although history and '
                          'model sit on different devices.')

        # pull
        time_pool_pull = 0
        if self._async:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for hist in self.histories:
                self.pool.async_pull(hist.emb, None, None, n_id[batch_size:])
            end_event.record()
            
            torch.cuda.synchronize()
            time_pool_pull = start_event.elapsed_time(end_event) / 1000.0  # Convert milliseconds to seconds



        # forward
        # out = self.forward(x, adj_t, batch_size, n_id, offset, count, **kwargs)
        out, time_forward_histories_movement = self.forward(x, adj_t, drift_norm, aggregate_combined, use_aggregation, batch_size, n_id, offset, count, **kwargs)

        # push
        time_pool_push = 0
        if self._async:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for hist in self.histories:
                self.pool.synchronize_push()
            end_event.record()

            torch.cuda.synchronize()
            time_pool_push = start_event.elapsed_time(end_event) / 1000.0  # Convert milliseconds to seconds

        self._async = False

        # return out
        return {
            'out': out,
            'time_pool_pull': time_pool_pull,
            'time_pool_push': time_pool_push,
            'time_forward_histories_movement': time_forward_histories_movement,
            # 'time_forward_total': time_forward_total
            'time_forward_total': 0
        }

    def VR_call(
        self,
        x: Optional[Tensor] = None,
        adj_t: Optional[SparseTensor] = None,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
        loader: EvalSubgraphLoader = None,
        debug_flag: bool = False,
        drift_norm: int = 2,
        epoch: int = 0,
        batch_idx: int = 0,
        **kwargs,
    ) -> Tensor:
        r"""Enhances the call of forward propagation by immediately start
        pulling historical embeddings for all layers asynchronously.
        After forward propogation is completed, the push of node embeddings to
        the histories will be synchronized.

        For example, given a mini-batch with node indices
        :obj:`n_id = [0, 1, 5, 6, 7, 3, 4]`, where the first 5 nodes
        represent the mini-batched nodes, and nodes :obj=`3` and :obj=`4`
        denote out-of-mini-batched nodes (i.e. the 1-hop neighbors of the
        mini-batch that are not included in the current mini-batch), then
        other input arguments should be given as:

        .. code-block:: python

            batch_size = 5
            offset = [0, 5]
            count = [2, 3]

        Args:
            x (Tensor, optional): Node feature matrix. (default: :obj=`None`)
            adj_t (SparseTensor, optional) The sparse adjacency matrix.
                (default: :obj=`None`)
            batch_size (int, optional): The in-mini-batch size of nodes.
                (default: :obj=`None`)
            n_id (Tensor, optional): The global indices of mini-batched and
                out-of-mini-batched nodes. (default: :obj=`None`)
            offset (Tensor, optional): The offset of mini-batched nodes inside
                a utilize a contiguous memory layout. (default: :obj=`None`)
            count (Tensor, optional): The number of mini-batched nodes inside a
                contiguous memory layout. (default: :obj=`None`)
            loader (EvalSubgraphLoader, optional): A subgraph loader used for
                evaluating the given GNN in a layer-wise fashsion.
        """

        # ipdb.set_trace() # check loader

        if loader is not None:
            return self.mini_inference(loader)

        # We only perform asynchronous history transfer in case the following
        # conditions are met:
        # ambershek 实际上运行会print到下一行
        # print("perform asynchronous history transfer")
        # ipdb.set_trace() # check self._async in VR_call()
        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)

        # ipdb.set_trace() # check self._async
        
        if (batch_size is not None and not self._async
                and str(self.emb_device) == 'cpu'
                and str(self.device)[:4] == 'cuda'):
            warnings.warn('Asynchronous I/O disabled, although history and '
                          'model sit on different devices.')
            # pass

        # pull
        time_pool_pull = 0
        if self._async:
            ### pull M_in and M_ag of batches
            for i in range(len(self.histories)):              
                # self.pool.async_pull(self.histories[i].emb, None, None, n_id[:batch_size]) # pull M_in of batches, indexed by n_id[:batch_size]
                # self.pool_ag.async_pull(self.histories_ag[i].emb, None, None, n_id[:batch_size]) # pull M_ag of batches, indexed by n_id[:batch_size]

                self.pool.async_pull(self.histories[i].emb, offset, count, torch.empty(0)) # pull M_in of batches, by chunks indexed by offset and count
                self.pool_ag.async_pull(self.histories_ag[i].emb, offset, count, torch.empty(0)) # pull M_ag of batches, by chunks indexed by offset and count        

        """
        # forward
        # out = self.forward(x, adj_t, batch_size, n_id, offset, count, **kwargs)
        # out, time_push_and_pull_all = self.VR_forward(x, adj_t, batch_size, n_id, offset, count, **kwargs)
        # out, time_push_and_pull_all, out_no_VR = self.VR_forward(x, adj_t, batch_size, n_id, offset, count, **kwargs)
        
        # if debug_flag:
        #     # print(f'x = {x}')
        #     # print(f'adj_t = {adj_t}')
        #     out, time_data_movement, out_no_VR = self.VR_forward_debug(x, adj_t, batch_size, n_id, offset, count, **kwargs)
        # else:
        #     # print(f'x = {x}')
        #     # print(f'adj_t = {adj_t}')
        #     time_forward_total = time.perf_counter()
        #     out, time_forward_histories_movement, num_in_batch_neighbors, num_out_batch_neighbors, out_no_VR, drift = self.VR_forward(x, adj_t, drift_norm, batch_size, n_id, offset, count, **kwargs)
        #     time_forward_total = time.perf_counter() - time_forward_total
        """

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()

        # ipdb.set_trace() # check VR_forward() in VR_call()
        out, time_forward_histories_movement, num_in_batch_neighbors, num_out_batch_neighbors = self.VR_forward(x, adj_t, drift_norm, epoch, batch_idx,  batch_size, n_id, offset, count, **kwargs)

        # end_event.record()
        # torch.cuda.synchronize()
        # time_forward_total=start_event.elapsed_time(end_event) / 1000.0  # Convert milliseconds to seconds

        # push
        time_pool_push = 0
        """
        # 20250112 unnecessary in current version
        if self._async:
            t = time.perf_counter()
            for hist in self.histories:
                self.pool.synchronize_push()
            time_pool_push = time.perf_counter() - t
        """

        self._async = False

        # return out
        # return out, time_data_movement
        return {
            'out': out,
            'time_pool_pull': time_pool_pull,
            'time_pool_push': time_pool_push,
            'time_forward_histories_movement': time_forward_histories_movement,
            # 'time_forward_total': time_forward_total,
            'time_forward_total': 0,
            'num_in_batch_neighbors': num_in_batch_neighbors,
            'num_out_batch_neighbors': num_out_batch_neighbors,
        }
    
    def push_and_pull(self, history, x: Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[Tensor] = None,
                      offset: Optional[Tensor] = None,
                      count: Optional[Tensor] = None) -> Tensor:
        r"""Pushes and pulls information from :obj:`x` to :obj:`history` and
        vice versa."""

        if n_id is None and x.size(0) != self.num_nodes:
            # print('push_and_pull ... do nothing')
            return x  # Do nothing...

        if n_id is None and x.size(0) == self.num_nodes:
            # GCN@CORA in this branch--1
            t = time.perf_counter()
            # print('push_and_pull ... push without global indices')
            history.push(x)
            time_push_and_pull = time.perf_counter() - t
            # return x
            return x, time_push_and_pull

        assert n_id is not None
        # print('push_and_pull ... global indices available')

        if batch_size is None:
            print("batch_size is None")
            history.push(x, n_id)
            return x

        # push synchronize
        # GCN@CORA in this branch--2
        if not self._async:
            # print('push_and_pull ... push synchronize')
            
            t = time.perf_counter()
            history.push(x[:batch_size], n_id[:batch_size], offset, count)     
            t_push = time.perf_counter() - t

            t = time.perf_counter()      
            h = history.pull(n_id[batch_size:])
            t_pull = time.perf_counter() - t

            # ipdb.set_trace()
            # time_push_and_pull = time.perf_counter() - t
            time_push_and_pull = t_push + t_pull

            return torch.cat([x[:batch_size], h], dim=0), time_push_and_pull
        
        # push asynchronize
        # PNA@arxiv会进入这一分支
        else:
            # print('push_and_pull ... push asynchronize')
            t = time.perf_counter()
            out = self.pool.synchronize_pull()[:n_id.numel() - batch_size] # OB neighbors
            # ipdb.set_trace()
            # if out.size(1) > self.hidden_channels:
            #     out = out[:, :self.hidden_channels].contiguous()

            # Check if the dimension of x[:batch_size] matches history.emb
            # ipdb.set_trace()
            # if x[:batch_size].size(1) < history.emb.size(1):
            #     # Expand the dimension of x[:batch_size] to match history.emb
            #     expanded_x = torch.zeros((x[:batch_size].size(0), history.emb.size(1)), device=x.device)
            #     expanded_x[:, :x[:batch_size].size(1)] = x[:batch_size]
            #     x_to_push = expanded_x
            # else:
            #     x_to_push = x[:batch_size]

            # ipdb.set_trace()
            self.pool.async_push(x[:batch_size], offset, count, history.emb)  # push current batch of x to history

            out = torch.cat([x[:batch_size], out], dim=0)
            self.pool.free_pull()
            time_push_and_pull = time.perf_counter() - t
            # print(f'----------Pushed and pulled! [{time_push_and_pull:.6f}s]')
            # return out
            return out, time_push_and_pull

    def push_only(self, history, x: Tensor,
                batch_size: Optional[int] = None,
                n_id: Optional[Tensor] = None,
                offset: Optional[Tensor] = None,
                count: Optional[Tensor] = None) -> Tensor:
        """
        Pushes updated embeddings to `history` without pulling historical embeddings.
        """

        if n_id is None and x.size(0) != self.num_nodes:
            # Do nothing if global node indices are unavailable and not full graph size.
            print('push_only ... do nothing')
            return x

        if n_id is None and x.size(0) == self.num_nodes:
            # Push embeddings for the entire graph (full-batch case).
            t = time.perf_counter()
            history.push(x)
            time_push = time.perf_counter() - t
            return x, time_push

        assert n_id is not None

        if batch_size is None:
            # Push embeddings for all nodes in `n_id` (used for cases without explicit batching).
            t = time.perf_counter()
            history.push(x, n_id)
            time_push = time.perf_counter() - t
            return x, time_push

        if not self._async:
            # Push for batched data
            t = time.perf_counter()
            history.push(x[:batch_size], n_id[:batch_size], offset, count)
            time_push = time.perf_counter() - t
            # Skip pulling historical embeddings for out-of-batch neighbors
            return x[:batch_size], time_push
        else:
            t = time.perf_counter()
            self.pool.async_push(x[:batch_size], offset, count, history.emb)
            time_push = time.perf_counter() - t
            return x[:batch_size], time_push


    @property
    def _out(self):
        if self.__out is None:
            self.__out = torch.empty(self.num_nodes, self.out_channels,
                                     pin_memory=True)
        return self.__out

    @torch.no_grad()
    def mini_inference(self, loader: SubgraphLoader, use_aggregation = True) -> Tensor:
        # We iterate over the loader in a layer-wise fashsion.
        # In order to re-use some intermediate representations, we maintain a
        # `state` dictionary for each individual mini-batch.
        loader = [sub_data + ({}, ) for sub_data in loader]

        # We push the outputs of the first layer to the history:
        for data, batch_size, n_id, offset, count, state in loader:
            x = data.x.to(self.device)
            # ipdb.set_trace()
            adj_t = data.adj_t.to(self.device)
            # print(adj_t)
            out = self.forward_layer(0, x, adj_t, state, use_aggregation)[:batch_size]
            # ipdb.set_trace() # check x and out shape

            """
            M_in_layer0 = x[:batch_size] # only keep the batch_size rows
            if M_in_layer0.size(1) < self.histories[0].emb.size(1):
                M_in_layer0_expanded = torch.zeros((M_in_layer0.size(0), self.histories[0].emb.size(1)), device=self.device)
                M_in_layer0_expanded[:, :M_in_layer0.size(1)] = M_in_layer0
                M_in_layer0 = M_in_layer0_expanded
            # ipdb.set_trace()
            self.pool.async_push(M_in_layer0, offset, count, self.histories[0].emb) # push x to history
            """

            # ipdb.set_trace()
            # gcn@cora上没有并行的pool，没法跑下面这行async_push()
            # self.pool.async_push(out, offset, count, self.histories[0].emb) # push out to history
            # if out.size(1) < self.histories[1].emb.size(1):
            #     out_expanded = torch.zeros((out.size(0), self.histories[1].emb.size(1)), device=self.device)
            #     out_expanded[:, :out.size(1)] = out
            #     out = out_expanded
            # ipdb.set_trace()
            # self.pool.async_push(out, offset, count, self.histories[0].emb)  # original change
            self.pool.async_push(out, offset, count, self.histories[1].emb)  # index change
            
            
        self.pool.synchronize_push()

        # ipdb.set_trace()

        # for i in range(1, len(self.histories)): # original index
        for i in range(1, len(self.histories)-1): # index change
            # ipdb.set_trace()
            # Pull the complete layer-wise history:
            for _, batch_size, n_id, offset, count, _ in loader:
                # self.pool.async_pull(self.histories[i - 1].emb, offset, count, n_id[batch_size:]) # original index
                self.pool.async_pull(self.histories[i].emb, offset, count, n_id[batch_size:]) # index change
                # ipdb.set_trace() # check the pool after pulling this batch

            # ipdb.set_trace() # check the complete layer-wise history

            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for batch, batch_size, n_id, offset, count, state in loader:
                adj_t = batch.adj_t.to(self.device)
                x = self.pool.synchronize_pull()[:n_id.numel()]
                # check and crop
                # if x.size(1) > self.hidden_channels:
                #     x = x[:, :self.hidden_channels]
                out = self.forward_layer(i, x, adj_t, state, use_aggregation)[:batch_size]

                # ipdb.set_trace()
                # if out.size(1) < self.histories[i+1].emb.size(1):
                #     out_expanded = torch.zeros((out.size(0), self.histories[i+1].emb.size(1)), device=self.device)
                #     out_expanded[:, :out.size(1)] = out
                #     out = out_expanded
                # ipdb.set_trace()
                # self.pool.async_push(out, offset, count, self.histories[i].emb) # original index
                self.pool.async_push(out, offset, count, self.histories[i+1].emb) # index change
                self.pool.free_pull()
            self.pool.synchronize_push()

        # We pull the histories from the last layer:
        for _, batch_size, n_id, offset, count, _ in loader:
            self.pool.async_pull(self.histories[-1].emb, offset, count,
                                 n_id[batch_size:])

        # And compute final output embeddings, which we write into a private
        # output embedding matrix:
        for batch, batch_size, n_id, offset, count, state in loader:
            adj_t = batch.adj_t.to(self.device)
            x = self.pool.synchronize_pull()[:n_id.numel()]
            # ipdb.set_trace()
            # check and crop
            # if x.size(1) > self.hidden_channels:
            #     x = x[:, :self.hidden_channels].contiguous()
            # ipdb.set_trace()
            out = self.forward_layer(self.num_layers - 1, x, adj_t, state, use_aggregation)[:batch_size]
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
        self.pool.synchronize_push()

        return self._out


    @torch.no_grad()
    def mini_inference_vr(self, loader: SubgraphLoader, use_aggregation = True) -> Tensor:
        raise NotImplementedError
    
    # not used in main.py, for archive
    @torch.no_grad()
    def update_periodical_hist_layerwise(self, loader: SubgraphLoader, use_aggregation = True) -> Tensor:
        r"""An implementation of layer-wise evaluation of GNNs.
        For each individual layer and mini-batch, :meth:`forward_layer` takes
        care of computing the next state of node embeddings.
        Additional state (such as residual connections) can be stored in
        a `state` directory."""

        # We iterate over the loader in a layer-wise fashsion.
        # In order to re-use some intermediate representations, we maintain a
        # `state` dictionary for each individual mini-batch.
        full_loader = loader
        loader = [sub_data + ({}, ) for sub_data in loader]

        # We push the outputs of the first layer to the history:
        for data, batch_size, n_id, offset, count, state in loader:
            adj_t = data.adj_t.to(self.device)
            x = data.x.to(self.device)

            # TODO Dreft, to check if the following code is correct
            M_in_layer0 = x[:batch_size] # only keep the batch_size rows
            M_in_layer0 = torch.zeros((batch_size, self.hidden_channels), device=self.device)
            M_in_layer0[:, :x.size(1)] = x[:batch_size]

            M_ag_layer0 = adj_t @ x # shape from (IB+OB) to IB
            M_ag_layer0_expanded = torch.zeros((M_ag_layer0.size(0), self.hidden_channels), device=self.device)
            M_ag_layer0_expanded[:, :M_ag_layer0.size(1)] = M_ag_layer0
            M_ag_layer0 = M_ag_layer0_expanded

            # ipdb.set_trace() # check M_ag_layer0.shape and self.histories_ag[0].emb.shape
            self.pool.async_push(M_in_layer0, offset, count, self.histories[0].emb)
            self.pool_ag.async_push(M_ag_layer0, offset, count, self.histories_ag[0].emb)

            M_in_layer1 = self.forward_layer(0, x, adj_t, state, use_aggregation) # shape from (IB+OB) to IB
            self.pool.async_push(M_in_layer1, offset, count, self.histories[1].emb) # push to histories[1]

        self.pool.synchronize_push()
        self.pool_ag.synchronize_push() # M_ag is not used in this function, so can be synchronized at the end?
        ipdb.set_trace()


        for i in range(1, len(self.histories)-1):
            # print(f'Update periodical history for layer {i}')
            # Pull the complete layer-wise history (M_in):
            for _, batch_size, n_id, offset, count, _ in loader:
                self.pool.async_pull(self.histories[i].emb, offset, count, n_id[batch_size:])

            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for batch, batch_size, n_id, offset, count, state in loader:
                adj_t = batch.adj_t.to(self.device)
                x = self.pool.synchronize_pull()[:n_id.numel()]
                # ipdb.set_trace() # check x.shape
                
                # TODO Draft, to check if the following code is correct
                # M_in_this_layer = x[:batch_size]
                M_ag_this_layer = adj_t @ x
                # self.pool.async_push(M_in_this_layer, offset, count, self.histories[i].emb)
                # ipdb.set_trace() # check M_ag_this_layer.shape
                self.pool_ag.async_push(M_ag_this_layer, offset, count, self.histories_ag[i].emb)

                M_in_next_layer = self.forward_layer(i, x, adj_t, state, use_aggregation)[:batch_size]
                self.pool.async_push(M_in_next_layer, offset, count, self.histories[i+1].emb)
                self.pool.free_pull()
            self.pool.synchronize_push()
            self.pool_ag.synchronize_push()
            ipdb.set_trace()

        # We pull the histories from the last layer:
        for _, batch_size, n_id, offset, count, _ in loader:
            self.pool.async_pull(self.histories[-1].emb, offset, count,
                                 n_id[batch_size:])

        # And compute final output embeddings, which we write into a private
        # output embedding matrix:
        for batch, batch_size, n_id, offset, count, state in loader:
            adj_t = batch.adj_t.to(self.device)
            x = self.pool.synchronize_pull()[:n_id.numel()]
            out = self.forward_layer(self.num_layers - 1, x, adj_t, state, use_aggregation)[:batch_size]
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
        self.pool.synchronize_push()
        self.pool_ag.synchronize_push()

        return self._out

    @torch.no_grad()
    def forward_layer(self, layer: int, x: Tensor, adj_t: SparseTensor,
                      state: Dict[str, Any]) -> Tensor:
        raise NotImplementedError


    # def VR_forward(self, x: Tensor, adj_t: SparseTensor, n_id: Tensor=None, *args) -> Tensor:
    def VR_forward(self, x: Tensor, adj_t: SparseTensor, full_adj_t: SparseTensor, *args) -> Tensor:
        raise NotImplementedError
    

    def VR_forward_debug(self, x: Tensor, adj_t: SparseTensor, full_adj_t: SparseTensor, *args) -> Tensor:
        raise NotImplementedError