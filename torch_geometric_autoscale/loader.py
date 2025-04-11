from typing import NamedTuple, List, Tuple

import time

import torch
# torch.set_default_dtype(torch.float64)

from torch import Tensor
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.data import Data

#  
from torch_sparse import sample_adj
import random
import sys
import time

import ipdb

from typing import Tuple

relabel_fn = torch.ops.torch_geometric_autoscale.relabel_one_hop
relabel_fn_within_batch = torch.ops.torch_geometric_autoscale.relabel_one_hop_within_batch

from torch_geometric.sampler import NeighborSampler, NodeSamplerInput


import logging
logger = logging.getLogger(__name__)

def sample_neighbors(src: SparseTensor, num_neighbors: int, unsampled_n_id: Tensor = None)-> Tuple[SparseTensor, torch.Tensor]:
    if num_neighbors < 0: # no sampling
        return src, unsampled_n_id

    rowptr, col, value = src.csr()
    num_target_nodes = rowptr.size(0)
        
    splited_cols = torch.tensor_split(input=col, tensor_indices_or_sections=rowptr[1:])
    if value != None:
        splited_values = torch.tensor_split(input=value, tensor_indices_or_sections=rowptr[1:])
    
    max_nums_neighbors = [int(splited_col.size(0)) for splited_col in splited_cols]
    max_nums_neighbors = Tensor(max_nums_neighbors).int()
    set_nums_neighbors = (torch.ones_like(max_nums_neighbors) * num_neighbors).int()
    nums_neighbors_to_sample = torch.minimum(input=max_nums_neighbors, other=set_nums_neighbors)

    sampled_rowptr = []
    sampled_row = []
    sampled_col = []
    if value != None:
        sampled_value = []
    
    ptr = 0
    for i in range(num_target_nodes):
        sampled_rowptr.append(ptr)
                
        max_num_neighbors = max_nums_neighbors[i]
        num_neighbors_to_sample = nums_neighbors_to_sample[i]
        sampled_row.append(torch.ones(num_neighbors_to_sample)*i)
        
        splited_col = splited_cols[i]
        sampled_index, _  = torch.sort(torch.LongTensor(random.sample(range(max_num_neighbors), num_neighbors_to_sample))) 
        sampled_splited_col = torch.index_select(input=splited_col, dim=0, index=sampled_index)
        sampled_col.append(sampled_splited_col)
        
        if value != None:
            splited_value = splited_values[i] 
            sampled_splited_value = torch.index_select(input=splited_value, dim=0, index=sampled_index)
            sampled_value.append(sampled_splited_value)

        ptr += int(num_neighbors_to_sample)

    sampled_rowptr = torch.LongTensor(sampled_rowptr)
    
    sampled_row = torch.concat(sampled_row).long()
    
    sampled_col = torch.concat(sampled_col)
    
    if value != None:
        sampled_value = torch.concat(sampled_value)
    else:
        sampled_value = None

    sampled_n_id = unsampled_n_id
   
    out = SparseTensor(rowptr=sampled_rowptr, row=sampled_row, col=sampled_col, 
                        value = sampled_value,
                        sparse_sizes=(num_target_nodes-1, sampled_n_id.size(0)),
                        is_sorted=True, 
                        trust_data=True)

    return out, sampled_n_id

class SubData(NamedTuple):
    data: Data
    batch_size: int
    n_id: Tensor  # The indices of mini-batched nodes
    offset: Tensor  # The offset of contiguous mini-batched nodes
    count: Tensor  # The number of contiguous mini-batched nodes
    
    def to(self, *args, **kwargs):
        return SubData(self.data.to(*args, **kwargs), self.batch_size,
                       self.n_id, self.offset, self.count)
        
class SubData_with_full_data(NamedTuple):
    data: Data
    batch_size: int
    n_id: Tensor  # The indices of mini-batched nodes
    offset: Tensor  # The offset of contiguous mini-batched nodes
    count: Tensor  # The number of contiguous mini-batched nodes
    data_ori: SparseTensor # The adj before neighbor sampling
    shuffled_batch_id: list # shuffled batch_id of this batch
    
    def to(self, *args, **kwargs):
        return SubData_with_full_data(self.data.to(*args, **kwargs), self.batch_size,
                       self.n_id, self.offset, self.count, self.data_ori, self.shuffled_batch_id)

class SubgraphLoader(DataLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors)."""
    def __init__(self, data: Data, ptr: Tensor, batch_size: int = 1, 
                 bipartite: bool = True, log: bool = True, num_neighbors=-1, type='eval', IB=False, **kwargs):

        self.data = data
        self.ptr = ptr
        self.bipartite = bipartite
        self.log = log
        self.num_neighbors = num_neighbors # 
        if 'shuffle' in kwargs:
            self.shuffle = kwargs['shuffle']
        self.shuffled_batch_id = []  # To store shuffled batch indices

        n_id = torch.arange(data.num_nodes)
        batches = n_id.split((ptr[1:] - ptr[:-1]).tolist())
        # print(f'batches: {batches}')
        batches = [(i, batches[i]) for i in range(len(batches))]
        # ipdb.set_trace() 0915
        
        logger.info(f'batch_size in SubgraphLoader: {batch_size}') 
        
        if type == 'train':
            if IB:
                logger.info('Building train loader with compute_subgraph_IB()')
                super().__init__(batches, batch_size=batch_size, collate_fn=self.compute_subgraph_IB, **kwargs)
            else:
                logger.info('Building train loader with compute_subgraph_NS()')
                super().__init__(batches, batch_size=batch_size, collate_fn=self.compute_subgraph_NS, **kwargs)            

        else:  
            logger.info('Building eval loader with GAS original implementation')
            if batch_size > 1:
                super().__init__(batches, batch_size=batch_size,
                             collate_fn=self.compute_subgraph, **kwargs)
            else: # If `batch_size=1`, we pre-process the subgraph generation:
                
                logger.info('If `batch_size=1`, we pre-process the subgraph generation')
                if log:
                    t = time.perf_counter()
                    print('Pre-processing subgraphs...', end=' ', flush=True)

                data_list = list(
                    DataLoader(batches, batch_size=batch_size, collate_fn=self.compute_subgraph, **kwargs)
                )

                if log:
                    print(f'Done! [{time.perf_counter() - t:.2f}s]')
                
                super().__init__(data_list, batch_size=batch_size, collate_fn=lambda x: x[0], **kwargs)

    def compute_subgraph(self, batches: List[Tuple[int, Tensor]]) -> SubData:
        print('Using original collate_fn')
        batch_ids, n_ids = zip(*batches)
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)
        batch_size = n_id.numel() 
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset) 
        rowptr, col, value = self.data.adj_t.csr()
        rowptr, col, value, n_id = relabel_fn(rowptr, col, value, n_id, self.bipartite)

        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                             sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                             is_sorted=True)

        data = self.data.__class__(adj_t=adj_t)
        for k, v in self.data:
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v.index_select(0, n_id)

        return SubData(data, batch_size, n_id, offset, count)

    def compute_subgraph_IB(self, batches: List[Tuple[int, Tensor]]) -> SubData:
        batch_ids, n_ids = zip(*batches)
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)
        batch_size = n_id.numel() #  
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset) 
        rowptr, col, value = self.data.adj_t.csr()
        rowptr, col, value, n_id = relabel_fn_within_batch(rowptr, col, value, n_id, self.bipartite)

        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                             sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                             is_sorted=True)


        data = self.data.__class__(adj_t=adj_t)
        for k, v in self.data:
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v.index_select(0, n_id)

        return SubData(data, batch_size, n_id, offset, count)

        
    def compute_subgraph_NS(self, batches: List[Tuple[int, Tensor]]) -> SubData_with_full_data:
        batch_ids, n_ids = zip(*batches)        
        n_id_before = torch.cat(n_ids, dim=0) 
        batch_id = torch.tensor(batch_ids)
        batch_size = n_id_before.numel() # for debug
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset)
              
        rowptr_before, col_before, value_before = self.data.adj_t.csr()

        self.old_n_id = n_id_before
              
        rowptr, col, value, n_id = relabel_fn(rowptr_before, col_before, value_before, n_id_before, self.bipartite) # one-hop

        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                            sparse_sizes=(rowptr.numel()-1, n_id.numel()),
                             is_sorted=True)

        sampled_adj_t, sampled_n_id = sample_neighbors(src=adj_t, num_neighbors=self.num_neighbors, unsampled_n_id=n_id, target_nodes_id = self.old_n_id)

        data_sampled = self.data.__class__(adj_t=sampled_adj_t) 
        for k, v in self.data:
            if isinstance(v, Tensor):
                if v.size(0) == self.data.num_nodes:
                    data_sampled[k] = v.index_select(0, sampled_n_id)

        self.sampled_n_id = sampled_n_id
        self.sampled_adj_t = sampled_adj_t
        
        return SubData(data_sampled, batch_size, sampled_n_id, offset, count) 

        

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __iter__(self):
        self.shuffled_batch_id = []

        iterator = super().__iter__()

        for batch_data in iterator:
            if hasattr(self, 'shuffle') and self.shuffle:
                indices = batch_data[-1] if isinstance(batch_data, tuple) else batch_data
                self.shuffled_batch_id.append(indices) 
            yield batch_data 

        

class EvalSubgraphLoader(SubgraphLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors).
    In contrast to :class:`SubgraphLoader`, this loader does not generate
    subgraphs from randomly sampled mini-batches, and should therefore only be
    used for evaluation.
    """
    def __init__(self, data: Data, ptr: Tensor, batch_size: int = 1,
                 bipartite: bool = True, log: bool = True, **kwargs):
        # 
        # print('Initializing EvalSubgraphLoader...')

        ptr = ptr[::batch_size]
        if int(ptr[-1]) != data.num_nodes:
            ptr = torch.cat([ptr, torch.tensor([data.num_nodes])], dim=0)

        super().__init__(data=data, ptr=ptr, batch_size=1, bipartite=bipartite,
                         log=log, shuffle=False, num_workers=0, **kwargs)

