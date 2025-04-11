from typing import NamedTuple, List, Tuple

import time

import torch
# torch.set_default_dtype(torch.float64)

from torch import Tensor
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.data import Data

# ambershek 
from torch_sparse import sample_adj
import random
import sys
import time

import ipdb

from typing import Tuple

relabel_fn = torch.ops.torch_geometric_autoscale.relabel_one_hop
relabel_fn_within_batch = torch.ops.torch_geometric_autoscale.relabel_one_hop_within_batch

from torch_geometric.sampler import NeighborSampler, NodeSamplerInput

# ambershhek for logging
import logging
# A logger for this file
logger = logging.getLogger(__name__)

# static_num_neighbors = 2

# target nodes保持不变
# 对各target nodes的neighbors进行采样
def sample_neighbors(src: SparseTensor, num_neighbors: int, unsampled_n_id: Tensor = None)-> Tuple[SparseTensor, torch.Tensor]:
    if num_neighbors < 0: # no sampling
        return src, unsampled_n_id

    rowptr, col, value = src.csr()
    num_target_nodes = rowptr.size(0)
        
    splited_cols = torch.tensor_split(input=col, tensor_indices_or_sections=rowptr[1:])
    # print(f'splited_cols: {splited_cols}\n')
    if value != None:
        splited_values = torch.tensor_split(input=value, tensor_indices_or_sections=rowptr[1:])
    
    max_nums_neighbors = [int(splited_col.size(0)) for splited_col in splited_cols]
    max_nums_neighbors = Tensor(max_nums_neighbors).int()
    # print(max_nums_neighbors, max_nums_neighbors.size())
    # print(f'max_num sum: {int(torch.sum(max_nums_neighbors))}')
    set_nums_neighbors = (torch.ones_like(max_nums_neighbors) * num_neighbors).int()
    nums_neighbors_to_sample = torch.minimum(input=max_nums_neighbors, other=set_nums_neighbors)
    # print(nums_neighbors_to_sample, nums_neighbors_to_sample.size())
    
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
        
        splited_col = splited_cols[i] # neighbor list to sample
        # print(f'BEFORE splited_col: {splited_col}')
        # print(f'BEFORE splited_col.shape: {splited_col.shape}')
        # 在该target nodes对应的splited_col中随机选取num_neighbors_to_sample个
        sampled_index, _  = torch.sort(torch.LongTensor(random.sample(range(max_num_neighbors), num_neighbors_to_sample))) 
        sampled_splited_col = torch.index_select(input=splited_col, dim=0, index=sampled_index)
        sampled_col.append(sampled_splited_col)
        
        if value != None:
            splited_value = splited_values[i] 
            sampled_splited_value = torch.index_select(input=splited_value, dim=0, index=sampled_index)
            sampled_value.append(sampled_splited_value)
            
        # print(f'BEFORE splited_col: {splited_col}')
        # print(f'AFTER sampled_splited_col: {sampled_splited_col}\n')
        
        ptr += int(num_neighbors_to_sample)

    # sampled_rowptr.pop(-1)
    sampled_rowptr = torch.LongTensor(sampled_rowptr)
    
    sampled_row = torch.concat(sampled_row).long()
    
    sampled_col = torch.concat(sampled_col)
    
    if value != None:
        sampled_value = torch.concat(sampled_value)
    else:
        sampled_value = None
    
    # sampled_n_id = torch.unique(torch.concat((sampled_col, sampled_row), dim=0), dim=0, sorted=False)
    # mask_index_in_col = torch.isin(elements=Tensor(range(unsampled_n_id.numel())), test_elements=col)
    
    # mask_index_in_col = torch.isin(elements=Tensor(range(unsampled_n_id.numel())), test_elements=sampled_col)
    # sampled_n_id = torch.masked_select(unsampled_n_id, mask_index_in_col)
    sampled_n_id = unsampled_n_id
       
    # import ipdb
    # ipdb.set_trace()    
    out = SparseTensor(rowptr=sampled_rowptr, row=sampled_row, col=sampled_col, 
                        #    value=value,
                        value = sampled_value,
                        #    sparse_sizes=(num_target_nodes-1, int(sampled_col.max()+1)),
                        sparse_sizes=(num_target_nodes-1, sampled_n_id.size(0)),
                        is_sorted=True, 
                        trust_data=True)
    # ipdb.set_trace()
    return out, sampled_n_id
    # return out, torch.LongTensor(range(int(sampled_col.max()+1))) # for debug,别用



# add another parameter 'target_nodes_id', but haven't used it in the function
def sample_neighbors(src: SparseTensor, num_neighbors: int, unsampled_n_id: torch.Tensor = None, target_nodes_id: torch.Tensor = None) -> Tuple[SparseTensor, torch.Tensor]:
    if num_neighbors < 0:  # no sampling
        return src, unsampled_n_id

    rowptr, col, value = src.csr()
    num_target_nodes = rowptr.size(0) - 1  # since rowptr includes one extra element

    device = rowptr.device  # make sure all tensors stay on the same device (GPU/CPU)
    max_nums_neighbors = (rowptr[1:] - rowptr[:-1]).int()  # fast way to get number of neighbors per node
    set_nums_neighbors = torch.full_like(max_nums_neighbors, num_neighbors)
    nums_neighbors_to_sample = torch.min(max_nums_neighbors, set_nums_neighbors).int()

    sampled_rowptr = torch.zeros(num_target_nodes + 1, dtype=torch.long, device=device)
    sampled_row = []
    sampled_col = []
    sampled_value = [] if value is not None else None

    ptr = 0
    for i in range(num_target_nodes):
        sampled_rowptr[i] = ptr
        num_neighbors_to_sample_i = nums_neighbors_to_sample[i].item()

        # Sample neighbors
        start_idx, end_idx = rowptr[i], rowptr[i + 1]
        splited_col = col[start_idx:end_idx]

        sampled_idx = torch.randperm(splited_col.size(0), device=device)[:num_neighbors_to_sample_i]
        sampled_col.append(splited_col[sampled_idx])
        sampled_row.append(torch.full((num_neighbors_to_sample_i,), i, device=device))

        if value is not None:
            splited_value = value[start_idx:end_idx]
            sampled_value.append(splited_value[sampled_idx])

        ptr += num_neighbors_to_sample_i

    sampled_rowptr[-1] = ptr
    sampled_row = torch.cat(sampled_row)
    sampled_col = torch.cat(sampled_col)

    if value is not None:
        sampled_value = torch.cat(sampled_value)
    else:
        sampled_value = None

    out = SparseTensor(rowptr=sampled_rowptr, row=sampled_row, col=sampled_col, value=sampled_value,
                       sparse_sizes=(num_target_nodes, unsampled_n_id.size(0)), is_sorted=True, trust_data=True)

    return out, unsampled_n_id

def sample_neighbors_pyg(src: SparseTensor, num_neighbors: int, n_id: torch.Tensor = None, target_nodes_id: torch.Tensor = None) -> Tuple[SparseTensor, torch.Tensor]:
    # if num_neighbors < 0:  # no sampling
    #     return src, torch.arange(src.size(0))

    # Convert SparseTensor to PyG Data object
    edge_index = torch.stack([src.storage.row(), src.storage.col()], dim=0)
    data = Data(edge_index=edge_index)

    # Initialize NeighborSampler
    sampler = NeighborSampler(
        data,
        num_neighbors=[num_neighbors] * len(target_nodes_id),  # List of num_neighbors for each target node
        subgraph_type='directional',  # Change as needed
        replace=False,  # Whether to allow duplicate neighbors
        disjoint=False  # Whether to sample disjoint subgraphs
    )

    # Construct NodeSamplerInput
    inputs = NodeSamplerInput(input_id=n_id, node=target_nodes_id)

    # Sample neighbors
    out = sampler.sample_from_nodes(inputs)

    # Assuming out contains the sampled edge indices
    # sampled_edges = out.metadata  # Adjust based on your output structure

    # Concatenate all sampled edge indices
    # ipdb.set_trace()
    # sampled_edge_index = torch.cat(sampled_edges, dim=1)
    # sampled_row, sampled_col = sampled_edge_index[0], sampled_edge_index[1]
    ipdb.set_trace()
    sampled_row, sampled_col = out.row, out.col

    # Create SparseTensor from sampled data
    # ipdb.set_trace()
    out_sparse = SparseTensor(row=sampled_row, col=sampled_col,
                               value=None,  # or use sampled_edge_weights if available
                               sparse_sizes=(len(target_nodes_id), src.size(1)),
                               is_sorted=True, trust_data=True)

    return out_sparse, torch.arange(src.size(0))

def are_sparse_tensors_equal(tensor1: SparseTensor, tensor2: SparseTensor) -> bool:
    # Check if sizes are equal
    if tensor1.sparse_sizes != tensor2.sparse_sizes:
        return False

    # Check if number of non-zero elements (nnz) are equal
    if tensor1.nnz() != tensor2.nnz():
        return False

    # Check if row, col, and value tensors are equal
    if not torch.equal(tensor1.row, tensor2.row):
        return False

    if not torch.equal(tensor1.col, tensor2.col):
        return False

    if tensor1.value is not None and tensor2.value is not None:
        if not torch.allclose(tensor1.value, tensor2.value, atol=1e-6):  # Adjust tolerance if needed
            return False
    elif tensor1.value is not None or tensor2.value is not None:
        return False

    return True

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
        # ambershek
        # print('Initializing SubgraphLoader...')

        self.data = data
        self.ptr = ptr
        self.bipartite = bipartite
        self.log = log
        self.num_neighbors = num_neighbors # ambershek
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
                    # DataLoader(batches, batch_size=batch_size, collate_fn=self.compute_subgraph_NS, **kwargs) # ambershek 加入neighbor sampling的
                )

                if log:
                    print(f'Done! [{time.perf_counter() - t:.2f}s]')
                
                super().__init__(data_list, batch_size=batch_size, collate_fn=lambda x: x[0], **kwargs)

    def compute_subgraph(self, batches: List[Tuple[int, Tensor]]) -> SubData:
        print('Using original collate_fn')
        batch_ids, n_ids = zip(*batches)
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)
        
        # print(f'n_id = {n_id}')
        # print(f'n_id.shape = {n_id.shape}')
        # print(f'batch_id = {batch_id}')

        # We collect the in-mini-batch size (`batch_size`), the offset of each
        # partition in the mini-batch (`offset`), and the number of nodes in
        # each partition (`count`)
        batch_size = n_id.numel() # ambershek in-mini-batch的target nodes数量
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset) # ambershek offset开始的count个nodes为in-mini-batch，之后的即为out-mini-batch的1-hop neighbors

        rowptr, col, value = self.data.adj_t.csr()
        # ipdb.set_trace() # check rowptr, col, value before relabel_fn
        rowptr, col, value, n_id = relabel_fn(rowptr, col, value, n_id, self.bipartite)
        # ipdb.set_trace() # check rowptr, col, value after relabel_fn
        

        # ambershek 这个应该是完整的邻接矩阵，2315598大概是arxiv数据集边数1.2M的两倍
        # print(f'self.data.adj_t: {self.data.adj_t}\n')

        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                             sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                             is_sorted=True)
        # print(f'adj_t: {adj_t}')

        data = self.data.__class__(adj_t=adj_t)
        for k, v in self.data:
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v.index_select(0, n_id)

        return SubData(data, batch_size, n_id, offset, count)
        # return SubData_with_full_adj(data, batch_size, n_id, offset, count, adj_t) # for debug，加了个参数

    def compute_subgraph_IB(self, batches: List[Tuple[int, Tensor]]) -> SubData:
        batch_ids, n_ids = zip(*batches)
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)
    
        # We collect the in-mini-batch size (`batch_size`), the offset of each
        # partition in the mini-batch (`offset`), and the number of nodes in
        # each partition (`count`)
        batch_size = n_id.numel() # ambershek in-mini-batch的target nodes数量
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset) # ambershek offset开始的count个nodes为in-mini-batch，之后的即为out-mini-batch的1-hop neighbors

        # ipdb.set_trace()
        rowptr, col, value = self.data.adj_t.csr()
        # ipdb.set_trace() # check rowptr, col, value before relabel_fn
        # rowptr, col, value, n_id = relabel_fn(rowptr, col, value, n_id, self.bipartite)
        rowptr, col, value, n_id = relabel_fn_within_batch(rowptr, col, value, n_id, self.bipartite)
        # ipdb.set_trace() # check rowptr, col, value after relabel_fn
        
        # ambershek 这个应该是完整的邻接矩阵，2315598大概是arxiv数据集边数1.2M的两倍
        # print(f'self.data.adj_t: {self.data.adj_t}\n')


        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                             sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                             is_sorted=True)
        # print(f'adj_t: {adj_t}')

        data = self.data.__class__(adj_t=adj_t)
        for k, v in self.data:
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v.index_select(0, n_id)

        return SubData(data, batch_size, n_id, offset, count)
        # return SubData_with_full_adj(data, batch_size, n_id, offset, count, adj_t) # for debug，加了个参数

        
    def compute_subgraph_NS(self, batches: List[Tuple[int, Tensor]]) -> SubData_with_full_data:
        batch_ids, n_ids = zip(*batches)        
        # n_id = torch.cat(n_ids, dim=0)
        n_id_before = torch.cat(n_ids, dim=0) # for debug
        batch_id = torch.tensor(batch_ids)
        # print(f"batch_id in collate fn: {batch_id}")

        # ipdb.set_trace() # check n_id 0915
                
        # We collect the in-mini-batch size (`batch_size`), the offset of each
        # partition in the mini-batch (`offset`), and the number of nodes in
        # each partition (`count`)

        # batch_size = n_id.numel()
        batch_size = n_id_before.numel() # for debug
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset)
              
        rowptr_before, col_before, value_before = self.data.adj_t.csr()
        # ipdb.set_trace() # before relabel_fn 0915
        
        # # for debug
        self.old_n_id = n_id_before
              
        rowptr, col, value, n_id = relabel_fn(rowptr_before, col_before, value_before, n_id_before, self.bipartite) # one-hop

        # for debug check whether the order of n_id matters
        # n_id_sorted = torch.LongTensor(range(169343))
        # _rowptr, _col, _value, _n_id = relabel_fn(rowptr_before, col_before, value_before, n_id_sorted, self.bipartite)

        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                            #  sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                            sparse_sizes=(rowptr.numel()-1, n_id.numel()),
                             is_sorted=True)
        # ipdb.set_trace() # after relabel_fn 0915

        # ambershek NS之前的adj_t，备用
        # self.full_adj_t = adj_t # 这个是当前batch的full_batch，不是full adj_t
        
        # sampled_adj_t, sampled_n_id = sample_adj(src=adj_t, subset=old_n_id, num_neighbors=-1)
        # torch.set_printoptions(profile="full")        
        # f = open('/home/xshi22/gas/large_benchmark/out_'+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +'.txt', 'w')
        # f = open('/home/xshi22/gas/large_benchmark/out_'+ +'.txt', 'w')
        # orig_stdout = sys.stdout
        # sys.stdout = f  
        
        # num_neighbors由本文件内static变量给定
        # sampled_adj_t, sampled_n_id = sample_neighbors(src=adj_t, num_neighbors=static_num_neighbors, unsampled_n_id=n_id, target_nodes_id = self.old_n_id)
        # num_neighbors由SubgraphLoader member data给定
        # ipdb.set_trace()
        sampled_adj_t, sampled_n_id = sample_neighbors(src=adj_t, num_neighbors=self.num_neighbors, unsampled_n_id=n_id, target_nodes_id = self.old_n_id)
        # sampled_adj_t_pyg, sampled_n_id_pyg = sample_neighbors_pyg(src=adj_t, num_neighbors=self.num_neighbors, n_id=n_id, target_nodes_id = self.old_n_id)
        # ipdb.set_trace()
        # are_sparse_tensors_equal(sampled_adj_t, sampled_adj_t_pyg)
        # ipdb.set_trace()


        # torch.set_printoptions(profile="default") # reset
        # sys.stdout = orig_stdout
        # f.close()
        
        # sampled_n_id = n_id # for debug NS不起作用时，直接用relabel后的n_id
        # print(f'sampled_adj_t: {sampled_adj_t}') #debug
        # print(f'Same before/after NS?: {adj_t==sampled_adj_t}\n') #debug

        """
        #采样之前的data
        data_ori = self.data.__class__(adj_t=adj_t)
        for k, v in self.data:
            if isinstance(v, Tensor):
                if v.size(0) == self.data.num_nodes:
                    data_ori[k] = v.index_select(0, n_id)
        """
        
            
        # 采样之后的data
        data_sampled = self.data.__class__(adj_t=sampled_adj_t) # 改成sampled后的邻接矩阵
        for k, v in self.data:
            if isinstance(v, Tensor):
                if v.size(0) == self.data.num_nodes:
                    data_sampled[k] = v.index_select(0, sampled_n_id)
        
        # ambershek NS之后的索引值，备用
        self.sampled_n_id = sampled_n_id
        self.sampled_adj_t = sampled_adj_t
        # self.batch_size = batch_size
 
        # return SubData(data, batch_size, n_id, offset, count)
        # return SubData(data_ori, batch_size, n_id, offset, count) # for debug 返回采样前的
        
        return SubData(data_sampled, batch_size, sampled_n_id, offset, count) # n_id返回采样后的
        # ipdb.set_trace() # 检查data_sample和data_ori
        # return SubData_with_full_data(data_sampled, batch_size, sampled_n_id, offset, count, data_ori, batch_id) # n_id返回采样后的

        

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __iter__(self):
        # Clear previous shuffled batch IDs
        self.shuffled_batch_id = []

        # Get the original DataLoader iterator
        iterator = super().__iter__()

        # Iterate over the batches and collect the indices
        for batch_data in iterator:
            if hasattr(self, 'shuffle') and self.shuffle:
                # Get the indices that would be used for this batch and store them
                # Assuming dataset has ordered indices (like range or list of indices)
                # ipdb.set_trace()
                indices = batch_data[-1] if isinstance(batch_data, tuple) else batch_data
                self.shuffled_batch_id.append(indices)  # Store the shuffled indices
                # Print the shuffled batch indices (optional)
                # print(f"Shuffled batch indices: {self.shuffled_batch_id}")
            yield batch_data  # Return the batch as usual

        

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
        # ambershek
        # print('Initializing EvalSubgraphLoader...')

        ptr = ptr[::batch_size]
        if int(ptr[-1]) != data.num_nodes:
            ptr = torch.cat([ptr, torch.tensor([data.num_nodes])], dim=0)

        super().__init__(data=data, ptr=ptr, batch_size=1, bipartite=bipartite,
                         log=log, shuffle=False, num_workers=0, **kwargs)

