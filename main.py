import time
import hydra
from omegaconf import OmegaConf

import torch
# torch.set_default_dtype(torch.float64)

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric_autoscale import (get_data, metis, permute,
                                       SubgraphLoader, EvalSubgraphLoader,
                                       models, compute_micro_f1, dropout)
from torch_geometric_autoscale.data import get_ppi


from torch_sparse import SparseTensor, sample, sample_adj

import ipdb

# ambershhek for logging
import logging
# A logger for this file
log = logging.getLogger(__name__)

import torch.profiler
import os

import random
# torch.manual_seed(123)
# torch.manual_seed(43)
# torch.manual_seed(42)

import sys
import numpy as np
import matplotlib.pyplot as plt


relabel_fn = torch.ops.torch_geometric_autoscale.relabel_one_hop

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

from torchviz import make_dot


# train one epoch
def mini_train(model, loader, criterion, optimizer, max_steps, grad_norm=None, edge_dropout=0.0, epoch=0, VR_update=False, period_updates_in_one_epoch=0, ptr=None, debug_flag = False, period_update_trigger=None, drift_norm = 2, aggregate_combined = True, use_aggregation=True):
    


    model.train()

    # Determine VR update frequency
    if period_updates_in_one_epoch != 0:
        period_update_frequency_iter = int(len(loader) / period_updates_in_one_epoch)
   

    for i, (batch, batch_size, n_id, offset, count) in enumerate(loader):     


        x = batch.x.to(model.device)
        adj_t = batch.adj_t.to(model.device)
        y = batch.y[:batch_size].to(model.device)
        train_mask = batch.train_mask[:batch_size].to(model.device)
        if train_mask.sum() == 0:
            continue

        adj_t = dropout(adj_t, p=edge_dropout) 

        if VR_update:       
            returned_dict = model.VR_call(x, adj_t, batch_size, n_id, offset, count, debug_flag=debug_flag, drift_norm=drift_norm, epoch=epoch, batch_idx=i)
            torch.cuda.synchronize()
        else:

            returned_dict = model(x, adj_t, batch_size, n_id, offset, count, drift_norm=drift_norm, aggregate_combined=aggregate_combined, use_aggregation=use_aggregation)
            torch.cuda.synchronize()

        out = returned_dict['out']
        optimizer.zero_grad()
        loss = criterion(out[train_mask], y[:batch_size][train_mask])
        loss.backward()

        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        total_loss += float(loss) * int(train_mask.sum())
        total_examples += int(train_mask.sum())

        # We may abort after a fixed number of steps to refresh histories...
        if (i + 1) >= max_steps and (i + 1) < len(loader):
            break

    return {
        "loss": total_loss / total_examples,
    }


@torch.no_grad()
def full_test(model, data):
    model.eval()
    return model(data.x.to(model.device), data.adj_t.to(model.device)).cpu()


@torch.no_grad()
def mini_test(model, loader, use_aggregation=True):
    # print('function mini_test called in main')
    model.eval()
    return model(loader=loader, use_aggregation=use_aggregation)


@hydra.main(config_path='conf', config_name='config')
def main(conf):
    # Set random seed
    seed = conf.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    conf.model.params = conf.model.params[conf.dataset.name]
    params = conf.model.params
    log.info(OmegaConf.to_yaml(conf)) 

    # Use the dynamically overridden dropout value from the configuration
    params.architecture.dropout = conf.dropout  # Update the dropout value in params.architecture
    log.info(f'Using dropout value: {params.architecture.dropout}')

    try:
        edge_dropout = params.edge_dropout
    except:  # noqa
        edge_dropout = 0.0
    grad_norm = None if isinstance(params.grad_norm, str) else params.grad_norm

    device = f'cuda:{conf.device}' if torch.cuda.is_available() else 'cpu'

    # load data
    t = time.perf_counter()
    data, in_channels, out_channels = get_data(conf.root, conf.dataset.name)
    log.info(f'Done! [{time.perf_counter() - t:.2f}s]')
    
    ### metis partitioning
    perm, ptr = metis(data.adj_t, num_parts=params.num_parts, log=True)
    data = permute(data, perm, log=True)

    if conf.model.loop:
        data.adj_t = data.adj_t.set_diag()
    
    if conf.model.norm:
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)

    if data.y.dim() == 1:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = SubgraphLoader(data, ptr, batch_size=conf.batch_size,
                                  shuffle=True, num_workers=params.num_workers,
                                  persistent_workers=params.num_workers > 0, num_neighbors=conf.num_neighbors, type='train', IB=conf.VR_update)
    

    eval_loader = EvalSubgraphLoader(data, ptr,
                                    batch_size=conf.batch_size)


    if conf.dataset.name == 'ppi':
        val_data, _, _ = get_ppi(conf.root, split='val')
        test_data, _, _ = get_ppi(conf.root, split='test')
        if conf.model.loop:
            val_data.adj_t = val_data.adj_t.set_diag()
            test_data.adj_t = test_data.adj_t.set_diag()
        if conf.model.norm:
            val_data.adj_t = gcn_norm(val_data.adj_t, add_self_loops=False)
            test_data.adj_t = gcn_norm(test_data.adj_t, add_self_loops=False)

    buffer_size = max([n_id.numel() for _, _, n_id, _, _ in eval_loader]) * 2
    log.info(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size}')

    kwargs = {}
    if conf.model.name[:3] == 'PNA':
        kwargs['deg'] = data.adj_t.storage.rowcount()

    # build model
    GNN = getattr(models, conf.model.name)
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        pool_size=params.pool_size,
        buffer_size=buffer_size,
        **params.architecture,
        **kwargs,
    ).to(device)
    
    optimizer = torch.optim.Adam([
        dict(params=model.reg_modules.parameters(),
             weight_decay=params.reg_weight_decay),
        dict(params=model.nonreg_modules.parameters(),
             weight_decay=params.nonreg_weight_decay)
    ], lr=params.lr)

    # Handling aggregate_combined and use_aggregation keys, only for those without VR
    aggregate_combined = getattr(conf, 'aggregate_combined', True)  # Default to True if not set
    use_aggregation = getattr(conf, 'use_aggregation', True)  # Default to True if not set

    t = time.perf_counter()
    print('Fill history...', end=' ', flush=True)
    
    ### use modified mini_inference_vr for VR updating
    model.eval()
    if conf.VR_update:
        model.mini_inference_vr(loader=eval_loader, use_aggregation=use_aggregation)
    else:
        model.mini_inference(loader=eval_loader, use_aggregation=use_aggregation)

    log.info(f'Done! [{time.perf_counter() - t:.2f}s]')

    period_update_trigger = None

    best_val_acc = test_acc = 0

    max_steps = params.max_steps if params.max_steps != -1 else int(params.num_parts / conf.batch_size)
    log.info(f'max_steps = {max_steps}')

    for epoch in range(params.epochs):
        results = mini_train(model, train_loader, criterion, optimizer, max_steps, grad_norm, edge_dropout, epoch, VR_update=conf.VR_update, period_updates_in_one_epoch=conf.period_updates_in_one_epoch, ptr=ptr, debug_flag=False, period_update_trigger=period_update_trigger, drift_norm=conf.drift_norm, aggregate_combined=aggregate_combined, use_aggregation=use_aggregation)

        loss = results['loss']

        model.eval()
        
        if conf.VR_update:
            out = model.mini_inference_vr(loader=eval_loader, use_aggregation=use_aggregation)
        else:
            out = model.mini_inference(loader=eval_loader, use_aggregation=use_aggregation)
        
        # calculate evaluation metrics
        train_acc = compute_micro_f1(out, data.y, data.train_mask)

        if conf.dataset.name != 'ppi':
            val_acc = compute_micro_f1(out, data.y, data.val_mask)
            tmp_test_acc = compute_micro_f1(out, data.y, data.test_mask)
        else:
            # We need to perform inference on a different graph as PPI is an
            # inductive dataset.
            val_acc = compute_micro_f1(full_test(model, val_data), val_data.y)
            tmp_test_acc = compute_micro_f1(full_test(model, test_data),
                                            test_data.y)

        # update best results
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            
        # output logs
        if epoch % conf.log_every == 0:
        # if (epoch+1) % conf.log_every == 0:
            log.info(f'Epoch: {epoch:04d}, Loss: {loss:.4f}, '
                f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                f'Test: {tmp_test_acc:.4f}, Final: {test_acc:.4f}')

    log.info('=========================')
    log.info(f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')



if __name__ == "__main__":
    main()

