#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>, torch::Tensor>
relabel_one_hop_cpu(torch::Tensor rowptr, torch::Tensor col,
                    torch::optional<torch::Tensor> optional_value,
                    torch::Tensor idx, bool bipartite);


std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>, torch::Tensor>
relabel_one_hop_within_batch_cpu(torch::Tensor rowptr, torch::Tensor col, torch::optional<torch::Tensor> optional_value, torch::Tensor idx, bool bipartite) ;
