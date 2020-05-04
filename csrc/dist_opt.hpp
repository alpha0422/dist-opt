#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>

struct DistributedFusedAdamOptions {
  DistributedFusedAdamOptions() {}
  TORCH_ARG(double, learning_rate) = 1e-3;
  TORCH_ARG(bool, bias_correction) = true;
  TORCH_ARG(double, beta1) = 0.9;
  TORCH_ARG(double, beta2) = 0.999;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(bool, eps_inside_sqrt) = false;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, max_grad_norm) = 0;
  TORCH_ARG(bool, amsgrad) = false;
  TORCH_ARG(bool, use_mt) = false;
  TORCH_ARG(double, amp_scale_adjustment) = 1;
  TORCH_ARG(bool, overlap_reductions) = true;
  TORCH_ARG(bool, full_pipeline) = true;
  TORCH_ARG(bool, compute_L2_grad_norm) = false;
  TORCH_ARG(long, distributed_weight_update) = 0;
  TORCH_ARG(long, dwu_group_size) = 0;
  TORCH_ARG(long, dwu_num_blocks) = 4;
  TORCH_ARG(long, dwu_num_rs_pg) = 1;
  TORCH_ARG(long, dwu_num_ar_pg) = 4;
  TORCH_ARG(long, dwu_num_ag_pg) = 0;
  TORCH_ARG(long, dwu_num_chunks) = 4;
  TORCH_ARG(long, revert_method) = 1;
  TORCH_ARG(bool, flat_mt) = false;
  TORCH_ARG(bool, predivide) = true;
  TORCH_ARG(bool, e5m2_allgather) = false;
  TORCH_ARG(bool, do_not_flatten_model) = false;
};

/* Before initializing this, should call these API at Python side:
 *   assert torch.distributed.is_initialized()
 *   world_size = torch.distributed.get_world_size()
 *   rank = torch.distributed.get_rank()
 */
class DistributedFusedAdam : public torch::optim::Optimizer {
  public:
    DistributedFusedAdam(
          const std::vector<torch::Tensor> &_params,
          const int _world_size,
          const int _rank,
          /** DistributedFusedAdamOptions, leave them here to keep
           *  them shown in Python front-end help.
           */
          double _learning_rate,
          bool _bias_correction,
          double _beta1,
          double _beta2,
          double _eps,
          bool _eps_inside_sqrt,
          double _weight_decay,
          double _max_grad_norm,
          bool _amsgrad,
          bool _use_mt,
          double _amp_scale_adjustment,
          bool _overlap_reductions,
          bool _full_pipeline,
          bool _compute_L2_grad_norm,
          long _distributed_weight_update,
          long _dwu_group_size,
          long _dwu_num_blocks,
          long _dwu_num_rs_pg,
          long _dwu_num_ar_pg,
          long _dwu_num_ag_pg,
          long _dwu_num_chunks,
          long _revert_method,
          bool _flat_mt,
          bool _predivide,
          bool _e5m2_allgather,
          bool _do_not_flatten_model);
    ~DistributedFusedAdam() {}
    void step() override;

  private:
    std::vector<torch::Tensor> params;
    const int world_size;
    const int rank;
    DistributedFusedAdamOptions options;
};

