#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <limits>

// Hook on AccumulateGrad by default
#undef DIST_OPT_HOOK_TENSOR

struct DistributedOptimizerOptions {
  TORCH_ARG(bool, bias_correction) = true;
  TORCH_ARG(int, eps_mode) = 1; // eps_inside_sqrt = False
  TORCH_ARG(double, max_grad_norm) = 0;
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

/** We only need step for now. */
struct DistributedFusedAdamParamState : public 
    torch::optim::OptimizerCloneableParamState<DistributedFusedAdamParamState> {
  TORCH_ARG(int64_t, step) = 0;

public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  ~DistributedFusedAdamParamState() = default;
};

class AccGradPostHook;

/* Before initializing this, should call these API at Python side, and make sure
 * environment variables WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
 * are set correctly:
 *
 *   torch.distributed.init_process_group(backend='nccl', init_method="env://")
 *   assert torch.distributed.is_initialized()
 *   world_size = torch.distributed.get_world_size()
 *   rank = torch.distributed.get_rank()
 *   torch.cuda.set_device(rank)
 */
class DistributedFusedAdam : public torch::optim::Adam {
  public:
    DistributedFusedAdam(
          /** Since we only assume single param group for now,
           *  let's not use OptimizerParamGroup.
           */
          const std::vector<torch::Tensor> &_params,
          /** DistributedFusedAdamOptions, leave them here to keep
           *  them shown in Python front-end help.
           */
          double _lr,
          std::tuple<double, double> _betas,
          double _eps,
          double _weight_decay,
          bool _amsgrad,
          bool _bias_correction,
          bool _eps_inside_sqrt,
          double _max_grad_norm,
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
    torch::Tensor step(LossClosure closure) override;

  protected:
#ifndef DIST_OPT_HOOK_TENSOR
    friend class AccGradPostHook;
#endif
    void do_overlapped_reduction(long param_i, long param_grads_size,
        long param_offset, at::Tensor &param);

  private:
    DistributedOptimizerOptions options;

    // For NCCL initialization
    const int world_size;
    const int rank;
    const std::string master_addr;
    const int master_port;
    const int group_size;
    const int num_groups;
    const int rank_in_group;

    // Distributed optimizer specifics
    bool _last_step = false;
    // Must set global scale first
    double _global_scale = std::numeric_limits<double>::quiet_NaN();
    bool _has_overflow = false;

    at::Tensor _overflow_buf = at::zeros({1}, at::TensorOptions().dtype(at::kLong)
        .device(at::kCUDA));
    at::Tensor _L2_grad_norm = at::zeros({1}, at::TensorOptions().dtype(at::kFloat)
        .device(at::kCUDA));

    // Pair of (param_grads_size, param_offset)
    std::vector<std::pair<int64_t, int64_t> > grads_info;
    std::vector<at::Tensor> model_params;
#ifndef DIST_OPT_HOOK_TENSOR
    std::vector<std::shared_ptr<torch::autograd::Node> > grad_accs;
#endif
};

