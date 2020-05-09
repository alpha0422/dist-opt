#include "dist_opt.hpp"
#include "../c10d/ProcessGroupNCCL.hpp"
#include "../c10d/TCPStore.hpp"
#include "../c10d/PrefixStore.hpp"

using namespace c10d;

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace pybind11::literals;

#ifndef DIST_OPT_HOOK_TENSOR
// Function hook executed after AccumulateGrad
class AccGradPostHook : public torch::autograd::FunctionPostHook {
  using variable_list = std::vector<torch::autograd::Variable>;

 public:
  /* implicit */ AccGradPostHook(DistributedFusedAdam* _dfa, long _p_i,
    long _p_grads_size, long _p_offset, at::Tensor& _param)
      : dfa(_dfa), p_i(_p_i), p_grads_size(_p_grads_size),
        p_offset(_p_offset), param(_param)  {}

  variable_list operator()(
      const variable_list& outputs,
      const variable_list& /* unused */) override {
    dfa->do_overlapped_reduction(p_i, p_grads_size, p_offset, param);
    return outputs;
  }

 protected:
  DistributedFusedAdam* dfa;
  const long p_i;
  const long p_grads_size;
  const long p_offset;
  at::Tensor& param;
};
#endif

void DistributedFusedAdamParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
}

void DistributedFusedAdamParamState::serialize(
    torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
}

DistributedFusedAdam::DistributedFusedAdam(
      const std::vector<torch::Tensor> &_params,
      /** DistributedFusedAdamOptions, leave them here to keep
       *  them shown in Python front-end help.
       */
      double _lr = 1e-3,
      std::tuple<double, double> _betas = std::make_tuple(0.9, 0.999),
      double _eps = 1e-8,
      double _weight_decay = 0,
      bool _amsgrad = false,
      bool _bias_correction = true,
      bool _eps_inside_sqrt = false,
      /** DistributedOptimizerOptions. */
      double _max_grad_norm = 0,
      bool _use_mt = false,
      double _amp_scale_adjustment = 1,
      bool _overlap_reductions = true,
      bool _full_pipeline = true,
      bool _compute_L2_grad_norm = false,
      long _distributed_weight_update = 0,
      long _dwu_group_size = 0,
      long _dwu_num_blocks = 4,
      long _dwu_num_rs_pg = 1,
      long _dwu_num_ar_pg = 4,
      long _dwu_num_ag_pg = 0,
      long _dwu_num_chunks = 4,
      long _revert_method = 1,
      bool _flat_mt = false,
      bool _predivide = true,
      bool _e5m2_allgather = false,
      bool _do_not_flatten_model = false)
    : torch::optim::Adam(_params,
        torch::optim::AdamOptions(_lr)
          .betas(_betas).weight_decay(_weight_decay)
          .eps(_eps).amsgrad(_amsgrad)),
      world_size(atoi(getenv("WORLD_SIZE"))),
      rank(atoi(getenv("RANK"))),
      master_addr(getenv("MASTER_ADDR")),
      master_port(atoi(getenv("MASTER_PORT"))),
      group_size(_dwu_group_size <= 0 ? torch::cuda::device_count() :
        _dwu_group_size),
      num_groups(atoi(getenv("WORLD_SIZE")) / (_dwu_group_size <= 0 ?
        torch::cuda::device_count() : _dwu_group_size)),
      rank_in_group(atoi(getenv("RANK")) / (_dwu_group_size <= 0 ?
        torch::cuda::device_count() : _dwu_group_size)) {

  options.bias_correction(_bias_correction)
         .eps_mode(_eps_inside_sqrt ? 0 : 1)
         .max_grad_norm(_max_grad_norm)
         .use_mt(_use_mt)
         .amp_scale_adjustment(_amp_scale_adjustment)
         .overlap_reductions(_overlap_reductions)
         .full_pipeline(_full_pipeline)
         .compute_L2_grad_norm(_compute_L2_grad_norm)
         .distributed_weight_update(_distributed_weight_update)
         .dwu_group_size(_dwu_group_size)
         .dwu_num_blocks(_dwu_num_blocks)
         .dwu_num_rs_pg(_dwu_num_rs_pg)
         .dwu_num_ar_pg(_dwu_num_ar_pg)
         .dwu_num_ag_pg(_dwu_num_ag_pg)
         .dwu_num_chunks(_dwu_num_chunks)
         .revert_method(_revert_method)
         .flat_mt(_flat_mt)
         .predivide(_predivide)
         .e5m2_allgather(_e5m2_allgather)
         .do_not_flatten_model(_do_not_flatten_model);

  // Parameter check
  assert(("DistributedFusedAdam does not support use_mt.", !options.use_mt()));
  assert(("DistributedFusedAdam does not support the AMSGrad variant.",
      !defaults.amsgrad()));
  assert(("More than one parameter group is not supported.",
      param_groups().size() == 1));

  if (options.revert_method() > 1) {
    std::cout << "revert_method -> double buffer fp32 parameters, "
        "will consume more memory" << std::endl;
  }

  // We don't have access to default process group here, build a new one
  // for parameter broadcast.
  auto store = std::make_shared<TCPStore>(master_addr, master_port, world_size);
  auto prefix_store = std::make_shared<PrefixStore>("dpg", store);
  ProcessGroupNCCL default_pg(prefix_store, rank, world_size);

  // Register backward hook
  long p_offset = 0, p_i = 0;
  for (auto& group : param_groups()) {
    at::Tensor *prev = nullptr;
    for (auto& p : group.params()) {
      size_t p_grads_size = p.numel();

      // Broadcast parameter of rank 0
      std::vector<at::Tensor> tensors = {p};
      std::shared_ptr<ProcessGroup::Work> work = default_pg.allreduce(tensors);
      work->wait();

      if (!p.requires_grad())  continue;

      model_params.push_back(p);
      auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
      if(param_state == state_.end()) {
        auto state = std::make_unique<DistributedFusedAdamParamState>();
        state->step(0);
        state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
      }

      std::printf("[%d:%d] hook %ld start at %ld size %ld\n", world_size, rank,
          p_i, p_offset, p_grads_size);

#ifdef DIST_OPT_HOOK_TENSOR
      // Hook on tensor
      auto hook = [&, p_i, p_grads_size, p_offset](at::Tensor grad) {
        do_overlapped_reduction(p_i, p_grads_size, p_offset, p);
      };
      p.register_hook(hook);
#else 
      // Hook on gradient accumulation function
      auto p_tmp = p.expand_as(p);
      assert(("Expect valid grad_fn.", !p_tmp.grad_fn()));
      auto grad_acc = p_tmp.grad_fn()->next_edge(0).function;
      auto allreduce_hook = AccGradPostHook(this, p_i, p_grads_size, p_offset, p);
      grad_acc->add_post_hook(torch::make_unique<AccGradPostHook>(allreduce_hook));
      grad_accs.push_back(std::move(grad_acc));
#endif

      p_offset += p_grads_size;
      if (prev && (reinterpret_cast<uint8_t*>(prev->data_ptr()) +
          prev->numel() * prev->element_size() ==
          reinterpret_cast<uint8_t*>(p.data_ptr()))) {
        p_offset = ((p_offset + 63) / 64) * 64;
      }
      prev = &p;
      p_i++;
    }
  }

  // Now start the TCP store daemon on the rank 0
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  at::Tensor tensor = at::empty({8, 8}, options);
  std::vector<at::Tensor> tensors;
  tensors.push_back(tensor);
  std::shared_ptr<ProcessGroup::Work> work = default_pg.allreduce(tensors);
  work->wait();

  std::printf("[%d:%d] DistributedFusedAdam init done.\n", world_size, rank);
}

void DistributedFusedAdam::do_overlapped_reduction(long param_i,
    long param_grads_size, long param_offset, at::Tensor &param) {
  auto& param_state = static_cast<DistributedFusedAdamParamState&>(
      *state_[c10::guts::to_string(
      model_params[0].unsafeGetTensorImpl())]);
  if (param_state.step() == 0)
    std::printf("[%d:%d] invoke hook %ld start at %ld size %ld\n", world_size, rank,
        param_i, param_offset, param_grads_size);
}

torch::Tensor DistributedFusedAdam::step(LossClosure closure = nullptr) {
  auto& group_options = static_cast<torch::optim::AdamOptions&>(param_groups()[0].options());
  std::cout << group_options.lr() << std::endl;
  return torch::empty({});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<DistributedFusedAdam>(m, "DistributedFusedAdam")
    .def(py::init<const std::vector<torch::Tensor> ,
      double, std::tuple<double, double>, double, double, bool, // amsgrad
      bool, bool, double, bool, double, bool, // overlap_reductions
      bool, bool, long, long, long, long, long, long, long,
      long, bool, bool, bool, bool>(),
      "params"_a,
      "lr"_a=1e-3,
      "betas"_a = std::make_tuple(0.9, 0.999),
      "eps"_a=1e-8,
      "weight_decay"_a=0,
      "amsgrad"_a=false,
      "bias_correction"_a=true,
      "eps_inside_sqrt"_a=false,
      "max_grad_norm"_a=0,
      "use_mt"_a=false,
      "amp_scale_adjustment"_a=1,
      "overlap_reductions"_a=true,
      "full_pipeline"_a=true,
      "compute_L2_grad_norm"_a=false,
      "distributed_weight_update"_a=0,
      "dwu_group_size"_a=0,
      "dwu_num_blocks"_a=4,
      "dwu_num_rs_pg"_a=1,
      "dwu_num_ar_pg"_a=4,
      "dwu_num_ag_pg"_a=0,
      "dwu_num_chunks"_a=4,
      "revert_method"_a=1,
      "flat_mt"_a=false,
      "predivide"_a=true,
      "e5m2_allgather"_a=false,
      "do_not_flatten_model"_a=false)
    .def("step", &DistributedFusedAdam::step, "closure"_a=nullptr);
}
