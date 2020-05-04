#include "dist_opt.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

DistributedFusedAdam::DistributedFusedAdam(
      const std::vector<torch::Tensor> &_params,
      const int _world_size,
      const int _rank,
      /** DistributedFusedAdamOptions, leave them here to keep
       *  them shown in Python front-end help.
       */
      double _learning_rate = 1e-3,
      bool _bias_correction = true,
      double _beta1 = 0.9,
      double _beta2 = 0.999,
      double _eps = 1e-8,
      bool _eps_inside_sqrt = false,
      double _weight_decay = 0,
      double _max_grad_norm = 0,
      bool _amsgrad = false,
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
    : params(_params), world_size(_world_size), rank(_rank)  {
  options.learning_rate(_learning_rate)
         .bias_correction(_bias_correction)
         .beta1(_beta1)
         .beta2(_beta2)
         .eps(_eps)
         .eps_inside_sqrt(_eps_inside_sqrt)
         .weight_decay(_weight_decay)
         .max_grad_norm(_max_grad_norm)
         .amsgrad(_amsgrad)
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
}

void DistributedFusedAdam::step() {
  std::cout << options.learning_rate() << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<DistributedFusedAdam>(m, "DistributedFusedAdam")
    .def(py::init<const std::vector<torch::Tensor> ,
      const int, const int, double, bool, double, double,
      double, bool, double, double, bool, bool, double, bool,
      bool, bool, long, long, long, long, long, long, long,
      long, bool, bool, bool, bool>(),
      "params"_a,
      "world_size"_a,
      "rank"_a,
      "learning_rate"_a=1e-3,
      "bias_correction"_a=true,
      "beta1"_a=0.9,
      "beta2"_a=0.999,
      "eps"_a=1e-8,
      "eps_inside_sqrt"_a=false,
      "weight_decay"_a=0,
      "max_grad_norm"_a=0,
      "amsgrad"_a=false,
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
    .def("step", &DistributedFusedAdam::step);
}
