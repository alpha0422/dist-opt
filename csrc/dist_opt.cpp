#include "dist_opt.hpp"

using torch::indexing::Slice;

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace pybind11::literals;

#define PG_NAME_LEN 128
#define NCCL_TIME_OUT (120 * 1000)

#define LOGGING(fmt, ...) do { \
  if (options.logging()) { \
    std::printf("[%d:%d] " fmt, options.world_size(), options.rank(), \
      ##__VA_ARGS__); \
    std::cout.flush(); \
  } \
} while(0)

#define LOGGING_VEC(prefix, fmt, vec) do { \
  if (options.logging()) { \
    std::printf("[%d:%d] " prefix, options.world_size(), options.rank()); \
    for (auto i : vec) { \
      std::printf(" " fmt, i); \
    } \
    std::printf("\n"); \
    std::cout.flush(); \
  } \
} while(0)

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
      current_block(_overlap_reductions ? _dwu_num_blocks : -1),
      l2_grad_norm_st(at::cuda::getStreamFromPool()),
      completion_st(at::cuda::getStreamFromPool()) {

  options.bias_correction(_bias_correction)
         .eps_mode(_eps_inside_sqrt ? 0 : 1)
         .max_grad_norm(_max_grad_norm)
         .use_mt(_use_mt)
         .amp_scale_adjustment(_amp_scale_adjustment)
         .overlap_reductions(_overlap_reductions)
         .full_pipeline(_full_pipeline)
         .compute_L2_grad_norm(_compute_L2_grad_norm)
         .distributed_weight_update(_distributed_weight_update)
         .num_blocks(_dwu_num_blocks)
         .num_rs_pg(_dwu_num_rs_pg)
         .num_ar_pg(_dwu_num_ar_pg)
         .num_ag_pg(_dwu_num_ag_pg)
         .num_chunks(_dwu_num_chunks)
         .revert_method(_revert_method)
         .flat_mt(_flat_mt)
         .predivide(_predivide)
         .e5m2_allgather(_e5m2_allgather)
         .do_not_flatten_model(_do_not_flatten_model);

  options.logging(getenv("DIST_OPT_LOG") ? true : false)
         .world_size(atoi(getenv("WORLD_SIZE")))
         .rank(atoi(getenv("RANK")))
         .master_addr(getenv("MASTER_ADDR"))
         .master_port(atoi(getenv("MASTER_PORT")));

  options.group_size(_dwu_group_size <= 0 ? torch::cuda::device_count() :
            _dwu_group_size)
         .num_groups(options.world_size() / options.group_size())
         .group_rank(options.rank() / options.group_size())
         .rank_in_group(options.rank() % options.group_size());

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

  LOGGING("master: %s:%d\n", options.master_addr().c_str(), options.master_port());
  LOGGING("group_size %ld num_groups %ld rank_in_group %ld\n",
    options.group_size(), options.num_groups(), options.rank_in_group());

  // We don't have access to default process group here, build a new one
  // for parameter broadcast.
  auto store = std::make_shared<TCPStore>(options.master_addr(),
    options.master_port(), options.world_size());
  auto dpg_prefix_store = std::make_shared<PrefixStore>("dpg", store);
  auto timeout = std::chrono::milliseconds(NCCL_TIME_OUT);
  ProcessGroupNCCL default_pg(dpg_prefix_store, options.rank(),
    options.world_size(), timeout);

  // Register backward hook
  long p_offset = 0, p_i = 0;
  for (auto& group : param_groups()) {
    at::Tensor *prev = nullptr;
    for (auto& p : group.params()) {
      // Broadcast parameter of rank 0
      std::vector<at::Tensor> tensors = {p};
      std::shared_ptr<ProcessGroup::Work> work = default_pg.allreduce(tensors);
      work->wait();

      if (!p.requires_grad())  continue;

      size_t p_grads_size = p.numel();
      model_params.push_back(p);
      auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
      if(param_state == state_.end()) {
        auto state = std::make_unique<DistributedFusedAdamParamState>();
        state->step(0);
        state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
      }

      LOGGING("hook param %ld start at %ld size %ld\n", p_i, p_offset, p_grads_size);

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

      grads_info.push_back(std::make_pair(p_grads_size, p_offset));
      p_offset += p_grads_size;

      /** Only enforce 128b alignment (64 * fp16) for non-consecutive parameters
       *  RNN is one example of consecutive parameters:
       *  (weight_ih, weight_hh, bias_ih, bias_hh)
       */
      if (prev && (reinterpret_cast<uint8_t*>(prev->data_ptr()) +
          prev->numel() * prev->element_size() !=
          reinterpret_cast<uint8_t*>(p.data_ptr()))) {
        p_offset = ((p_offset + 63) / 64) * 64;
      }
      prev = &p;
      p_i++;
    }
  }
  grads_generated.resize(p_i, false);

  // Arguments for distributed optimizer
  options.net_total_param_size(p_offset);
  long total_param_size = p_offset;
  const long dwu_min_page_size = 256 * options.num_blocks() *
    options.num_chunks() * options.group_size();
  total_param_size = (total_param_size + dwu_min_page_size - 1) /
    dwu_min_page_size * dwu_min_page_size;
  options.block_size(total_param_size / options.num_blocks());
  options.chunk_size(options.block_size() / options.num_chunks());
  options.shard_size(options.chunk_size() / options.group_size());

  LOGGING("Sizes: net_total_param=%ld, total_param=%ld, min_page=%ld, "
          "block=%ld, chunk=%ld, shard=%ld\n", options.net_total_param_size(),
          total_param_size, dwu_min_page_size, options.block_size(),
          options.chunk_size(), options.shard_size());

  // FIXME: you'll get "low_param_i: 0 0 1 1" for a single LSTM layer
  low_param_i.resize(options.num_blocks(), 0);
  for (int block_id = options.num_blocks()-1; block_id >= 0; block_id--) {
    int p_idx = p_i - 1;
    while (p_idx > 0 && grads_info[p_idx].second > block_id *
        options.block_size()) {
      p_idx--;
    }
    low_param_i[block_id] = p_idx;
  }
  LOGGING_VEC("low_param_i:", "%ld", low_param_i);

  // Flattened parameters, gradients, states
  const long mega_shard_size = options.num_blocks() * options.num_chunks() *
    options.shard_size();
  auto base_options = at::TensorOptions().device(at::kCUDA);
  auto byte_options = base_options.dtype(at::kByte);
  auto half_options = base_options.dtype(at::kHalf);
  auto float_options = base_options.dtype(at::kFloat);
  flat_grads = at::zeros({total_param_size}, half_options);
  new_params = at::zeros({total_param_size}, options.e5m2_allgather() ?
    byte_options : half_options);
  fp32_p = at::zeros({mega_shard_size}, float_options);
  fp32_m = at::zeros({mega_shard_size}, float_options);
  fp32_v = at::zeros({mega_shard_size}, float_options);
  // FIXME: Rethink fp16 label since it's either uint8 or fp16
  fp16_p = at::zeros({mega_shard_size}, options.e5m2_allgather() ?
    byte_options : half_options);
  fp16_g = at::zeros({mega_shard_size}, half_options);

  for (int i=0; i<p_i; i++) {
    long start = grads_info[i].second;
    long stop = start + grads_info[i].first;
    auto fgrads = flat_grads.index({Slice(start, stop)});
    individual_flat_grads.push_back(fgrads.view_as(model_params[i]));
  }

  // Gradients organized as: blocks x chunks x groups
  for (int idx_b=0; idx_b<options.num_blocks(); idx_b++) { // blocks
    long start = idx_b * options.block_size();
    long stop = start + options.block_size();
    auto _block = flat_grads.index({Slice(start, stop)});
    flat_grads_blocks.push_back(_block);

    std::vector<at::Tensor> _chunks;
    std::vector<std::vector<at::Tensor> > _shards;
    for (int idx_c=0; idx_c<options.num_chunks(); idx_c++) { // chunks
      long start = idx_c * options.chunk_size();
      long stop = start + options.chunk_size();
      auto _chunk = _block.index({Slice(start, stop)});
      _chunks.push_back(_chunk);

      std::vector<at::Tensor> __shards;
      for (int idx_s=0; idx_s<options.group_size(); idx_s++) { // shards
        long start = idx_s * options.shard_size();
        long stop = start + options.shard_size();
        auto _shard = _chunk.index({Slice(start, stop)});
        __shards.push_back(_shard);
      }
      _shards.push_back(__shards);
    }
    flat_grads_chunks.push_back(_chunks);
    flat_grads_shards.push_back(_shards);
  }

  // Parameters organized as: groups x blocks x chunks
  for (int idx_s=0; idx_s<options.group_size(); idx_s++) { // shards
    long start = idx_s * mega_shard_size;
    long stop = start + mega_shard_size;
    auto _shard = new_params.index({Slice(start, stop)});
    new_params_mega_shards.push_back(_shard);

    std::vector<at::Tensor> _blocks;
    std::vector<std::vector<at::Tensor> > _chunks;
    for (int idx_b=0; idx_b<options.num_blocks(); idx_b++) { // blocks
      long start = idx_b * options.num_chunks() * options.shard_size();
      long stop = start + options.num_chunks() *options.shard_size();
      auto _block = _shard.index({Slice(start, stop)});
      _blocks.push_back(_block);

      std::vector<at::Tensor> __chunks;
      for (int idx_c=0; idx_c<options.num_chunks(); idx_c++) { // chunks
        long start = idx_c * options.shard_size();
        long stop = start + options.shard_size();
        auto _chunk = _block.index({Slice(start, stop)});
        __chunks.push_back(_chunk);
      }
      _chunks.push_back(__chunks);
    }
    new_params_mega_blocks.push_back(_blocks);
    new_params_mega_chunks.push_back(_chunks);
  }

  // Packed states organized as: blocks x chunks
  auto _packed_split = [this](std::vector<at::Tensor>& p_blocks,
      std::vector<std::vector<at::Tensor> >& p_chunks, at::Tensor &p) {
    for (int idx_b=0; idx_b<options.num_blocks(); idx_b++) { // blocks
      long start = idx_b * options.num_chunks() * options.shard_size();
      long stop = start + options.num_chunks() * options.shard_size();
      auto _p_block = p.index({Slice(start, stop)});
      p_blocks.push_back(_p_block);

      /** In the packed format, each chunk contains one shard, so
       *  packed_chunk_size == shard_size.
       */
      std::vector<at::Tensor> _p_chunks;
      for (int idx_c=0; idx_c<options.num_chunks(); idx_c++) { // chunks
        long start = idx_c * options.shard_size();
        long stop = start + options.shard_size();
        auto _p_chunk = _p_block.index({Slice(start, stop)});
        _p_chunks.push_back(_p_chunk);
      }
      p_chunks.push_back(_p_chunks);
    }
  };
  _packed_split(fp32_p_blocks, fp32_p_chunks, fp32_p);
  _packed_split(fp32_m_blocks, fp32_m_chunks, fp32_m);
  _packed_split(fp32_v_blocks, fp32_v_chunks, fp32_v);
  _packed_split(fp16_p_blocks, fp16_p_chunks, fp16_p);
  _packed_split(fp16_g_blocks, fp16_g_chunks, fp16_g);

  /** This paragraph does two things:
   *  1) Copy model parameters into master buffer
   *  2) Create tensor lists for unpacking new parameter tensor after all-gather
   */
  std::vector<at::Tensor> _p_in, _p_out;
  for (int shard_id=0; shard_id<options.group_size(); shard_id++) {
    for (int block_id=0; block_id<options.num_blocks(); block_id++) {
      for (int chunk_id=0; chunk_id<options.num_chunks(); chunk_id++) {
        long flat_shard_start = ((block_id * options.num_blocks() + chunk_id)
          * options.group_size() + shard_id) * options.shard_size();
        long flat_shard_end = flat_shard_start + options.shard_size();
        for (int p_idx=0; p_idx<p_i; p_idx++) {
          long flat_grad_start = grads_info[p_idx].second;
          long flat_grad_end = flat_grad_start + grads_info[p_idx].first;
          long clipped_start = std::max(flat_grad_start, flat_shard_start);
          long clipped_end = std::min(flat_grad_end, flat_shard_end);

          if (clipped_start >= clipped_end)  continue;

          long grad_offset = clipped_start - flat_grad_start;
          long grad_length = clipped_end - clipped_start;
          long shard_offset = clipped_start - flat_shard_start;
          auto model_param_fragment = model_params[p_idx].view(-1).index(
            {Slice(grad_offset, grad_offset+grad_length)});
          auto new_param_packed_fragment = new_params_mega_chunks \
            [shard_id][block_id][chunk_id].index({Slice(
            shard_offset, shard_offset+grad_length)});
          _p_in.push_back(new_param_packed_fragment);
          _p_out.push_back(model_param_fragment);

          // Copy model parameters into master buffer
          if (shard_id == options.rank_in_group()) {
            auto master_param_fragment = fp32_p_chunks[block_id][chunk_id] \
              .index({Slice(shard_offset, shard_offset+grad_length)});
            master_param_fragment.copy_(model_param_fragment);

            LOGGING("Sizes: model_param_fragment=%ld, new_param_packed_fragment=%ld"
              ", master_param_fragment=%ld\n", model_param_fragment.numel(),
              new_param_packed_fragment.numel(), master_param_fragment.numel());
          }
        }
      }
    }
  }
  packed_flat_to_model_params.push_back(_p_in);
  packed_flat_to_model_params.push_back(_p_out);
   
  /** Build process groups and CUDA streams.
   *  Ranks not in the process group are not necessary to participate
   *  communicator construction in c10d C++ API.
   */
  char _pg_name[PG_NAME_LEN];
  std::vector<at::Tensor> _tensors = {at::zeros({1}, float_options)};
  if (options.num_groups() > 1) {
    for (int i=0; i<options.num_ar_pg(); i++) {
      std::snprintf(_pg_name, PG_NAME_LEN, "ar_pg_%ld_%d",
        options.rank_in_group(), i);
      auto prefix_store = std::make_shared<PrefixStore>(_pg_name, store);
      auto _ar_pg = std::make_shared<ProcessGroupNCCL>(prefix_store,
        options.group_rank(), options.num_groups(), timeout);

      LOGGING("Building process group %s: world_size %ld, rank %ld\n",
        _pg_name, options.num_groups(), options.group_rank());

      /** For faster initialization.
       *  NCCL communicators are lazy initialized in c10d, and c10d keeps
       *  a reference to the store.
       */
      std::shared_ptr<ProcessGroup::Work> work = _ar_pg->allreduce(_tensors);
      work->wait();
    
      ar_pg.emplace_back(_ar_pg);
      ar_st.emplace_back(at::cuda::getStreamFromPool());
    }
    cudaDeviceSynchronize();
  }

  // Re-use reduce-scatter process group for all-gather if num_ag_pg is 0
  for (int i=0; i<options.num_rs_pg(); i++) {
    std::snprintf(_pg_name, PG_NAME_LEN, "rs_pg_%ld_%d",
      options.group_rank(), i);
    auto prefix_store = std::make_shared<PrefixStore>(_pg_name, store);
    auto _rs_pg = std::make_shared<ProcessGroupNCCL>(prefix_store,
      options.rank_in_group(), options.group_size(), timeout);

    LOGGING("Building process group %s: world_size %ld, rank %ld\n",
      _pg_name, options.group_size(), options.rank_in_group());

    std::shared_ptr<ProcessGroup::Work> work = _rs_pg->allreduce(_tensors);
    work->wait();

    rs_pg.push_back(_rs_pg);
    rs_st.emplace_back(at::cuda::getStreamFromPool());
    if (options.num_ag_pg() == 0) {
      ag_pg.push_back(_rs_pg);
      ag_st.emplace_back(at::cuda::getStreamFromPool());
    }
  }
  cudaDeviceSynchronize();
  if (options.num_ag_pg() != 0) {
    for (int i=0; i<options.num_ag_pg(); i++) {
      std::snprintf(_pg_name, PG_NAME_LEN, "ag_pg_%ld_%d",
        options.group_rank(), i);
      auto prefix_store = std::make_shared<PrefixStore>(_pg_name, store);
      auto _ag_pg = std::make_shared<ProcessGroupNCCL>(prefix_store,
        options.rank_in_group(), options.group_size(), timeout);

      LOGGING("Building process group %s: world_size %ld, rank %ld\n",
        _pg_name, options.group_size(), options.rank_in_group());

      std::shared_ptr<ProcessGroup::Work> work = _ag_pg->allreduce(_tensors);
      work->wait();

      ag_pg.emplace_back(_ag_pg);
      ag_st.emplace_back(at::cuda::getStreamFromPool());
    }
    cudaDeviceSynchronize();
  }

  // Process group for L2 gradient norm
  std::snprintf(_pg_name, PG_NAME_LEN, "l2_grad_norm_pg_%ld",
    options.group_rank());
  auto l2_grad_norm_prefix_store = std::make_shared<PrefixStore>(_pg_name,
    store);
  l2_grad_norm_pg = std::make_unique<ProcessGroupNCCL>(
    l2_grad_norm_prefix_store, options.rank_in_group(), options.group_size(),
    timeout);
  LOGGING("Building process group %s: world_size %ld, rank %ld\n",
    _pg_name, options.group_size(), options.rank_in_group());
  std::shared_ptr<ProcessGroup::Work> _l2_grad_norm_work =
    l2_grad_norm_pg->allreduce(_tensors);
  _l2_grad_norm_work->wait();
  cudaDeviceSynchronize();

  LOGGING("DistributedFusedAdam init done.\n");
}

void DistributedFusedAdam::do_overlapped_reduction(long param_i,
    long param_grads_size, long param_offset, at::Tensor &param) {
  auto& param_state = static_cast<DistributedFusedAdamParamState&>(
      *state_[c10::guts::to_string(
      model_params[0].unsafeGetTensorImpl())]);
  if (param_state.step() == 0)
    LOGGING("invoke hook param %ld start at %ld size %ld\n", param_i,
        param_offset, param_grads_size);
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
