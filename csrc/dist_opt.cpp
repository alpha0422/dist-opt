#include <torch/extension.h>
#include <ATen/ATen.h>

class DistributedFusedAdam : public torch::optim::Optimizer {
  public:
    DistributedFusedAdam(const std::vector<torch::Tensor> &params)
        : params(params) {}
    ~DistributedFusedAdam() {}
    void step() {}
  private:
    std::vector<torch::Tensor> params;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<DistributedFusedAdam>(m, "DistributedFusedAdam")
      .def(py::init<const std::vector<torch::Tensor> >())
      .def("step", &DistributedFusedAdam::step);
}
