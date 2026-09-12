#include "runtime.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace vt {

// Pick the GPU device. VT_BACKEND=vulkan|hip selects by name (HIP devices are
// named "ROCm*"); otherwise the first GPU is used.
static ggml_backend_dev_t pick_gpu() {
    const char* want_env = std::getenv("VT_BACKEND");
    std::string want = want_env ? want_env : "";
    std::transform(want.begin(), want.end(), want.begin(), ::tolower);

    const size_t n = ggml_backend_dev_count();
    ggml_backend_dev_t first = nullptr;
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        if (first == nullptr) first = d;
        if (want.empty()) continue;

        std::string name = ggml_backend_dev_name(d);
        std::string desc = ggml_backend_dev_description(d);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        std::transform(desc.begin(), desc.end(), desc.begin(), ::tolower);
        const bool hit = name.find(want) != std::string::npos ||
                         desc.find(want) != std::string::npos ||
                         (want == "hip" && name.find("rocm") != std::string::npos);
        if (hit) return d;
    }
    return first;
}

Runtime::Runtime() {
    ggml_backend_load_all();

    ggml_backend_dev_t gpu = pick_gpu();
    if (gpu == nullptr) {
        throw std::runtime_error("Runtime: no GPU backend available (is ggml-vulkan built?)");
    }

    backend_ = ggml_backend_dev_init(gpu, nullptr);
    if (backend_ == nullptr) {
        throw std::runtime_error("Runtime: ggml_backend_dev_init failed");
    }
    name_ = std::string(ggml_backend_dev_name(gpu)) + ": " +
            ggml_backend_dev_description(gpu);

    // CPU backend as a fallback for ops the GPU backend does not implement.
    ggml_backend_t backends[2];
    int n = 0;
    backends[n++] = backend_;
    const bool gpu_only = std::getenv("MINITORCH_GPU_ONLY") != nullptr;
    ggml_backend_dev_t cpu = gpu_only ? nullptr : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        backend_cpu_ = ggml_backend_dev_init(cpu, nullptr);
        if (backend_cpu_ != nullptr) backends[n++] = backend_cpu_;
    }

    sched_ = ggml_backend_sched_new(backends, nullptr, n, /*graph_size=*/16384,
                                    /*parallel=*/false, /*op_offload=*/true);
    if (sched_ == nullptr) {
        throw std::runtime_error("Runtime: ggml_backend_sched_new failed");
    }
}

Runtime::~Runtime() {
    if (sched_ != nullptr) ggml_backend_sched_free(sched_);
    if (backend_cpu_ != nullptr) ggml_backend_free(backend_cpu_);
    if (backend_ != nullptr) ggml_backend_free(backend_);
}

}  // namespace vt
