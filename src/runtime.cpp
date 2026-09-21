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

// The scheduler references the backends, so it must go first.
BackendState::~BackendState() {
    if (sched != nullptr) ggml_backend_sched_free(sched);
    if (backend_cpu != nullptr) ggml_backend_free(backend_cpu);
    if (backend != nullptr) ggml_backend_free(backend);
}

Runtime::Runtime() {
    ggml_backend_load_all();

    // Built into a local first: if we throw part-way, the shared_ptr unwinds and
    // frees whatever was already created (a throwing constructor never runs its own
    // destructor, so the old in-place version leaked the backend on failure).
    auto state = std::make_shared<BackendState>();

    ggml_backend_dev_t gpu = pick_gpu();
    if (gpu == nullptr) {
        throw std::runtime_error("Runtime: no GPU backend available (is ggml-vulkan built?)");
    }

    state->backend = ggml_backend_dev_init(gpu, nullptr);
    if (state->backend == nullptr) {
        throw std::runtime_error("Runtime: ggml_backend_dev_init failed");
    }
    name_ = std::string(ggml_backend_dev_name(gpu)) + ": " +
            ggml_backend_dev_description(gpu);

    // CPU backend as a fallback for ops the GPU backend does not implement.
    ggml_backend_t backends[2];
    int n = 0;
    backends[n++] = state->backend;
    const bool gpu_only = std::getenv("MINITORCH_GPU_ONLY") != nullptr;
    ggml_backend_dev_t cpu = gpu_only ? nullptr : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        state->backend_cpu = ggml_backend_dev_init(cpu, nullptr);
        if (state->backend_cpu != nullptr) backends[n++] = state->backend_cpu;
    }

    state->sched = ggml_backend_sched_new(backends, nullptr, n, /*graph_size=*/16384,
                                          /*parallel=*/false, /*op_offload=*/true);
    if (state->sched == nullptr) {
        throw std::runtime_error("Runtime: ggml_backend_sched_new failed");
    }

    state_ = std::move(state);
}

// Look the capability up through the backend registry rather than linking the
// Vulkan backend directly: src/ is backend-agnostic (VT_BACKEND picks vulkan or
// hip), so a missing or non-Vulkan backend must simply answer false.
bool Device::conv_coopmat() const {
    if (backend_ == nullptr) return false;

    ggml_backend_dev_t dev = ggml_backend_get_device(backend_);
    if (dev == nullptr) return false;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) return false;

    // Backends that do not provide it (CPU, HIP, a stub reg) answer NULL.
    typedef int (*fn_t)(ggml_backend_t);
    fn_t fn = reinterpret_cast<fn_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_conv_coopmat_available"));
    return fn != nullptr && fn(backend_) != 0;
}

}  // namespace vt
