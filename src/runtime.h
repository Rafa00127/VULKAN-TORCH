#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace vt {

class Runtime;

// A compute device: a handle to a ggml backend (GPU or CPU). Mirrors torch's
// torch.device. Placement is explicit: a Graph runs its nodes on one Device.
class Device {
public:
    Device() = default;
    Device(ggml_backend_t backend, std::string name)
        : backend_(backend), name_(std::move(name)) {}

    ggml_backend_t backend() const { return backend_; }
    const std::string& name() const { return name_; }
    bool defined() const { return backend_ != nullptr; }

private:
    ggml_backend_t backend_ = nullptr;
    std::string name_;
};

// Owns the ggml backend and scheduler. Reused across captures.
class Runtime {
public:
    Runtime();
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    const char* name() const { return name_.c_str(); }
    ggml_backend_t backend() const { return backend_; }
    ggml_backend_sched_t sched() const { return sched_; }

    Device gpu() const { return Device(backend_, name_); }
    Device cpu() const { return Device(backend_cpu_, "cpu"); }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t backend_cpu_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    std::string name_;
};

}  // namespace vt
