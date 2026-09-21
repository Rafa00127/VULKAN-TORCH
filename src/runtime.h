#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vt {

class Runtime;

// The ggml objects a Device borrows from, refcounted so they cannot be freed while
// anything still points at them.
//
// Why this exists: a Graph stores the handles it takes from a Runtime (the scheduler
// and the backend) as bare pointers, and their owner used to be the Runtime alone.
// That is fine in C++, where locals die in reverse order -- but the objects here are
// usually owned by *Python*, which drops its module globals in an arbitrary order at
// interpreter shutdown. Destroy the Runtime first and a later ~Graph dereferences a
// freed scheduler (use-after-free, segfault, sometimes heap corruption). Moving the
// owned handles into a shared state and having Device hold a share of it makes the
// destruction order irrelevant: whoever still holds a Device keeps them alive.
struct BackendState {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ~BackendState();
};

// A compute device: a handle to a ggml backend (GPU or CPU). Mirrors torch's
// torch.device. Placement is explicit: a Graph runs its nodes on one Device.
class Device {
public:
    Device() = default;
    // `state` is the Runtime's; pass it unless you own the handles yourself (in
    // which case you also own keeping them alive for as long as this Device lives).
    Device(ggml_backend_t backend, std::string name,
           std::shared_ptr<BackendState> state = nullptr)
        : backend_(backend), name_(std::move(name)), state_(std::move(state)) {}

    ggml_backend_t backend() const { return backend_; }
    const std::string& name() const { return name_; }
    bool defined() const { return backend_ != nullptr; }

    // Where the backend runs its fused 2D conv (GGML_OP_CONV_2D) on matrix cores.
    // That is what decides whether the fused conv beats the im2col one, so callers
    // route on it. Backend-generic: an unknown backend (or a CPU device) answers
    // false, and the fused path must then be treated as the scalar-FMA fallback.
    bool conv_coopmat() const;

private:
    ggml_backend_t backend_ = nullptr;
    std::string name_;

    // Keeping this alive is the whole point: a Graph/Memory/GgufFile that copies the
    // Device can then outlive the Runtime it was built from.
    std::shared_ptr<BackendState> state_;
};

// Owns the ggml backend and scheduler. Reused across captures.
class Runtime {
public:
    Runtime();
    ~Runtime() = default;

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    const char* name() const { return name_.c_str(); }
    ggml_backend_t backend() const { return state_->backend; }
    ggml_backend_sched_t sched() const { return state_->sched; }

    Device gpu() const { return Device(state_->backend, name_, state_); }
    Device cpu() const { return Device(state_->backend_cpu, "cpu", state_); }

private:
    // Shared with every Device handed out; frees the ggml objects when the last of
    // them (and this Runtime) is gone.
    std::shared_ptr<BackendState> state_;
    std::string name_;
};

}  // namespace vt
