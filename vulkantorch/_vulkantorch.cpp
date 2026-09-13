#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gguf_file.h"
#include "graph.h"
#include "memory.h"
#include "ops.h"
#include "runtime.h"
#include "tensor.h"

namespace py = pybind11;
using namespace vt;

PYBIND11_MODULE(_vulkantorch, m) {
    m.doc() = "vulkantorch: a minimal PyTorch-style tensor library on ggml-vulkan";

    py::class_<Device>(m, "Device")
        .def("name", [](const Device& d) { return d.name(); })
        .def("__repr__", [](const Device& d) { return "Device(" + d.name() + ")"; });

    py::class_<Runtime>(m, "Runtime")
        .def(py::init<>())
        .def("name", &Runtime::name)
        .def("gpu", &Runtime::gpu)
        .def("cpu", &Runtime::cpu);

    py::class_<Tensor>(m, "Tensor")
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("dim", &Tensor::dim)
        .def("to_host", &Tensor::to_host)
        .def("to_bytes", [](const Tensor& t) { return py::bytes(t.to_host_bytes()); })
        .def("mark_output", &Tensor::mark_output)
        .def("backend_name", &Tensor::backend_name)
        .def("__repr__", &Tensor::repr);

    py::class_<Graph>(m, "Graph")
        .def(py::init<Runtime&>(), py::arg("runtime"))
        .def(py::init<Runtime&, Device>(), py::arg("runtime"), py::arg("device"))
        .def("enter", &Graph::enter)
        .def("exit", &Graph::exit)
        .def("input", py::overload_cast<const std::vector<int64_t>&>(&Graph::input),
             py::arg("shape"))
        .def("input",
             py::overload_cast<const std::vector<int64_t>&, const std::vector<float>&>(
                 &Graph::input),
             py::arg("shape"), py::arg("data"))
        .def(
            "input",
            [](Graph& g, const std::vector<int64_t>& shape, py::bytes b) {
                const std::string s = b;
                return g.input(shape, s.data(), s.size());
            },
            py::arg("shape"), py::arg("data_bytes"))
        .def(
            "input_i32",
            [](Graph& g, const std::vector<int64_t>& shape, py::bytes b) {
                const std::string s = b;
                return g.input_i32(shape, s.data(), s.size());
            },
            py::arg("shape"), py::arg("data_bytes"))
        // graph-cache replay: update a captured input, then compute()
        // compute() is one-shot; replaying a scheduler-allocated graph changes the result,
        // so use alloc_static() once after capture, then compute_static() every replay.
        .def("compute", &Graph::compute)
        .def("alloc_static", &Graph::alloc_static)
        .def("compute_static", &Graph::compute_static)
        .def(
            "set_input",
            [](Graph& g, const Tensor& t, py::bytes b) {
                const std::string s = b;
                g.set_input(t.raw(), s.data(), s.size());
            },
            py::arg("tensor"), py::arg("data_bytes"));

    py::class_<Memory, std::shared_ptr<Memory>>(m, "Memory")
        .def(py::init<Device, size_t>(), py::arg("device"), py::arg("bytes"))
        .def("device", &Memory::device)
        .def("tensor", [](Memory& mem, const std::vector<int64_t>& shape) { return mem.tensor(shape); },
             py::arg("shape"))
        .def(
            "tensor",
            [](Memory& mem, const std::vector<int64_t>& shape, py::bytes b) {
                const std::string s = b;
                return mem.tensor(shape, s.data(), s.size());
            },
            py::arg("shape"), py::arg("data_bytes"))
        .def(
            "tensor",
            [](Memory& mem, const std::vector<int64_t>& shape, int dtype, py::bytes b) {
                const std::string s = b;
                return mem.tensor(shape, static_cast<ggml_type>(dtype), s.data(), s.size());
            },
            py::arg("shape"), py::arg("dtype"), py::arg("data_bytes"));

    py::class_<GgufFile>(m, "GgufFile")
        .def(py::init<const std::string&, Device, size_t>(), py::arg("path"), py::arg("device"),
             py::arg("arena_bytes"))
        .def("names", &GgufFile::names)
        .def("has", &GgufFile::has, py::arg("name"))
        .def("shape", &GgufFile::shape, py::arg("name"))
        .def("type_name", &GgufFile::type_name, py::arg("name"))
        .def("nbytes", &GgufFile::nbytes, py::arg("name"))
        .def("tensor", &GgufFile::tensor, py::arg("name"));

    m.attr("F32") = static_cast<int>(GGML_TYPE_F32);
    m.attr("F16") = static_cast<int>(GGML_TYPE_F16);

    m.def("matmul", &matmul, py::arg("a"), py::arg("b"));
    m.def("mul_mat", &mul_mat, py::arg("a"), py::arg("b"));
    m.def("add", &add, py::arg("a"), py::arg("b"));
    m.def("sub", &sub, py::arg("a"), py::arg("b"));
    m.def("mul", &mul, py::arg("a"), py::arg("b"));
    m.def("div", &vt::div, py::arg("a"), py::arg("b"));
    m.def("scale", &scale, py::arg("a"), py::arg("s"));
    m.def("scale_bias", &scale_bias, py::arg("a"), py::arg("s"), py::arg("b"));
    m.def("transpose", &transpose, py::arg("a"));
    m.def("contiguous", &contiguous, py::arg("a"));
    m.def("reshape", &reshape, py::arg("a"), py::arg("shape"));
    m.def("repeat", &repeat, py::arg("a"), py::arg("shape"));
    m.def("gelu", &gelu, py::arg("a"));
    m.def("gelu_erf", &gelu_erf, py::arg("a"));
    m.def("relu", &relu, py::arg("a"));
    m.def("sigmoid", &sigmoid, py::arg("a"));
    m.def("exp", &vt::exp, py::arg("a"));
    m.def("elu", &elu, py::arg("a"));
    m.def("silu", &silu, py::arg("a"));
    m.def("tanh", &vt::tanh, py::arg("a"));
    m.def("soft_max", &soft_max, py::arg("a"));
    m.def("softplus", &vt::softplus, py::arg("a"));
    m.def("sin", &vt::sin, py::arg("a"));
    m.def("cos", &vt::cos, py::arg("a"));
    m.def("sqrt", &vt::sqrt, py::arg("a"));
    m.def("sqr", &vt::sqr, py::arg("a"));
    m.def("linear", &linear, py::arg("x"), py::arg("w"));
    m.def("rms_norm", &rms_norm, py::arg("a"), py::arg("eps"));
    m.def("layer_norm", &layer_norm, py::arg("a"), py::arg("eps"));
    m.def("group_norm", &group_norm, py::arg("a"), py::arg("n_groups"), py::arg("eps"));
    m.def("diag_mask_inf", &diag_mask_inf, py::arg("a"), py::arg("n_past"));
    m.def("concat", &concat, py::arg("a"), py::arg("b"), py::arg("dim"));
    m.def("argmax", &argmax, py::arg("a"));
    m.def("get_rows", &get_rows, py::arg("a"), py::arg("ids"));
    m.def("sum_rows", &sum_rows, py::arg("a"));
    m.def("cast", [](const Tensor& a, int t) { return cast(a, static_cast<ggml_type>(t)); },
          py::arg("a"), py::arg("type"));
    m.def("cpy", &cpy, py::arg("a"), py::arg("dst"));
    m.def("set_rows", &set_rows, py::arg("dst"), py::arg("src"), py::arg("idx"));
    m.def("view_1d", &view_1d, py::arg("a"), py::arg("ne0"), py::arg("offset"));
    m.def("view_2d", &view_2d, py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("nb1"),
          py::arg("offset"));
    m.def("view_3d", &view_3d, py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"),
          py::arg("nb1"), py::arg("nb2"), py::arg("offset"));
    m.def("view_4d", &view_4d, py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"),
          py::arg("ne3"), py::arg("nb1"), py::arg("nb2"), py::arg("nb3"), py::arg("offset"));
    m.def("permute_pt", &permute_pt, py::arg("a"), py::arg("p0"), py::arg("p1"), py::arg("p2"),
          py::arg("p3"));
    m.def("rope", &rope, py::arg("a"), py::arg("pos"), py::arg("n_dims"), py::arg("mode"),
          py::arg("n_ctx_orig"), py::arg("freq_base"), py::arg("freq_scale"), py::arg("ext_factor"),
          py::arg("attn_factor"), py::arg("beta_fast"), py::arg("beta_slow"));
    m.def(
        "flash_attn",
        [](const Tensor& q, const Tensor& k, const Tensor& v, py::object mask, float scale,
           float max_bias, float logit_softcap) {
            Tensor m;
            if (!mask.is_none()) m = mask.cast<Tensor>();
            return flash_attn(q, k, v, m, scale, max_bias, logit_softcap);
        },
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask") = py::none(),
        py::arg("scale"), py::arg("max_bias") = 0.0f, py::arg("logit_softcap") = 0.0f);
    m.def("conv1d", &conv1d, py::arg("x"), py::arg("w"), py::arg("stride") = 1, py::arg("pad") = 0,
          py::arg("dilation") = 1);
    m.def("conv1d_dw", &conv1d_dw, py::arg("x"), py::arg("w"), py::arg("stride") = 1,
          py::arg("pad") = 0, py::arg("dilation") = 1);
    m.def("conv2d", &conv2d, py::arg("a"), py::arg("b"), py::arg("s0") = 1, py::arg("s1") = 1,
          py::arg("p0") = 0, py::arg("p1") = 0, py::arg("d0") = 1, py::arg("d1") = 1);
    m.def("conv2d_dw", &conv2d_dw, py::arg("a"), py::arg("b"), py::arg("s0") = 1,
          py::arg("s1") = 1, py::arg("p0") = 0, py::arg("p1") = 0, py::arg("d0") = 1,
          py::arg("d1") = 1);
    m.def("conv_transpose_2d", &conv_transpose_2d, py::arg("a"), py::arg("b"), py::arg("stride"));
    m.def("pool_2d", &pool_2d, py::arg("a"), py::arg("op"), py::arg("k0"), py::arg("k1"),
          py::arg("s0"), py::arg("s1"), py::arg("p0") = 0.0f, py::arg("p1") = 0.0f);
    m.def("upsample", &upsample, py::arg("a"), py::arg("scale_factor"), py::arg("mode") = 0);
    m.def("pad", &pad, py::arg("a"), py::arg("p0"), py::arg("p1"), py::arg("p2") = 0,
          py::arg("p3") = 0);
    m.def("clamp", &clamp, py::arg("a"), py::arg("min_v"), py::arg("max_v"));
    m.def("conv_transpose_1d", &conv_transpose_1d, py::arg("x"), py::arg("w_perm"),
          py::arg("stride"), py::arg("oc"));
    m.def("col2im_1d", &col2im_1d, py::arg("col"), py::arg("s0"), py::arg("oc"), py::arg("p0"));
    m.def("snake_1d", &snake_1d, py::arg("x"), py::arg("alpha"));
    m.def("im2col_rafa", [](const Tensor& x, int K, int s0, int p0, int d0) {
        return im2col_rafa(x, K, s0, p0, d0, GGML_TYPE_F32);
    }, py::arg("x"), py::arg("K"), py::arg("s0"), py::arg("p0"), py::arg("d0"));
}
