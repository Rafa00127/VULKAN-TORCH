// Vulkan backend operator microbenchmark.
//
// A single, self-contained source that compiles against TWO ggml trees (the
// project's vendored 0.10.2 and an external reference 0.23.0) so the same op
// set can be timed against both Vulkan backends and compared head-to-head.
//
// Timing follows ggml's tests/test-backend-ops.cpp perf mode: build the graph,
// allocate via ggml_backend_alloc_ctx_tensors, warm up, then time a graph made
// of N copies of the same node until >=1s of work, reporting average per-op
// time (plus GFLOP/s for compute-bound ops).
//
// NOTE: this measures the raw backend (ggml_backend_graph_compute), NOT the
// project's scheduler path in src/runtime.cpp. Absolute numbers are therefore
// not comparable to bench_cpp.exe -- but A vs B use the identical primitive, so
// the cross-tree comparison is valid.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#ifndef BENCH_VARIANT
#define BENCH_VARIANT "unknown"
#endif

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static uint32_t g_rng = 0x12345678u;

static float frand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((g_rng >> 8) & 0xffff) / 32768.0f - 1.0f;  // [-1, 1)
}

// Fill a tensor's host buffer with type-appropriate data. Integer tensors get
// random values in [0, ne0) (safe indices for get_rows/set_rows on this case).
static void init_tensor(ggml_tensor * t, int64_t int_lo = 0, int64_t int_hi = -1) {
    const size_t n = (size_t) ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> data(n);
        for (auto & x : data) x = frand();
        ggml_backend_tensor_set(t, data.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data(n);
        for (auto & x : data) x = ggml_fp32_to_fp16(frand());
        ggml_backend_tensor_set(t, data.data(), 0, n * sizeof(ggml_fp16_t));
    } else if (t->type == GGML_TYPE_I32) {
        const int64_t hi = int_hi >= 0 ? int_hi : (t->ne[0] > 0 ? t->ne[0] : 1);
        std::vector<int32_t> data(n);
        for (auto & x : data) x = int_lo + (int32_t) (std::abs((int) frand() * 100000) % (hi - int_lo > 0 ? hi - int_lo : 1));
        ggml_backend_tensor_set(t, data.data(), 0, n * sizeof(int32_t));
    } else if (t->type == GGML_TYPE_I64) {
        const int64_t hi = int_hi >= 0 ? int_hi : (t->ne[0] > 0 ? t->ne[0] : 1);
        std::vector<int64_t> data(n);
        for (auto & x : data) x = int_lo + (int64_t) (std::abs((int) frand() * 100000) % (hi - int_lo > 0 ? hi - int_lo : 1));
        ggml_backend_tensor_set(t, data.data(), 0, n * sizeof(int64_t));
    } else {
        // anything else (quantized...): leave zeroed
        std::vector<uint8_t> data(ggml_nbytes(t), 0);
        ggml_backend_tensor_set(t, data.data(), 0, data.size());
    }
}

// ---------------------------------------------------------------------------
// case interface
// ---------------------------------------------------------------------------

struct bench_case {
    virtual ~bench_case() = default;

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;
    virtual const char  * op()   const = 0;
    virtual std::string   vars() const { return ""; }
    virtual uint64_t      flops(ggml_tensor * out) const { (void) out; return 0; }
    virtual bool          composite() const { return false; }

    // Optional independent numeric cross-check of this op (--verify). Empty means
    // this case has no reference to compare against. Return a one-line report.
    virtual std::string   verify(ggml_backend_t backend) { (void) backend; return std::string(); }

    // Default: fill every tensor uniformly. Index-driven ops override this.
    virtual void initialize(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor(t);
        }
    }
};

using case_ptr = std::unique_ptr<bench_case>;

// Convenience: create + name a tensor.
static ggml_tensor * nt(ggml_context * ctx, ggml_type type, const char * name,
                        int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, type, n0, n1, n2, n3);
    ggml_set_name(t, name);
    return t;
}

// ---------------------------------------------------------------------------
// fused-node cases (clean backend comparison)
// ---------------------------------------------------------------------------

struct case_mul_mat : bench_case {
    ggml_type ta, tb;
    int64_t k, m, n, bs;
    case_mul_mat(ggml_type ta, ggml_type tb, int64_t k, int64_t n, int64_t m, int64_t bs = 1)
        : ta(ta), tb(tb), k(k), m(m), n(n), bs(bs) {}
    const char * op() const override { return "mul_mat"; }
    std::string vars() const override {
        char b[128];
        snprintf(b, sizeof(b), "k=%lld n=%lld m=%lld bs=%lld",
                 (long long) k, (long long) n, (long long) m, (long long) bs);
        return std::string(b) + " src0=" + ggml_type_name(ta) + " src1=" + ggml_type_name(tb);
    }
    uint64_t flops(ggml_tensor * out) const override {
        (void) out;
        return (uint64_t) 2 * k * m * n * bs;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // ggml layout: a = [k, m, bs], b = [k, n, bs] -> out [n, m, bs]
        ggml_tensor * a = nt(ctx, ta, "a", k, m, bs);
        ggml_tensor * b = nt(ctx, tb, "b", k, n, bs);
        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_flash_attn : bench_case {
    int64_t hd, nh, nkv, T;
    case_flash_attn(int64_t hd, int64_t nh, int64_t nkv, int64_t T) : hd(hd), nh(nh), nkv(nkv), T(T) {}
    const char * op() const override { return "flash_attn_ext"; }
    std::string vars() const override {
        char b[128];
        snprintf(b, sizeof(b), "hd=%lld nh=%lld nkv=%lld T=%lld", (long long) hd, (long long) nh, (long long) nkv, (long long) T);
        return b;
    }
    uint64_t flops(ggml_tensor * out) const override {
        (void) out;
        // ~ 4 * T * T * hd * nh (qk + softmax*V), times 2
        return (uint64_t) 4 * T * T * hd * nh;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * q = nt(ctx, GGML_TYPE_F32, "q", hd, T, nh, 1);
        ggml_tensor * k = nt(ctx, GGML_TYPE_F16, "k", hd, T, nkv, 1);
        ggml_tensor * v = nt(ctx, GGML_TYPE_F16, "v", hd, T, nkv, 1);
        ggml_tensor * mask = nt(ctx, GGML_TYPE_F16, "mask", T, T, 1, 1);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_im2col : bench_case {
    bool is_2d;
    int64_t W, H, C, KW, KH;
    // is_2d: x:[W,H,C,N], k:[KW,KH,IC=C,1]
    // 1d:    x:[T,C,1,1],  k:[K,IC=C,1,1]
    case_im2col(bool is_2d, int64_t W, int64_t H, int64_t C, int64_t KW, int64_t KH = 1)
        : is_2d(is_2d), W(W), H(H), C(C), KW(KW), KH(KH) {}
    const char * op() const override { return "im2col"; }
    std::string vars() const override {
        char b[128];
        if (is_2d) snprintf(b, sizeof(b), "2d W=%lld H=%lld IC=%lld K=%lldx%lld", (long long) W, (long long) H, (long long) C, (long long) KW, (long long) KH);
        else       snprintf(b, sizeof(b), "1d T=%lld IC=%lld K=%lld", (long long) W, (long long) C, (long long) KW);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x, * k, * out;
        if (is_2d) {
            x = nt(ctx, GGML_TYPE_F32, "x", W, H, C, 1);
            k = nt(ctx, GGML_TYPE_F32, "k", KW, KH, C, 1);
            out = ggml_im2col(ctx, k, x, 1, 1, KH / 2, KW / 2, 1, 1, true, GGML_TYPE_F32);
        } else {
            x = nt(ctx, GGML_TYPE_F32, "x", W, C, 1, 1);
            k = nt(ctx, GGML_TYPE_F32, "k", KW, C, 1, 1);
            out = ggml_im2col(ctx, k, x, 1, 0, KW / 2, 0, 1, 0, false, GGML_TYPE_F32);
        }
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_col2im_1d : bench_case {
    int64_t K, OC, T_in, s;
    case_col2im_1d(int64_t K, int64_t OC, int64_t T_in, int64_t s) : K(K), OC(OC), T_in(T_in), s(s) {}
    const char * op() const override { return "col2im_1d"; }
    std::string vars() const override {
        char b[128];
        snprintf(b, sizeof(b), "K=%lld OC=%lld T_in=%lld s=%lld", (long long) K, (long long) OC, (long long) T_in, (long long) s);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * col = nt(ctx, GGML_TYPE_F32, "col", K * OC, T_in);
        ggml_tensor * out = ggml_col2im_1d(ctx, col, (int) s, (int) OC, 0);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_conv_transpose_2d : bench_case {
    int64_t IC, OC, IH, IW, KH, KW, stride, N;
    case_conv_transpose_2d(int64_t IC, int64_t OC, int64_t IH, int64_t IW, int64_t KH, int64_t KW, int64_t stride)
        : IC(IC), OC(OC), IH(IH), IW(IW), KH(KH), KW(KW), stride(stride), N(1) {}
    const char * op() const override { return "conv_transpose_2d"; }
    std::string vars() const override {
        char b[160];
        snprintf(b, sizeof(b), "IC=%lld OC=%lld H=%lld W=%lld K=%lldx%lld s=%lld",
                 (long long) IC, (long long) OC, (long long) IH, (long long) IW, (long long) KH, (long long) KW, (long long) stride);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // conv_transpose_2d_p0 wants kernel ne-space [KW, KH, OC, IC] (PT [IC, OC, KH, KW])
        // and data [IW, IH, IC, N]; assert is a->ne[3] (IC) == b->ne[2] (IC).
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", KW, KH, OC, IC);
        ggml_tensor * b = nt(ctx, GGML_TYPE_F32, "b", IW, IH, IC, N);
        ggml_tensor * out = ggml_conv_transpose_2d_p0(ctx, a, b, (int) stride);
        ggml_set_name(out, "out");
        return out;
    }

    // No composite alternative exists for this op (unlike conv2d, whose im2col path
    // is a second implementation), so the reference is a double-precision CPU
    // scatter, mirroring ggml-cpu's:
    //     dst[oc, ih*s + kh, iw*s + kw] += sum_ic w[kw,kh,oc,ic] * x[iw,ih,ic]
    // (p0 = 0, so the output is the full (I-1)*s + K). Spatial is shrunk for the CPU
    // pass; IC/OC/K -- hence the reduction and the backend tile -- are untouched.
    std::string verify(ggml_backend_t backend) override {
        const int64_t vw = IW > 96 ? 68 : IW, vh = IH > 32 ? 20 : IH;
        const int64_t OW = (vw - 1) * stride + KW, OH = (vh - 1) * stride + KH;
        const int64_t vn = OW * OH * OC * N;

        const size_t na = (size_t) (KW * KH * OC * IC);
        const size_t nb = (size_t) (vw * vh * IC * N);
        std::vector<float> ha(na), hb(nb);
        for (auto & x : ha) x = frand();
        for (auto & x : hb) x = frand();

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(1024, false),
            /* .mem_base   = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return "FAIL: ggml_init";

        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", KW, KH, OC, IC);
        ggml_tensor * b = nt(ctx, GGML_TYPE_F32, "b", vw, vh, IC, N);
        ggml_tensor * out = ggml_conv_transpose_2d_p0(ctx, a, b, (int) stride);
        ggml_set_name(out, "out");

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { ggml_free(ctx); return "FAIL: alloc_ctx_tensors"; }
        ggml_backend_tensor_set(a, ha.data(), 0, na * sizeof(float));
        ggml_backend_tensor_set(b, hb.data(), 0, nb * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(gf, out);
        ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return std::string("FAIL: graph_compute (") + ggml_status_to_string(st) + ")";
        }
        if (ggml_nelements(out) != vn) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            char m[160];
            snprintf(m, sizeof(m), "FAIL: nelements %lld, expected %lld", (long long) ggml_nelements(out), (long long) vn);
            return m;
        }
        std::vector<float> vo((size_t) vn);
        ggml_backend_tensor_get(out, vo.data(), 0, (size_t) vn * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        // ne=[OW,OH,OC,N] / a ne=[KW,KH,OC,IC] / b ne=[IW,IH,IC,N]
        std::vector<double> vc((size_t) vn, 0.0);
        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t ih = 0; ih < vh; ++ih) {
                for (int64_t iw = 0; iw < vw; ++iw) {
                    for (int64_t kh = 0; kh < KH; ++kh) {
                        for (int64_t kw = 0; kw < KW; ++kw) {
                            const int64_t oh = ih * stride + kh, ow = iw * stride + kw;
                            double acc = 0.0;
                            for (int64_t ic = 0; ic < IC; ++ic) {
                                acc += (double) ha[(size_t) (kw + kh * KW + oc * KW * KH + ic * KW * KH * OC)]
                                     * (double) hb[(size_t) (iw + ih * vw + ic * vw * vh)];
                            }
                            vc[(size_t) (ow + oh * OW + oc * OW * OH)] += acc;
                        }
                    }
                }
            }
        }

        size_t n_bad = 0;
        double max_abs = 0.0, mean = 0.0, scale = 0.0;
        for (int64_t i = 0; i < vn; ++i) {
            const double x = vo[(size_t) i];
            if (!std::isfinite(x)) ++n_bad;
            scale = std::max(scale, std::fabs(vc[(size_t) i]));
            const double d = std::fabs(x - vc[(size_t) i]);
            max_abs = std::max(max_abs, d);
            mean += d;
        }
        char m[240];
        snprintf(m, sizeof(m),
                 "vs double cpu: max=%.3e rel=%.3e mean=%.3e  nonfinite=%zu/%lld  stride=%lld [spatial %lldx%lld]",
                 max_abs, scale > 0 ? max_abs / scale : 0.0, mean / (double) vn, n_bad, (long long) vn,
                 (long long) stride, (long long) vw, (long long) vh);
        return std::string(n_bad ? "FAIL: non-finite  " : "ok   ") + m;
    }
};

struct case_pool_2d : bench_case {
    ggml_op_pool pool;
    int64_t W, H, C, K, S;
    case_pool_2d(ggml_op_pool pool, int64_t W, int64_t H, int64_t C, int64_t K, int64_t S)
        : pool(pool), W(W), H(H), C(C), K(K), S(S) {}
    const char * op() const override { return "pool_2d"; }
    std::string vars() const override {
        char b[128];
        snprintf(b, sizeof(b), "%s W=%lld H=%lld C=%lld k=%lld s=%lld",
                 pool == GGML_OP_POOL_MAX ? "max" : "avg", (long long) W, (long long) H, (long long) C, (long long) K, (long long) S);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", W, H, C, 1);
        ggml_tensor * out = ggml_pool_2d(ctx, a, pool, (int) K, (int) K, (int) S, (int) S, 0.0f, 0.0f);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_upscale : bench_case {
    ggml_scale_mode mode;
    int64_t W, H, C, S;
    case_upscale(ggml_scale_mode mode, int64_t W, int64_t H, int64_t C, int64_t S)
        : mode(mode), W(W), H(H), C(C), S(S) {}
    const char * op() const override { return "upscale"; }
    std::string vars() const override {
        char b[128];
        snprintf(b, sizeof(b), "%s W=%lld H=%lld C=%lld s=%lld",
                 mode == GGML_SCALE_MODE_BILINEAR ? "bilinear" : "nearest", (long long) W, (long long) H, (long long) C, (long long) S);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", W, H, C, 1);
        ggml_tensor * out = ggml_upscale(ctx, a, (int) S, mode);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_pad : bench_case {
    int64_t W, H, C, p0, p1;
    case_pad(int64_t W, int64_t H, int64_t C, int64_t p0, int64_t p1) : W(W), H(H), C(C), p0(p0), p1(p1) {}
    const char * op() const override { return "pad"; }
    std::string vars() const override {
        char b[128];
        snprintf(b, sizeof(b), "W=%lld H=%lld C=%lld p=%lld,%lld", (long long) W, (long long) H, (long long) C, (long long) p0, (long long) p1);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", W, H, C, 1);
        ggml_tensor * out = ggml_pad(ctx, a, (int) p0, (int) p1, 0, 0);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_soft_max : bench_case {
    int64_t W, H; bool with_mask;
    case_soft_max(int64_t W, int64_t H, bool with_mask) : W(W), H(H), with_mask(with_mask) {}
    const char * op() const override { return "soft_max"; }
    std::string vars() const override {
        char b[96];
        snprintf(b, sizeof(b), "W=%lld H=%lld mask=%d", (long long) W, (long long) H, with_mask ? 1 : 0);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", W, H, 1, 1);
        ggml_tensor * mask = with_mask ? nt(ctx, GGML_TYPE_F16, "mask", W, H, 1, 1) : nullptr;
        ggml_tensor * out = ggml_soft_max_ext(ctx, a, mask, 1.0f, 0.0f);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_scale : bench_case {
    int64_t W, H, C;
    case_scale(int64_t W, int64_t H, int64_t C) : W(W), H(H), C(C) {}
    const char * op() const override { return "scale"; }
    std::string vars() const override {
        char b[96];
        snprintf(b, sizeof(b), "W=%lld H=%lld C=%lld", (long long) W, (long long) H, (long long) C);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", W, H, C, 1);
        ggml_tensor * out = ggml_scale(ctx, a, 1.5f);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_rope : bench_case {
    int64_t hd, T, nh;
    case_rope(int64_t hd, int64_t T, int64_t nh) : hd(hd), T(T), nh(nh) {}
    const char * op() const override { return "rope"; }
    std::string vars() const override {
        char b[96];
        snprintf(b, sizeof(b), "hd=%lld T=%lld nh=%lld", (long long) hd, (long long) T, (long long) nh);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // ggml rope: a = [hd, nh, T, 1] (ne2 = n_tokens); pos vector ne0 = T.
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", hd, nh, T, 1);
        ggml_tensor * pos = nt(ctx, GGML_TYPE_I32, "pos", T, 1, 1, 1);
        ggml_tensor * out = ggml_rope_ext(ctx, a, pos, nullptr, (int) hd, 0, 0,
                                          10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_rms_norm : bench_case {
    int64_t W, H;
    case_rms_norm(int64_t W, int64_t H) : W(W), H(H) {}
    const char * op() const override { return "rms_norm"; }
    std::string vars() const override {
        char b[96];
        snprintf(b, sizeof(b), "W=%lld H=%lld", (long long) W, (long long) H);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", W, H, 1, 1);
        ggml_tensor * out = ggml_rms_norm(ctx, a, 1e-6f);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_group_norm : bench_case {
    int64_t W, C, groups;
    case_group_norm(int64_t W, int64_t C, int64_t groups) : W(W), C(C), groups(groups) {}
    const char * op() const override { return "group_norm"; }
    std::string vars() const override {
        char b[96];
        snprintf(b, sizeof(b), "W=%lld C=%lld g=%lld", (long long) W, (long long) C, (long long) groups);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", W, C, 1, 1);
        ggml_tensor * out = ggml_group_norm(ctx, a, (int) groups, 1e-6f);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_unary : bench_case {
    enum kind_t { SILU, GELU } kind;
    int64_t W, H;
    case_unary(kind_t kind, int64_t W, int64_t H) : kind(kind), W(W), H(H) {}
    const char * op() const override { return kind == SILU ? "silu" : "gelu"; }
    std::string vars() const override {
        char b[96];
        snprintf(b, sizeof(b), "W=%lld H=%lld", (long long) W, (long long) H);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", W, H, 1, 1);
        ggml_tensor * out = kind == SILU ? ggml_silu(ctx, a) : ggml_gelu(ctx, a);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_concat : bench_case {
    int64_t W1, W2, H;
    case_concat(int64_t W1, int64_t W2, int64_t H) : W1(W1), W2(W2), H(H) {}
    const char * op() const override { return "concat"; }
    std::string vars() const override {
        char b[96];
        snprintf(b, sizeof(b), "W1=%lld W2=%lld H=%lld", (long long) W1, (long long) W2, (long long) H);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", W1, H, 1, 1);
        ggml_tensor * b = nt(ctx, GGML_TYPE_F32, "b", W2, H, 1, 1);
        ggml_tensor * out = ggml_concat(ctx, a, b, 0);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_get_rows : bench_case {
    int64_t in_dim, rows;  // a: [in_dim, n_rows_out]; ids: [n_rows_out]
    case_get_rows(int64_t in_dim, int64_t rows) : in_dim(in_dim), rows(rows) {}
    const char * op() const override { return "get_rows"; }
    std::string vars() const override {
        char b[96];
        snprintf(b, sizeof(b), "n_embd=%lld n_rows=%lld type=f16", (long long) in_dim, (long long) rows);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F16, "a", in_dim, 4096, 1, 1);  // embedding table
        ggml_tensor * ids = nt(ctx, GGML_TYPE_I32, "ids", rows, 1, 1, 1);
        ggml_tensor * out = ggml_get_rows(ctx, a, ids);
        ggml_set_name(out, "out");
        return out;
    }
    void initialize(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) init_tensor(t, 0, 4096);
            else init_tensor(t);
        }
    }
};

struct case_set_rows : bench_case {
    int64_t in_dim, rows, cap;
    case_set_rows(int64_t in_dim, int64_t rows, int64_t cap) : in_dim(in_dim), rows(rows), cap(cap) {}
    const char * op() const override { return "set_rows"; }
    std::string vars() const override {
        char b[128];
        snprintf(b, sizeof(b), "n_embd=%lld n_set=%lld cap=%lld", (long long) in_dim, (long long) rows, (long long) cap);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * dst = nt(ctx, GGML_TYPE_F32, "dst", in_dim, cap, 1, 1);
        ggml_tensor * src = nt(ctx, GGML_TYPE_F32, "src", in_dim, rows, 1, 1);
        ggml_tensor * idx = nt(ctx, GGML_TYPE_I32, "idx", rows, 1, 1, 1);
        ggml_tensor * out = ggml_set_rows(ctx, dst, src, idx);
        ggml_set_name(out, "out");
        return out;
    }
    void initialize(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) init_tensor(t, 0, cap);
            else init_tensor(t);
        }
    }
};

// ---------------------------------------------------------------------------
// composite cases (im2col->matmul subgraphs; A/B may mix in algorithmic diff)
// ---------------------------------------------------------------------------

struct case_conv2d : bench_case {
    // b (data) [IW, IH, IC, N], a (kernel) [KW, KH, IC, OC] -> out [OW, OH, OC, N]
    int64_t IW, IH, IC, OC, KW, KH, s;
    case_conv2d(int64_t IW, int64_t IH, int64_t IC, int64_t OC, int64_t KW, int64_t KH, int64_t s)
        : IW(IW), IH(IH), IC(IC), OC(OC), KW(KW), KH(KH), s(s) {}
    const char * op() const override { return "conv2d"; }
    bool composite() const override { return true; }
    std::string vars() const override {
        char b[160];
        snprintf(b, sizeof(b), "IC=%lld OC=%lld H=%lld W=%lld K=%lldx%lld s=%lld",
                 (long long) IC, (long long) OC, (long long) IH, (long long) IW, (long long) KW, (long long) KH, (long long) s);
        return b;
    }
    uint64_t flops(ggml_tensor * out) const override {
        (void) out;
        const int64_t OH = (IH + 2 * (KH / 2) - KH) / s + 1;
        const int64_t OW = (IW + 2 * (KW / 2) - KW) / s + 1;
        return (uint64_t) 2 * OW * OH * OC * (IC * KH * KW);
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", KW, KH, IC, OC);
        ggml_tensor * b = nt(ctx, GGML_TYPE_F32, "b", IW, IH, IC, 1);
        ggml_tensor * out = ggml_conv_2d(ctx, a, b, (int) s, (int) s, (int) (KW / 2), (int) (KH / 2), 1, 1);
        ggml_set_name(out, "out");
        return out;
    }
};

// Single-node 2D conv: one GGML_OP_CONV_2D node lowered by the implicit-GEMM conv
// shader (conv2d_mm.comp), NOT the im2col->mul_mat composite above. This is the
// path the det router picks for big non-3x3 kernels, so it is the one the coopmat
// work targets.
struct case_conv2d_direct : bench_case {
    // b (data) [IW, IH, IC, N=1], a (kernel) [KW, KH, IC, OC] -> out [OW, OH, OC, 1]
    int64_t IW, IH, IC, OC, KW, KH, s;
    case_conv2d_direct(int64_t IW, int64_t IH, int64_t IC, int64_t OC, int64_t KW, int64_t KH, int64_t s)
        : IW(IW), IH(IH), IC(IC), OC(OC), KW(KW), KH(KH), s(s) {}
    const char * op() const override { return "conv2d_direct"; }
    std::string vars() const override {
        char b[160];
        snprintf(b, sizeof(b), "IC=%lld OC=%lld H=%lld W=%lld K=%lldx%lld s=%lld",
                 (long long) IC, (long long) OC, (long long) IH, (long long) IW, (long long) KW, (long long) KH, (long long) s);
        return b;
    }
    uint64_t flops(ggml_tensor * out) const override {
        (void) out;
        const int64_t OH = (IH + 2 * (KH / 2) - KH) / s + 1;
        const int64_t OW = (IW + 2 * (KW / 2) - KW) / s + 1;
        return (uint64_t) 2 * OW * OH * OC * (IC * KH * KW);
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", KW, KH, IC, OC);
        ggml_tensor * b = nt(ctx, GGML_TYPE_F32, "b", IW, IH, IC, 1);
        ggml_tensor * out = ggml_conv_2d_direct(ctx, a, b, (int) s, (int) s, (int) (KW / 2), (int) (KH / 2), 1, 1);
        ggml_set_name(out, "out");
        return out;
    }

    // The fused node and the im2col->mul_mat composite both emit ne=[OW,OH,OC,N],
    // so they are directly comparable. They share the kernel/input tensors, which
    // makes the inputs bit-identical. mul_mat is a different shader entirely, so
    // it is an independent implementation, not a copy of what we are checking.
    // Three-way check at a small spatial extent (so a double-precision CPU
    // reference stays cheap) while IC/OC/K -- and hence CRS, the accumulation
    // depth the fp32-accumulator decision is about -- stay at the shape under
    // test. The CPU number is the only absolute one: it says whether the fused
    // kernel is fp32-accurate, whereas the im2col GPU path is itself a suspect
    // (its own im2col/GEMM may quantize), so "agrees with it" alone proves little.
    std::string verify(ggml_backend_t backend) override {
        const int p0 = (int) (KW / 2), p1 = (int) (KH / 2);
        const int64_t vw = KW + 6, vh = KH + 2;
        const int64_t OW = (vw + 2 * p0 - KW) / s + 1;
        const int64_t OH = (vh + 2 * p1 - KH) / s + 1;

        const size_t na = (size_t) (KW * KH * IC * OC);
        const size_t nb = (size_t) (vw * vh * IC);
        std::vector<float> ha(na), hb(nb);
        for (auto & x : ha) x = frand();
        for (auto & x : hb) x = frand();

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(1024, false),
            /* .mem_base   = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return "FAIL: ggml_init";

        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", KW, KH, IC, OC);
        ggml_tensor * b = nt(ctx, GGML_TYPE_F32, "b", vw, vh, IC, 1);
        ggml_tensor * fused = ggml_conv_2d_direct(ctx, a, b, (int) s, (int) s, p0, p1, 1, 1);
        ggml_tensor * ref   = ggml_conv_2d(ctx, a, b, (int) s, (int) s, p0, p1, 1, 1);
        ggml_set_name(fused, "fused");
        ggml_set_name(ref, "ref");

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { ggml_free(ctx); return "FAIL: alloc_ctx_tensors"; }
        ggml_backend_tensor_set(a, ha.data(), 0, na * sizeof(float));
        ggml_backend_tensor_set(b, hb.data(), 0, nb * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(gf, fused);
        ggml_build_forward_expand(gf, ref);

        ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return std::string("FAIL: graph_compute (") + ggml_status_to_string(st) + ")";
        }

        const int64_t nf = ggml_nelements(fused), nr = ggml_nelements(ref);
        if (nf != nr || nf != OW * OH * OC) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            char m[160];
            snprintf(m, sizeof(m), "FAIL: nelements fused=%lld ref=%lld expected=%lld",
                     (long long) nf, (long long) nr, (long long) (OW * OH * OC));
            return m;
        }
        std::vector<float> vf((size_t) nf), vr((size_t) nf);
        ggml_backend_tensor_get(fused, vf.data(), 0, (size_t) nf * sizeof(float));
        ggml_backend_tensor_get(ref,   vr.data(), 0, (size_t) nf * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        // CPU reference in double. Accumulate with ic outermost to match the
        // shader's CRS = ic*(KH*KW) + kh*KW + kw ordering.
        std::vector<double> vc((size_t) nf, 0.0);
        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    double acc = 0.0;
                    for (int64_t ic = 0; ic < IC; ++ic) {
                        for (int64_t kh = 0; kh < KH; ++kh) {
                            const int64_t ih = oh * s - p1 + kh;
                            if (ih < 0 || ih >= vh) continue;
                            for (int64_t kw = 0; kw < KW; ++kw) {
                                const int64_t iw = ow * s - p0 + kw;
                                if (iw < 0 || iw >= vw) continue;
                                acc += (double) ha[(size_t) (kw + kh * KW + ic * KW * KH + oc * KW * KH * IC)]
                                     * (double) hb[(size_t) (iw + ih * vw + ic * vw * vh)];
                            }
                        }
                    }
                    vc[(size_t) (ow + oh * OW + oc * OW * OH)] = acc;
                }
            }
        }

        struct err_t { double max_abs, rel, mean_abs; size_t bad; };
        auto err_vs = [&](const std::vector<float> & v) {
            err_t e{0.0, 0.0, 0.0, 0};
            double scale = 0.0;
            for (int64_t i = 0; i < nf; ++i) {
                const double x = v[(size_t) i];
                if (!std::isfinite(x)) ++e.bad;
                scale = std::max(scale, std::fabs(vc[(size_t) i]));
            }
            for (int64_t i = 0; i < nf; ++i) {
                const double d = std::fabs(v[(size_t) i] - vc[(size_t) i]);
                e.max_abs = std::max(e.max_abs, d);
                e.mean_abs += d;
            }
            e.mean_abs /= (double) nf;
            e.rel = scale > 0 ? e.max_abs / scale : 0.0;
            return e;
        };
        const err_t ef = err_vs(vf), er = err_vs(vr);

        double d_fr = 0.0;
        for (int64_t i = 0; i < nf; ++i) {
            d_fr = std::max(d_fr, std::fabs((double) vf[(size_t) i] - (double) vr[(size_t) i]));
        }

        char m[400];
        snprintf(m, sizeof(m),
                 "fused vs cpu: max=%.3e rel=%.3e | im2col vs cpu: max=%.3e rel=%.3e | fused vs im2col: max=%.3e"
                 "  [OC=%lld CRS=%lld NPQ=%lld nonfinite=%zu/%zu]",
                 ef.max_abs, ef.rel, er.max_abs, er.rel, d_fr,
                 (long long) OC, (long long) (IC * KW * KH), (long long) (OW * OH),
                 ef.bad, er.bad);
        return std::string((ef.bad || er.bad) ? "FAIL: non-finite  " : "ok   ") + m;
    }
};

struct case_conv2d_dw : bench_case {
    int64_t IW, IH, C, KW, KH, s;
    case_conv2d_dw(int64_t IW, int64_t IH, int64_t C, int64_t KW, int64_t KH, int64_t s)
        : IW(IW), IH(IH), C(C), KW(KW), KH(KH), s(s) {}
    const char * op() const override { return "conv2d_dw"; }
    bool composite() const override { return true; }
    std::string vars() const override {
        char b[160];
        snprintf(b, sizeof(b), "C=%lld H=%lld W=%lld K=%lldx%lld s=%lld",
                 (long long) C, (long long) IH, (long long) IW, (long long) KW, (long long) KH, (long long) s);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", KW, KH, 1, C);
        ggml_tensor * b = nt(ctx, GGML_TYPE_F32, "b", IW, IH, C, 1);
        ggml_tensor * out = ggml_conv_2d_dw(ctx, a, b, (int) s, (int) s, (int) (KW / 2), (int) (KH / 2), 1, 1);
        ggml_set_name(out, "out");
        return out;
    }
};

struct case_conv1d : bench_case {
    // x [T, Cin] -> kernel [K, Cin, Cout]; ggml composite ggml_conv_1d
    int64_t T, Cin, Cout, K, s;
    case_conv1d(int64_t T, int64_t Cin, int64_t Cout, int64_t K, int64_t s)
        : T(T), Cin(Cin), Cout(Cout), K(K), s(s) {}
    const char * op() const override { return "conv_1d"; }
    bool composite() const override { return true; }
    std::string vars() const override {
        char b[160];
        snprintf(b, sizeof(b), "T=%lld Cin=%lld Cout=%lld K=%lld s=%lld", (long long) T, (long long) Cin, (long long) Cout, (long long) K, (long long) s);
        return b;
    }
    uint64_t flops(ggml_tensor * out) const override {
        (void) out;
        const int64_t T_out = (T + 2 * (K / 2) - K) / s + 1;
        return (uint64_t) 2 * T_out * Cin * K * Cout;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // ggml: a (kernel) [K, Cin, Cout], b (data) [T, Cin]
        ggml_tensor * a = nt(ctx, GGML_TYPE_F32, "a", K, Cin, Cout);
        ggml_tensor * b = nt(ctx, GGML_TYPE_F32, "b", T, Cin, 1);
        ggml_tensor * out = ggml_conv_1d(ctx, a, b, (int) s, (int) (K / 2), 1);
        ggml_set_name(out, "out");
        return out;
    }
};

// generic im2col->reshape->matmul conv1d, works on BOTH trees. Mirrors the
// generic ggml_im2col path (not the fork's im2col_rafa specialization).
struct case_conv1d_im2col : bench_case {
    int64_t T, Cin, Cout, K, s;
    case_conv1d_im2col(int64_t T, int64_t Cin, int64_t Cout, int64_t K, int64_t s)
        : T(T), Cin(Cin), Cout(Cout), K(K), s(s) {}
    const char * op() const override { return "conv1d_im2col"; }
    bool composite() const override { return true; }
    std::string vars() const override {
        char b[160];
        snprintf(b, sizeof(b), "T=%lld Cin=%lld Cout=%lld K=%lld s=%lld", (long long) T, (long long) Cin, (long long) Cout, (long long) K, (long long) s);
        return b;
    }
    uint64_t flops(ggml_tensor * out) const override {
        (void) out;
        const int64_t T_out = (T + 2 * (K / 2) - K) / s + 1;
        return (uint64_t) 2 * T_out * Cin * K * Cout;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // ggml 1D im2col contract: x:[T, IC, 1, 1], k:[K, IC, 1, 1]
        // -> [T_out, K*IC, 1, 1]; then [T_out, K*IC] x w[K*IC, Cout]
        ggml_tensor * x = nt(ctx, GGML_TYPE_F32, "x", T, Cin, 1, 1);
        ggml_tensor * k = nt(ctx, GGML_TYPE_F32, "k", K, Cin, 1, 1);
        ggml_tensor * w = nt(ctx, GGML_TYPE_F32, "w", K * Cin, Cout, 1, 1);
        ggml_tensor * cols = ggml_im2col(ctx, k, x, (int) s, 0, (int) (K / 2), 0, 1, 0, false, GGML_TYPE_F32); // [T_out, K*Cin]
        ggml_tensor * out = ggml_mul_mat(ctx, w, cols); // [Cout, T_out]
        ggml_set_name(out, "out");
        return out;
    }
};

#ifdef BENCH_FORK
// Fork-only: conv1d via im2col_rafa (the path src/ops.cpp actually uses).
struct case_conv1d_rafa : bench_case {
    int64_t T, Cin, Cout, K, s;
    case_conv1d_rafa(int64_t T, int64_t Cin, int64_t Cout, int64_t K, int64_t s)
        : T(T), Cin(Cin), Cout(Cout), K(K), s(s) {}
    const char * op() const override { return "conv1d_rafa"; }
    bool composite() const override { return true; }
    std::string vars() const override {
        char b[160];
        snprintf(b, sizeof(b), "T=%lld Cin=%lld Cout=%lld K=%lld s=%lld", (long long) T, (long long) Cin, (long long) Cout, (long long) K, (long long) s);
        return b;
    }
    uint64_t flops(ggml_tensor * out) const override {
        (void) out;
        const int64_t T_out = (T + 2 * (K / 2) - K) / s + 1;
        return (uint64_t) 2 * T_out * Cin * K * Cout;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = nt(ctx, GGML_TYPE_F32, "x", Cin, T);   // im2col_rafa wants [C, T_in]
        ggml_tensor * w = nt(ctx, GGML_TYPE_F32, "w", K * Cin, Cout, 1, 1);
        ggml_tensor * cols = ggml_im2col_rafa(ctx, x, (int) K, (int) s, (int) (K / 2), 1, GGML_TYPE_F32);
        ggml_tensor * cols2 = ggml_reshape_2d(ctx, cols, cols->ne[0], cols->ne[1] * cols->ne[2] * cols->ne[3]);
        ggml_tensor * out = ggml_mul_mat(ctx, w, cols2);
        ggml_set_name(out, "out");
        return out;
    }
};

// Fork-only: the im2col_rafa node alone (our 1D conv im2col).
struct case_im2col_rafa : bench_case {
    int64_t C, T, K, s;
    case_im2col_rafa(int64_t C, int64_t T, int64_t K, int64_t s) : C(C), T(T), K(K), s(s) {}
    const char * op() const override { return "im2col_rafa"; }
    std::string vars() const override {
        char b[128];
        snprintf(b, sizeof(b), "C=%lld T=%lld K=%lld s=%lld", (long long) C, (long long) T, (long long) K, (long long) s);
        return b;
    }
    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = nt(ctx, GGML_TYPE_F32, "x", C, T);
        ggml_tensor * out = ggml_im2col_rafa(ctx, x, (int) K, (int) s, (int) (K / 2), 1, GGML_TYPE_F32);
        ggml_set_name(out, "out");
        return out;
    }
};
#endif

// ---------------------------------------------------------------------------
// case registry
// ---------------------------------------------------------------------------

static std::vector<case_ptr> make_cases() {
    std::vector<case_ptr> v;
    auto add = [&](bench_case * c) { v.emplace_back(c); };

    // --- matmul ---
    add(new case_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 1280, 3840, 512));        // GPT2 c_attn
    add(new case_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 1280, 3840, 512));
    add(new case_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 512 * 511, 512, 512));    // Conformer big-K
    add(new case_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 2560, 2560, 512, 1));     // single-token proj
    add(new case_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128, 128, 4096));         // bmm-ish

    // --- attention ---
    add(new case_flash_attn(128, 32, 8, 512));
    add(new case_flash_attn(128, 32, 8, 16));

    // --- conv (composite) ---
    add(new case_conv2d(64, 64, 1, 512, 3, 3, 2));      // Conformer embed-ish
    // --- conv2d_direct (single implicit-GEMM node) vs the im2col composite ---
    // Paired (OC, K, spatial) sweep: OC picks the backend tile (K>64 -> 128x128,
    // K<=32 -> 32x256, else 64x32), which is what decides whether the coopmat cm1
    // path applies at all, so it is the axis that matters for routing. OC=100 is
    // not a multiple of the 64-row K tile, so it also covers the partial-K store.
    // Read with `tools/bench_ops_compare.py --pair`.
    //
    // The im2col reference needs OH*OW*IC*KH*KW*4 bytes resident, so 9x9 only
    // pairs up to ~48x232; at det's real 464x96 the 9x9 reference is 3.7 GB and
    // cannot be allocated at all (which is the reason the fused path exists).
    // det's target shape, paired only where the reference fits:
    add(new case_conv2d_direct(464, 96, 256, 64, 3, 3, 1));
    add(new case_conv2d(464, 96, 256, 64, 3, 3, 1));
    for (int64_t oc : { (int64_t) 32, (int64_t) 64, (int64_t) 100, (int64_t) 128, (int64_t) 256 }) {
        add(new case_conv2d_direct(232, 48, 256, oc, 9, 9, 1));
        add(new case_conv2d(232, 48, 256, oc, 9, 9, 1));
        add(new case_conv2d_direct(464, 96, 256, oc, 3, 3, 1));
        add(new case_conv2d(464, 96, 256, oc, 3, 3, 1));
    }
    // spatial dependence at the two tile regimes that matter (64x32 vs 128x128)
    for (int64_t oc : { (int64_t) 64, (int64_t) 128 }) {
        for (int64_t hw : { (int64_t) 116, (int64_t) 232 }) {
            add(new case_conv2d_direct(hw, 24, 256, oc, 3, 3, 1));
            add(new case_conv2d(hw, 24, 256, oc, 3, 3, 1));
        }
    }
    // rec's regime: 3x3 on a short, wide map (W picked so the unchunked im2col
    // reference still fits -- one conv2d_tiled chunk). Maps the (IC, OC) crossover
    // where the fused path stops paying.
    for (int64_t oc : { (int64_t) 128, (int64_t) 256, (int64_t) 512, (int64_t) 768 }) {
        add(new case_conv2d_direct(800, 48, 512, oc, 3, 3, 1));
        add(new case_conv2d(800, 48, 512, oc, 3, 3, 1));
    }
    for (int64_t ic : { (int64_t) 256, (int64_t) 512 }) {
        for (int64_t oc : { (int64_t) 64, (int64_t) 512 }) {
            add(new case_conv2d_direct(800, 48, ic, oc, 3, 3, 1));
            add(new case_conv2d(800, 48, ic, oc, 3, 3, 1));
        }
    }
    // 1x1 (pointwise) at the backbone's channel counts: both det and rec carry
    // 512-896 channel pointwise convs, and K=1 makes CRS=IC, i.e. the im2col
    // reference is at its cheapest here.
    for (auto s : { std::make_tuple((int64_t) 928, (int64_t) 192, (int64_t) 512, (int64_t) 896),
                    std::make_tuple((int64_t) 464, (int64_t) 96, (int64_t) 896, (int64_t) 896),
                    std::make_tuple((int64_t) 800, (int64_t) 48, (int64_t) 768, (int64_t) 768),
                    std::make_tuple((int64_t) 800, (int64_t) 48, (int64_t) 512, (int64_t) 768) }) {
        const int64_t w = std::get<0>(s), h = std::get<1>(s), ic = std::get<2>(s), oc = std::get<3>(s);
        add(new case_conv2d_direct(w, h, ic, oc, 1, 1, 1));
        add(new case_conv2d(w, h, ic, oc, 1, 1, 1));
    }

    add(new case_conv2d(64, 64, 128, 128, 3, 3, 1));    // OCR LCNet-ish
    add(new case_conv2d(64, 64, 256, 256, 7, 7, 1));    // det neck strip conv (big kernel)
    add(new case_conv2d(64, 64, 256, 256, 1, 1, 1));    // LCNetV4 pointwise (K=1)
    add(new case_conv2d_dw(64, 64, 128, 3, 3, 2));      // OCR token_conv
    add(new case_conv1d(256, 1536, 768, 8, 1));         // BigVGAN up sample
    add(new case_conv1d_im2col(256, 1536, 768, 8, 1));  // same shape, im2col->matmul
    add(new case_conv1d(512, 1024, 1024, 31, 1));       // large K head-to-head
    add(new case_conv1d_im2col(512, 1024, 1024, 31, 1));
#ifdef BENCH_FORK
    add(new case_conv1d_rafa(256, 1536, 768, 8, 1));    // fork path (A only)
    add(new case_conv1d_rafa(512, 1024, 1024, 31, 1));  // W2vBert-like large K
    add(new case_im2col_rafa(1536, 256, 8, 1));         // the im2col_rafa node alone
    add(new case_im2col_rafa(1024, 512, 31, 1));        // W2vBert-like K=31
    add(new case_im2col_rafa(1536, 512, 8, 2));         // strided (upsample down path)
#endif

    // --- conv_transpose_2d (OCR DB head) / col2im_1d (DacDecoder) ---
    add(new case_conv_transpose_2d(64, 64, 32, 32, 3, 3, 2));
    // det's DB head `head.conv_up` (PT weight [64, 64, 2, 2]): the one transposed conv
    // in the ported models on a tile cm1 covers (OC=64 -> 64x32). `head.conv_final` is
    // [64, 1, 2, 2] -> OC=1 -> 32x256 tile, which cm1 cannot use.
    add(new case_conv_transpose_2d(64, 64, 96, 464, 2, 2, 2));
    // det's DB head `head.conv_final` (PT weight [64, 1, 2, 2]): OC=1 -> 32x256 tile,
    // so this one does NOT get cm1.
    add(new case_conv_transpose_2d(64, 1, 192, 928, 2, 2, 2));
    add(new case_col2im_1d(16, 32, 128, 8));            // K=16, OC=32, T_in=128, s=8

    // --- pooling / sampling / pad ---
    add(new case_pool_2d(GGML_OP_POOL_AVG, 64, 64, 128, 2, 2));
    add(new case_pool_2d(GGML_OP_POOL_MAX, 64, 64, 128, 2, 2));
    add(new case_upscale(GGML_SCALE_MODE_NEAREST, 64, 64, 128, 2));
    add(new case_upscale(GGML_SCALE_MODE_BILINEAR, 64, 64, 128, 2));
    add(new case_pad(256, 256, 32, 1, 1));

    // --- im2col (OCR-relevant shapes) ---
    add(new case_im2col(false, 256, 1, 256, 3));          // 1D (K=3, IC=256)
    add(new case_im2col(true, 64, 64, 128, 3, 3));        // 2D 3x3
    add(new case_im2col(true, 64, 64, 256, 7, 7));        // det neck strip conv (K=7)
    add(new case_im2col(true, 64, 64, 256, 5, 5));        // det neck strip conv (K=5)
    add(new case_im2col(true, 64, 64, 256, 1, 1));        // LCNetV4 pointwise (K=1)

    // --- norm / activation ---
    add(new case_soft_max(18710, 128, false));          // rec logits
    add(new case_soft_max(512, 512, true));             // attention scores
    add(new case_rms_norm(2560, 128));
    add(new case_group_norm(256, 512, 32));
    add(new case_unary(case_unary::SILU, 2560, 128));
    add(new case_unary(case_unary::GELU, 384, 256));
    add(new case_scale(2560, 128, 1));
    add(new case_rope(128, 512, 32));
    add(new case_concat(672, 192, 128));

    // --- indexing ---
    add(new case_get_rows(1280, 512));
    add(new case_set_rows(128, 512, 2048));

    return v;
}

// ---------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------

struct result_t {
    std::string op, vars;
    bool supported = false;
    bool composite = false;
    double avg_us = 0;
    double gflops = 0;
    double gbps = 0;
};

static bool run_case(ggml_backend_t backend, bench_case * c, result_t & r) {
    static const size_t graph_nodes = 16384;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(graph_nodes, false),
        /* .mem_base   = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * out = c->build_graph(ctx);
    r.op = c->op();
    r.vars = c->vars();
    r.composite = c->composite();

    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        r.supported = false;
        return true;
    }
    r.supported = true;

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    c->initialize(ctx);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
    ggml_build_forward_expand(gf, out);

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph_compute failed (%s)\n", c->op(), ggml_status_to_string(st));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    const uint64_t fl = c->flops(out);

    // Capture the subgraph this op expanded into (in build order). For fused
    // nodes this is just `out`; for composites (conv_1d/conv_2d/...) it is the
    // whole im2col->reshape->matmul(- > permute/cont) chain, so timing it must
    // recompute the entire chain, not just the last node.
    const int sub_n = ggml_graph_n_nodes(gf);
    std::vector<ggml_tensor *> sub_nodes(sub_n);
    for (int i = 0; i < sub_n; ++i) sub_nodes[i] = ggml_graph_node(gf, i);

    // op_size: output + sources + internal subgraph tensors (view/input nodes
    // excluded), mirroring how the number is reported upstream.
    size_t sz = ggml_nbytes(out);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (out->src[i]) sz += ggml_nbytes(out->src[i]);
    }
    for (int k = 0; k < sub_n; ++k) {
        ggml_tensor * t = sub_nodes[k];
        bool is_input = (t->op == GGML_OP_NONE && t != out);
        if (!is_input && t != out) sz += ggml_nbytes(t);
    }

    int64_t n_per_dose;
    if (fl > 0) {
        const uint64_t target = 100ull * 1000 * 1000 * 1000;  // 100 GFLOP
        n_per_dose = (int64_t) (target / fl);
    } else {
        const size_t target = 32ull << 30;  // 32 GiB
        n_per_dose = (int64_t) (target / (sz ? sz : 1));
    }
    const int64_t cap_doses = ((int64_t) ggml_graph_size(gf) - sub_n) / (sub_n > 0 ? sub_n : 1);

    int n_runs = (int) std::min<int64_t>(std::max<int64_t>(cap_doses, 0), n_per_dose) + 1;
    if (n_runs < 1) n_runs = 1;

    for (int i = 1; i < n_runs; ++i) {
        for (int k = 0; k < sub_n; ++k) ggml_graph_add_node(gf, sub_nodes[k]);
    }

    int64_t total_us = 0;
    int total_runs = 0;
    do {
        int64_t t0 = ggml_time_us();
        ggml_status s = ggml_backend_graph_compute(backend, gf);
        int64_t t1 = ggml_time_us();
        if (s != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: graph_compute failed (%s)\n", c->op(), ggml_status_to_string(s));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }
        total_us += t1 - t0;
        total_runs += n_runs;
    } while (total_us < 1000 * 1000);

    r.avg_us = (double) total_us / total_runs;
    r.gflops = fl > 0 ? (double) fl * total_runs / (total_us / 1e6) / 1e9 : 0.0;
    r.gbps   = fl == 0 ? (double) (sz * (size_t) total_runs) / (total_us / 1e6) / 1e9 : 0.0;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static void usage(const char * argv0) {
    printf("usage: %s [--filter SUBSTR] [--list] [--verify]\n", argv0);
    printf("  --verify  run each case's numeric cross-check instead of timing it\n");
}

int main(int argc, char ** argv) {
    const char * filter = nullptr;
    bool list = false;
    bool check = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) filter = argv[++i];
        else if (strcmp(argv[i], "--list") == 0) list = true;
        else if (strcmp(argv[i], "--verify") == 0) check = true;
        else { usage(argv[0]); return 1; }
    }

    auto cases = make_cases();
    if (list) {
        for (auto & c : cases) printf("%-20s %s\n", c->op(), c->vars().c_str());
        return 0;
    }

    ggml_backend_load_all();

    // pick a GPU device (override with BENCH_DEV name substring)
    const char * want = getenv("BENCH_DEV");
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        if (!want) { dev = d; break; }
        std::string nm = ggml_backend_dev_name(d);
        if (nm.find(want) != std::string::npos) { dev = d; break; }
    }
    if (!dev) { fprintf(stderr, "no GPU backend device found\n"); return 1; }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "ggml_backend_dev_init failed\n"); return 1; }

    printf("# variant=%s\n", BENCH_VARIANT);
    printf("# device=%s | %s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    if (check) {
        for (auto & c : cases) {
            if (filter && c->op() != std::string(filter) && std::string(c->op()).find(filter) == std::string::npos) continue;
            const std::string r = c->verify(backend);
            printf("%-22s %-52s %s\n", c->op(), c->vars().c_str(),
                   r.empty() ? "(no cross-check defined)" : r.c_str());
        }
        ggml_backend_free(backend);
        return 0;
    }

    printf("# %-20s %-52s %12s %11s %10s\n", "op", "vars", "avg_us", "GFLOP/s", "GB/s");

    for (auto & c : cases) {
        if (filter && c->op() != std::string(filter) && std::string(c->op()).find(filter) == std::string::npos) continue;
        result_t r;
        if (!run_case(backend, c.get(), r)) {
            printf("%-22s %-52s %12s\n", r.op.c_str(), r.vars.c_str(), "FAILED");
            continue;
        }
        if (!r.supported) {
            printf("%-22s %-52s %12s\n", r.op.c_str(), r.vars.c_str(), "unsupported");
            continue;
        }
        printf("%-22s %-52s %12.2f %11.2f %10.1f%s\n",
               r.op.c_str(), r.vars.c_str(), r.avg_us, r.gflops, r.gbps,
               r.composite ? "  (composite)" : "");
    }

    ggml_backend_free(backend);
    return 0;
}
