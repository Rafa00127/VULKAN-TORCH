// GGUF re-quantizer for vulkan-torch.
//
// Re-quantizes every eligible weight tensor in a GGUF to a target ggml type,
// preserving metadata and copying non-eligible tensors (norms, biases, 1-D
// tensors, pre-fused conv kernels) untouched.
//
//   quantize in.gguf out.gguf <type>
//
//   <type> = q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k | f16
//
// Block types (q*) only rewrite 2-D weight tensors; everything else is copied.
// f16 downcasts weight tensors (2-D or higher, non-norm, non-pre-fused-conv) to
// F16 and leaves 1-D tensors (norms, biases, snake-alpha), scalars and pre-fused
// conv kernels untouched - the runtime reads those as F32. Mirrors ggml's
// GGML_FTYPE_MOSTLY_F16.
//
// Adapted from NovelReader/HiggsTTS.cpp tools/higgs_quantize, trimmed to the
// generic path (the per-architecture skip tables are not carried over).

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

// On Windows argv paths arrive as UTF-16 (via wmain) and the filesystem is
// UTF-16, but ggml's gguf loader and our own strings are UTF-8. These helpers
// bridge the two; vt_fopen opens a UTF-8 path through _wfopen so non-ASCII
// paths (e.g. a model directory whose name is non-ASCII) resolve.
#ifdef _WIN32
static std::string to_utf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}
static std::wstring to_wide(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
#endif

static FILE* vt_fopen(const std::string& path, const char* mode) {
#ifdef _WIN32
    std::wstring wmode;
    for (const char* p = mode; *p != '\0'; ++p) wmode.push_back((wchar_t)*p);
    return _wfopen(to_wide(path).c_str(), wmode.c_str());
#else
    return fopen(path.c_str(), mode);
#endif
}

// Conv kernels a converter already materialized into a 2-D matrix - e.g. the
// Higgs ConvTranspose1d wperm, written as [K*OC, IC] and consumed by a custom
// decoder instead of ggml_conv_*. They are not plain matmul weights, so block
// quantization wrecks the upsampling path. Detect them by name.
static bool is_prefused_conv(const std::string& n) {
    static const char* markers[] = {"conv_t", "convtranspose", "conv_transpose", "wperm", "w_perm"};
    for (const char* m : markers) {
        if (n.find(m) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Smaller-block type to fall back to when ne[0] is not a multiple of the
// target's block size (K-quants use 256; Q4_0/Q5_0/Q8_0 use 32).
static ggml_type block_fallback(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K: return GGML_TYPE_Q4_0;
        case GGML_TYPE_Q5_K: return GGML_TYPE_Q5_0;
        case GGML_TYPE_Q6_K: return GGML_TYPE_Q8_0;
        default:             return GGML_TYPE_COUNT;
    }
}

static bool parse_type(const std::string& s, ggml_type& qtype, ggml_ftype& ftype, bool& to_f16) {
    to_f16 = false;
    if (s == "f16") {
        qtype = GGML_TYPE_F16;
        ftype = GGML_FTYPE_MOSTLY_F16;
        to_f16 = true;
        return true;
    }
    struct Entry { const char* name; ggml_type qt; ggml_ftype ft; };
    static const Entry table[] = {
        {"q4_0", GGML_TYPE_Q4_0, GGML_FTYPE_MOSTLY_Q4_0},
        {"q4_1", GGML_TYPE_Q4_1, GGML_FTYPE_MOSTLY_Q4_1},
        {"q5_0", GGML_TYPE_Q5_0, GGML_FTYPE_MOSTLY_Q5_0},
        {"q5_1", GGML_TYPE_Q5_1, GGML_FTYPE_MOSTLY_Q5_1},
        {"q8_0", GGML_TYPE_Q8_0, GGML_FTYPE_MOSTLY_Q8_0},
        {"q2_k", GGML_TYPE_Q2_K, GGML_FTYPE_MOSTLY_Q2_K},
        {"q3_k", GGML_TYPE_Q3_K, GGML_FTYPE_MOSTLY_Q3_K},
        {"q4_k", GGML_TYPE_Q4_K, GGML_FTYPE_MOSTLY_Q4_K},
        {"q5_k", GGML_TYPE_Q5_K, GGML_FTYPE_MOSTLY_Q5_K},
        {"q6_k", GGML_TYPE_Q6_K, GGML_FTYPE_MOSTLY_Q6_K},
    };
    for (const Entry& e : table) {
        if (s == e.name) {
            qtype = e.qt;
            ftype = e.ft;
            return true;
        }
    }
    return false;
}

static void print_types(FILE* fp) {
    fprintf(fp, "  type = q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k f16\n");
}

static bool requantize(const std::string& in_path, const std::string& out_path,
                       ggml_type qtype, ggml_ftype ftype, bool to_f16) {
    printf("quantize: loading '%s'\n", in_path.c_str());

    ggml_context* ctx_in_ggml = nullptr;
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = &ctx_in_ggml;
    gguf_context* ctx_in = gguf_init_from_file(in_path.c_str(), params);
    if (ctx_in == nullptr || ctx_in_ggml == nullptr) {
        fprintf(stderr, "quantize: failed to load '%s'\n", in_path.c_str());
        return false;
    }

    gguf_context* ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", ftype);

    const int n_tensors = (int)gguf_get_n_tensors(ctx_in);
    std::vector<ggml_type> target((size_t)n_tensors, GGML_TYPE_COUNT);

    // First pass: decide each tensor's output type and register its descriptor,
    // so gguf_add_tensor computes correct (quantized) offsets up front.
    ggml_init_params sp = {ggml_tensor_overhead() * (size_t)n_tensors + 1024, nullptr, true};
    ggml_context* ctx_scratch = ggml_init(sp);

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx_in, i);
        ggml_tensor* t = ggml_get_tensor(ctx_in_ggml, name);
        const std::string sname(name);

        ggml_type dst = t->type;  // default: copy verbatim
        const bool src_float = (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
        // A plain weight matrix a converter wrote: not a norm, not a pre-fused
        // conv kernel (which the runtime reads as F32).
        const bool is_special = sname.find("norm") != std::string::npos || is_prefused_conv(sname);

        if (to_f16) {
            // Downcast weights (2-D or higher) to F16; leave 1-D norms/biases,
            // scalars and pre-fused conv kernels in their original type.
            if (t->type == GGML_TYPE_F32 && ggml_n_dims(t) >= 2 && !is_special) {
                dst = GGML_TYPE_F16;
            }
        } else if (src_float && ggml_n_dims(t) == 2 && !is_special) {
            ggml_type qt = qtype;
            if (t->ne[0] % ggml_blck_size(qt) != 0) {
                const ggml_type fb = block_fallback(qtype);
                qt = (fb != GGML_TYPE_COUNT && t->ne[0] % ggml_blck_size(fb) == 0) ? fb : GGML_TYPE_COUNT;
            }
            if (qt != GGML_TYPE_COUNT) {
                dst = qt;
            }
        }
        target[i] = dst;

        if (dst != t->type) {
            ggml_tensor* t_out = ggml_new_tensor(ctx_scratch, dst, ggml_n_dims(t), t->ne);
            ggml_set_name(t_out, name);
            gguf_add_tensor(ctx_out, t_out);
        } else {
            gguf_add_tensor(ctx_out, t);
        }
    }

    printf("quantize: writing to '%s'\n", out_path.c_str());
    FILE* fout = vt_fopen(out_path, "w+b");
    if (fout == nullptr) {
        fprintf(stderr, "quantize: failed to open '%s' for writing\n", out_path.c_str());
        gguf_free(ctx_in);
        gguf_free(ctx_out);
        ggml_free(ctx_in_ggml);
        ggml_free(ctx_scratch);
        return false;
    }

    const size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> meta_data(meta_size, 0);
    fwrite(meta_data.data(), 1, meta_size, fout);

    FILE* fin = vt_fopen(in_path, "rb");
    if (fin == nullptr) {
        fprintf(stderr, "quantize: failed to reopen '%s'\n", in_path.c_str());
        fclose(fout);
        return false;
    }
    const size_t data_offset_in = gguf_get_data_offset(ctx_in);

    std::vector<float> f32_data;
    std::vector<ggml_fp16_t> f16_data;
    std::vector<uint8_t> out_data;
    int n_requantized = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx_in, i);
        ggml_tensor* t = ggml_get_tensor(ctx_in_ggml, name);

        const ggml_type src = t->type;
        const ggml_type dst = target[i];
        const size_t src_bytes = ggml_nbytes(t);
        const size_t offset = data_offset_in + gguf_get_tensor_offset(ctx_in, i);

        printf("[%3d/%3d] %-44s %8s -> %-8s ", i + 1, n_tensors, name, ggml_type_name(src),
               ggml_type_name(dst));
        fflush(stdout);

        // 64-bit seek: `long` is 32-bit on Windows even on x86_64.
#ifdef _WIN32
        _fseeki64(fin, (__int64)offset, SEEK_SET);
#else
        fseeko(fin, (off_t)offset, SEEK_SET);
#endif

        if (dst == src) {
            out_data.resize(src_bytes);
            if (fread(out_data.data(), 1, src_bytes, fin) != src_bytes) {
                fprintf(stderr, "failed to read raw data\n");
                return false;
            }
            printf("copy\n");
        } else if (ggml_is_quantized(dst)) {
            const int64_t nelements = ggml_nelements(t);
            f32_data.resize((size_t)nelements);
            if (src == GGML_TYPE_F32) {
                if (fread(f32_data.data(), sizeof(float), (size_t)nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f32 data\n");
                    return false;
                }
            } else {
                f16_data.resize((size_t)nelements);
                if (fread(f16_data.data(), sizeof(ggml_fp16_t), (size_t)nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f16 data\n");
                    return false;
                }
                for (int64_t j = 0; j < nelements; j++) {
                    f32_data[j] = ggml_fp16_to_fp32(f16_data[j]);
                }
            }
            const size_t max_q = ggml_row_size(dst, t->ne[0]) * (size_t)(nelements / t->ne[0]);
            out_data.resize(max_q);
            const size_t q_size = ggml_quantize_chunk(dst, f32_data.data(), out_data.data(), 0,
                                                      nelements / t->ne[0], t->ne[0], nullptr);
            out_data.resize(q_size);
            printf("quant blocksize=%d\n", (int)ggml_blck_size(dst));
            n_requantized++;
        } else {
            // dst == F16, src == F32
            const int64_t nelements = ggml_nelements(t);
            f32_data.resize((size_t)nelements);
            if (fread(f32_data.data(), sizeof(float), (size_t)nelements, fin) != (size_t)nelements) {
                fprintf(stderr, "failed to read f32 data\n");
                return false;
            }
            f16_data.resize((size_t)nelements);
            for (int64_t j = 0; j < nelements; j++) {
                f16_data[j] = ggml_fp32_to_fp16(f32_data[j]);
            }
            out_data.assign(reinterpret_cast<uint8_t*>(f16_data.data()),
                            reinterpret_cast<uint8_t*>(f16_data.data()) + (size_t)nelements * sizeof(ggml_fp16_t));
            printf("F32 -> F16\n");
            n_requantized++;
        }

        fwrite(out_data.data(), 1, out_data.size(), fout);
        const size_t pad = GGML_PAD(out_data.size(), GGUF_DEFAULT_ALIGNMENT) - out_data.size();
        for (size_t j = 0; j < pad; j++) {
            fputc(0, fout);
        }
    }

    printf("quantize: converted %d / %d tensors\n", n_requantized, n_tensors);

    fflush(fout);
    fseek(fout, 0, SEEK_SET);
    gguf_get_meta_data(ctx_out, meta_data.data());
    fwrite(meta_data.data(), 1, meta_size, fout);
    fflush(fout);

    fclose(fin);
    fclose(fout);
    gguf_free(ctx_in);
    gguf_free(ctx_out);
    ggml_free(ctx_in_ggml);
    ggml_free(ctx_scratch);
    return true;
}

static int run(const std::vector<std::string>& args) {
    if (args.size() != 4) {
        fprintf(stderr, "usage: %s in.gguf out.gguf <type>\n", args[0].c_str());
        print_types(stderr);
        return 1;
    }

    ggml_type qtype = GGML_TYPE_COUNT;
    ggml_ftype ftype = GGML_FTYPE_UNKNOWN;
    bool to_f16 = false;
    if (!parse_type(args[3], qtype, ftype, to_f16)) {
        fprintf(stderr, "%s: unknown type '%s'\n", args[0].c_str(), args[3].c_str());
        print_types(stderr);
        return 1;
    }

    if (!requantize(args[1], args[2], qtype, ftype, to_f16)) {
        fprintf(stderr, "quantize: failed\n");
        return 1;
    }
    return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    args.reserve((size_t)argc);
    for (int i = 0; i < argc; i++) {
        args.push_back(to_utf8(argv[i]));
    }
    return run(args);
}
#else
int main(int argc, char** argv) {
    return run(std::vector<std::string>(argv, argv + argc));
}
#endif
