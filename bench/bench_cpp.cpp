// vulkantorch benchmark (C++) — mirrored by bench/bench_py.py and
// example/CSharp/VulkanTorch.Bench. Weights live in GPU Memory; each rep builds
// one graph, runs the op, and reads back only a tiny reduction (sum_rows), so the
// timing reflects compute rather than host<->device transfer. Best of N.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <tuple>
#include <vector>

#include "graph.h"
#include "memory.h"
#include "ops.h"
#include "runtime.h"
#include "tensor.h"

using namespace vt;

static double now_us() {
    using namespace std::chrono;
    return static_cast<double>(duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count());
}

static const int REPS = 10;

int main() {
    Runtime rt;
    Device gpu = rt.gpu();
    std::printf("backend: %s\n", rt.name());

    std::mt19937 gen(0);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::printf("\n%-22s | %11s\n", "matmul shape", "ms");
    std::printf("%s\n", std::string(38, '-').c_str());
    for (auto [M, K, N] : {std::tuple{1024, 1024, 1024}, std::tuple{2048, 2048, 2048},
                           std::tuple{4096, 4096, 4096}}) {
        std::vector<float> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N);
        for (auto& v : A) v = nd(gen);
        for (auto& v : B) v = nd(gen);
        Memory ma(gpu, A.size() * 4), mb(gpu, B.size() * 4);
        Tensor ta = ma.tensor({M, K}, A.data(), A.size() * 4);
        Tensor tb = mb.tensor({K, N}, B.data(), B.size() * 4);

        double best = 1e18;
        for (int r = 0; r < REPS; ++r) {
            double t0 = now_us();
            Graph g(rt, gpu);
            g.enter();
            Tensor y = matmul(ta, tb);
            Tensor s = sum_rows(y);
            s.mark_output();
            g.exit();
            s.to_host_bytes();
            best = std::min(best, (now_us() - t0) / 1000.0);
        }
        char label[64];
        std::snprintf(label, sizeof(label), "%dx%dx%d", M, K, N);
        std::printf("%-22s | %8.2f ms\n", label, best);
    }

    std::printf("\n%-22s | %11s\n", "conv1d T Cin->Cout", "ms");
    std::printf("%s\n", std::string(38, '-').c_str());
    for (auto [T, Cin, Cout, K, pad] : {std::tuple{8192, 256, 256, 7, 3},
                                        std::tuple{16384, 512, 512, 7, 3}}) {
        std::vector<float> x(static_cast<size_t>(T) * Cin), w(static_cast<size_t>(Cout) * Cin * K);
        for (auto& v : x) v = nd(gen);
        for (auto& v : w) v = nd(gen);
        Memory mx(gpu, x.size() * 4), mw(gpu, w.size() * 4);
        Tensor tx = mx.tensor({T, Cin}, x.data(), x.size() * 4);
        Tensor tw = mw.tensor({Cout, Cin, K}, w.data(), w.size() * 4);

        double best = 1e18;
        for (int r = 0; r < REPS; ++r) {
            double t0 = now_us();
            Graph g(rt, gpu);
            g.enter();
            Tensor y = conv1d(tx, tw, 1, pad, 1);
            Tensor s = sum_rows(y);
            s.mark_output();
            g.exit();
            s.to_host_bytes();
            best = std::min(best, (now_us() - t0) / 1000.0);
        }
        char label[64];
        std::snprintf(label, sizeof(label), "%d %d->%d K%d", T, Cin, Cout, K);
        std::printf("%-22s | %8.2f ms\n", label, best);
    }
    return 0;
}
