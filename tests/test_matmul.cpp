#include "graph.h"
#include "ops.h"
#include "runtime.h"
#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace vt;

static std::vector<double> matmul_ref(const std::vector<float>& A, const std::vector<float>& B,
                                      int M, int K, int N) {
    std::vector<double> C(static_cast<std::size_t>(M) * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += double(A[m * K + k]) * B[k * N + n];
            C[std::size_t(m) * N + n] = s;
        }
    return C;
}

static double max_abs_err(const std::vector<double>& ref, const std::vector<float>& got) {
    double e = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) e = std::max(e, std::fabs(ref[i] - got[i]));
    return e;
}

static int run() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Runtime rt;
    std::printf("backend: %s\n\n", rt.name());

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int fails = 0;

    // ── matmul across shapes ────────────────────────────────────────────────
    struct Case { int M, K, N; };
    for (const Case& c : std::vector<Case>{{2, 3, 4}, {16, 16, 16},
                                           {64, 128, 32}, {128, 256, 64}, {5, 7, 3}}) {
        std::vector<float> A(std::size_t(c.M) * c.K), B(std::size_t(c.K) * c.N);
        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);

        std::vector<float> got;
        {
            Graph g(rt.backend(), rt.sched());
            Graph::Scope scope(&g);
            auto a = g.input({c.M, c.K}, A.data(), A.size() * sizeof(float));
            auto b = g.input({c.K, c.N}, B.data(), B.size() * sizeof(float));
            got = matmul(a, b).to_host();
        }
        const double err = max_abs_err(matmul_ref(A, B, c.M, c.K, c.N), got);
        const bool ok = err < 1e-4 * c.K;
        std::printf("[%s] matmul  M=%d K=%d N=%d  max_err=%.3g\n",
                    ok ? " OK " : "FAIL", c.M, c.K, c.N, err);
        if (!ok) ++fails;
    }

    // ── captured 2-op graph: (A @ B) + C ────────────────────────────────────
    {
        const int M = 8, K = 6, N = 5;
        std::vector<float> A(M * K), B(K * N), C(M * N);
        for (float& x : A) x = dist(rng);
        for (float& x : B) x = dist(rng);
        for (float& x : C) x = dist(rng);

        std::vector<float> got;
        {
            Graph g(rt.backend(), rt.sched());
            Graph::Scope scope(&g);
            auto a = g.input({M, K}, A.data(), A.size() * sizeof(float));
            auto b = g.input({K, N}, B.data(), B.size() * sizeof(float));
            auto c = g.input({M, N}, C.data(), C.size() * sizeof(float));
            got = add(matmul(a, b), c).to_host();
        }
        const auto ref = matmul_ref(A, B, M, K, N);
        double err = 0.0;
        for (int i = 0; i < M * N; ++i) err = std::max(err, std::fabs(ref[i] + C[i] - got[i]));
        const bool ok = err < 1e-4 * K;
        std::printf("[%s] (A@B)+C              max_err=%.3g\n", ok ? " OK " : "FAIL", err);
        if (!ok) ++fails;
    }

    // ── views: transpose + contiguous ───────────────────────────────────────
    {
        const int R = 4, Cc = 3;
        std::vector<float> X(R * Cc);
        for (float& x : X) x = dist(rng);

        std::vector<float> got;
        {
            Graph g(rt.backend(), rt.sched());
            Graph::Scope scope(&g);
            auto x = g.input({R, Cc}, X.data(), X.size() * sizeof(float));
            got = contiguous(transpose(x)).to_host();
        }
        double err = 0.0;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < Cc; ++c) err = std::max(err, std::fabs(double(X[r * Cc + c]) - got[c * R + r]));
        const bool ok = err == 0.0;
        std::printf("[%s] transpose+contig     max_err=%.3g\n", ok ? " OK " : "FAIL", err);
        if (!ok) ++fails;
    }

    // ── reshape smoke ───────────────────────────────────────────────────────
    {
        std::vector<float> X(12);
        for (float& x : X) x = dist(rng);
        std::vector<float> got;
        std::vector<int64_t> shape_out;
        {
            Graph g(rt.backend(), rt.sched());
            Graph::Scope scope(&g);
            auto x = g.input({3, 4}, X.data(), X.size() * sizeof(float));
            auto r = reshape(x, {2, 2, 3});
            shape_out = r.shape();
            got = r.to_host();
        }
        bool ok = (shape_out == std::vector<int64_t>{2, 2, 3}) && got == X;
        std::printf("[%s] reshape [3,4]->[2,2,3]\n", ok ? " OK " : "FAIL");
        if (!ok) ++fails;
    }

    std::printf("\n%s\n", fails ? "some cases FAILED" : "all cases passed");
    return fails ? 1 : 0;
}

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
}
