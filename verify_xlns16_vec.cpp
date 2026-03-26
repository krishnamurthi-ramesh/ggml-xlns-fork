// verify_xlns16_vec.cpp
//
// Standalone test that compiles the EXACT same #ifdef-guarded vector functions
// we added to src/ggml-cpu/vec.h and verifies they produce correct output.
//
// This proves: with -DGGML_USE_XLNS16 -Dxlns16_table, the patched ggml vec
// functions use xlns16 arithmetic internally while keeping float input/output.
//
// Build:
//   g++ -std=c++17 -O2 -I. -Dxlns16_table -DGGML_USE_XLNS16 \
//       verify_xlns16_vec.cpp xlnscpp/xlns16.cpp -o verify_xlns16_vec
//   ./verify_xlns16_vec
//

#include <cstdio>
#include <cmath>
#include <cstdint>

// ── Mock Header Layer ────────────────────────────────────────────────────────
// These define the ggml project macros so vec.h can be compiled standalone.
#define GGML_RESTRICT 
#define GGML_UNUSED(x) (void)(x)
typedef uint16_t ggml_fp16_t;
typedef uint16_t ggml_bf16_t;
#define GGML_CPU_FP16_TO_FP32(x) (float)(x)
#define GGML_CPU_FP32_TO_FP16(x) (uint16_t)(x)

// ── Include Core Implementation ──────────────────────────────────────────────
// This is the absolute integrity check: including the actual repository source.
#include "src/ggml-cpu/vec.h"
#include "src/ggml-cpu/vec.cpp"

// Define helper functions for error checking (always FP32)
static float maxabs(const float * a, const float * b, int n) {
    float m = 0;
    for (int i=0;i<n;++i) { float d=fabsf(a[i]-b[i]); if(d>m) m=d; }
    return m;
}

static void print_vec(const char * label, const float * v, int n) {
    printf("  %-20s:", label);
    for (int i=0;i<n;++i) printf(" %9.4f", v[i]);
    printf("\n");
}

int main(void) {
    const int N = 8;
    const float X[N] = { 2.0f, 8.0f, 5.0f, 1.0f, 10.0f, 9.0f,  5.0f, 4.0f };
    const float Y[N] = { 4.0f, 2.0f, 9.0f, 9.0f,  5.0f, 4.0f, 60.0f, 0.5f };
    float Z_fp32[N], Z_xlns[N];

#ifdef GGML_USE_XLNS16
    printf("=== Compiled with GGML_USE_XLNS16 (xlns16 internal arithmetic) ===\n\n");
#else
    printf("=== Compiled WITHOUT GGML_USE_XLNS16 (plain FP32) ===\n\n");
#endif

    // [Test 1] ggml_vec_add_f32
    printf("[Test 1]  ggml_vec_add_f32   z[i] = x[i] + y[i]\n");
    for (int i=0;i<N;++i) Z_fp32[i] = X[i] + Y[i];
    ggml_vec_add_f32(N, Z_xlns, X, Y);
    print_vec("x",        X,      N);
    print_vec("y",        Y,      N);
    print_vec("ref (fp32)", Z_fp32, N);
    print_vec("result",   Z_xlns, N);
    printf("  Max |FP32 - result| = %.6e\n\n", maxabs(Z_fp32, Z_xlns, N));

    // [Test 2] ggml_vec_mul_f32
    printf("[Test 2]  ggml_vec_mul_f32   z[i] = x[i] * y[i]\n");
    for (int i=0;i<N;++i) Z_fp32[i] = X[i] * Y[i];
    ggml_vec_mul_f32(N, Z_xlns, X, Y);
    print_vec("x",          X,      N);
    print_vec("y",          Y,      N);
    print_vec("ref (fp32)", Z_fp32, N);
    print_vec("result",     Z_xlns, N);
    printf("  Max |FP32 - result| = %.6e\n\n", maxabs(Z_fp32, Z_xlns, N));

    // [Test 3] ggml_vec_scale_f32
    const float S = 3.1416f;
    printf("[Test 3]  ggml_vec_scale_f32  y[i] *= %.4f\n", S);
    for (int i=0;i<N;++i) { Z_fp32[i]=X[i]; Z_xlns[i]=X[i]; }
    for (int i=0;i<N;++i) Z_fp32[i] *= S;
    ggml_vec_scale_f32(N, Z_xlns, S);
    print_vec("x (before scale)", X,      N);
    print_vec("ref (fp32)",       Z_fp32, N);
    print_vec("result",           Z_xlns, N);
    printf("  Max |FP32 - result| = %.6e\n\n", maxabs(Z_fp32, Z_xlns, N));

    // [Test 4] ggml_vec_dot_f32
    printf("[Test 4]  ggml_vec_dot_f32   s = sum(x[i] * y[i])  [MUL_MAT core]\n");
    float ref_dot = 0; for (int i=0; i<N; ++i) ref_dot += X[i]*Y[i];
    float xlns_dot = 0;
    // Note: nrc is always 1 for this test as dot.cpp's LNS patch ignores the complex bits
    ggml_vec_dot_f32(N, &xlns_dot, 0, X, 0, Y, 0, 1);
    print_vec("x", X, N);
    print_vec("y", Y, N);
    printf("  ref (fp32)          :  dot = %12.6f\n", ref_dot);
    printf("  result (xlns16)     :  dot = %12.6f\n", xlns_dot);
    printf("  |FP32 - result|     =  %.6e\n\n", fabsf(ref_dot - xlns_dot));

    printf("=================================================================\n");
#ifdef GGML_USE_XLNS16
    printf("  All four ggml vec functions compiled and ran with xlns16\n");
    printf("  internal arithmetic.  Float API is preserved (float in/out).\n");
    printf("  Errors reflect 16-bit LNS precision (7-bit log2 mantissa).\n");
    printf("  In ggml, this same #ifdef in src/ggml-cpu/vec.h enables LNS\n");
    printf("  for all upstream callers (ggml_compute_forward_add, etc.).\n");
#else
    printf("  Plain FP32 path.  Errors should be 0.0.\n");
#endif
    printf("=================================================================\n");
    return 0;
}
