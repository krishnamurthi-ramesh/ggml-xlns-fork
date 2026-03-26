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

#include <cstdio>
#include <cmath>
#include <cfloat>

// ── Include xlns16 ──────────────────────────────────────────────────────────
// xlns16_table must be defined before including xlns16.cpp
#ifdef GGML_USE_XLNS16
#include "xlnscpp/xlns16.cpp"
#endif

// ── Replicate the patched vec functions (exact copies from vec.h) ───────────

static void ggml_vec_add_f32(const int n, float * z,
                              const float * x, const float * y) {
#ifdef GGML_USE_XLNS16
    for (int i = 0; i < n; ++i)
        z[i] = xlns16_2float(float2xlns16_(x[i]) + float2xlns16_(y[i]));
#else
    for (int i = 0; i < n; ++i)
        z[i] = x[i] + y[i];
#endif
}

static void ggml_vec_mul_f32(const int n, float * z,
                              const float * x, const float * y) {
#ifdef GGML_USE_XLNS16
    for (int i = 0; i < n; ++i)
        z[i] = xlns16_2float(float2xlns16_(x[i]) * float2xlns16_(y[i]));
#else
    for (int i = 0; i < n; ++i)
        z[i] = x[i] * y[i];
#endif
}

static void ggml_vec_scale_f32(const int n, float * y, const float v) {
#ifdef GGML_USE_XLNS16
    const xlns16_float lv = float2xlns16_(v);
    for (int i = 0; i < n; ++i)
        y[i] = xlns16_2float(float2xlns16_(y[i]) * lv);
#else
    for (int i = 0; i < n; ++i)
        y[i] *= v;
#endif
}

// ggml_vec_dot_f32 — powers MUL_MAT, the most critical operator
// Accumulator stays in xlns16_float throughout: MUL is exact, ADD uses sb/db tables
static float ggml_vec_dot_f32(const int n,
                               const float * x, const float * y) {
#ifdef GGML_USE_XLNS16
    xlns16_float sum = float2xlns16_(0.0f);
    for (int i = 0; i < n; ++i)
        sum = sum + (float2xlns16_(x[i]) * float2xlns16_(y[i]));
    return xlns16_2float(sum);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i] * y[i];
    return sum;
#endif
}

// ── FP32 reference (always uses plain float) ────────────────────────────────
static void ref_add  (const int n, float * z, const float * x, const float * y)
                      { for (int i=0;i<n;++i) z[i] = x[i]+y[i]; }
static void ref_mul  (const int n, float * z, const float * x, const float * y)
                      { for (int i=0;i<n;++i) z[i] = x[i]*y[i]; }
static void ref_scale(const int n, float * y, const float v)
                      { for (int i=0;i<n;++i) y[i] *= v; }

// ── Helpers ─────────────────────────────────────────────────────────────────
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

// ── main ────────────────────────────────────────────────────────────────────
int main(void) {
    const int N = 8;

    // Test vectors — same dynamic range as ggml blog matrices
    const float X[N] = { 2.0f, 8.0f, 5.0f, 1.0f, 10.0f, 9.0f,  5.0f, 4.0f };
    const float Y[N] = { 4.0f, 2.0f, 9.0f, 9.0f,  5.0f, 4.0f, 60.0f, 0.5f };
    const float S    = 3.14159f;   // scalar for scale test

    float R[N], Z_fp32[N], Z_xlns[N], TMP[N];

#ifdef GGML_USE_XLNS16
    printf("=== Compiled with GGML_USE_XLNS16 (xlns16 internal arithmetic) ===\n\n");
#else
    printf("=== Compiled WITHOUT GGML_USE_XLNS16 (plain FP32) ===\n\n");
#endif

    // ── Test 1: ggml_vec_add_f32 ─────────────────────────────────────────
    printf("[Test 1]  ggml_vec_add_f32   z[i] = x[i] + y[i]\n");
    ref_add(N, Z_fp32, X, Y);
    ggml_vec_add_f32(N, Z_xlns, X, Y);
    print_vec("x",        X,      N);
    print_vec("y",        Y,      N);
    print_vec("ref (fp32)", Z_fp32, N);
    print_vec("result",   Z_xlns, N);
    printf("  Max |FP32 - result| = %.6e\n\n", maxabs(Z_fp32, Z_xlns, N));

    // ── Test 2: ggml_vec_mul_f32 ─────────────────────────────────────────
    printf("[Test 2]  ggml_vec_mul_f32   z[i] = x[i] * y[i]\n");
    ref_mul(N, Z_fp32, X, Y);
    ggml_vec_mul_f32(N, Z_xlns, X, Y);
    print_vec("x",          X,      N);
    print_vec("y",          Y,      N);
    print_vec("ref (fp32)", Z_fp32, N);
    print_vec("result",     Z_xlns, N);
    printf("  Max |FP32 - result| = %.6e\n\n", maxabs(Z_fp32, Z_xlns, N));

    // ── Test 3: ggml_vec_scale_f32 ───────────────────────────────────────
    printf("[Test 3]  ggml_vec_scale_f32  y[i] *= %.4f\n", S);
    for (int i=0;i<N;++i) { Z_fp32[i]=X[i]; Z_xlns[i]=X[i]; }
    ref_scale(N, Z_fp32, S);
    ggml_vec_scale_f32(N, Z_xlns, S);
    print_vec("x (before scale)", X,      N);
    print_vec("ref (fp32)",       Z_fp32, N);
    print_vec("result",           Z_xlns, N);
    printf("  Max |FP32 - result| = %.6e\n\n", maxabs(Z_fp32, Z_xlns, N));

    // ── Test 4: ggml_vec_dot_f32 — powers MUL_MAT ───────────────────────
    printf("[Test 4]  ggml_vec_dot_f32   s = sum(x[i] * y[i])  [MUL_MAT core]\n");
    float ref_dot = 0.0f;
    for (int i=0;i<N;++i) ref_dot += X[i]*Y[i];
    float xlns_dot = ggml_vec_dot_f32(N, X, Y);
    printf("  x                   :");
    for (int i=0;i<N;++i) printf(" %9.4f", X[i]); printf("\n");
    printf("  y                   :");
    for (int i=0;i<N;++i) printf(" %9.4f", Y[i]); printf("\n");
    printf("  ref (fp32)          :  dot = %12.6f\n", ref_dot);
    printf("  result (xlns16)     :  dot = %12.6f\n", xlns_dot);
    printf("  |FP32 - result|     =  %.6e\n\n", fabsf(ref_dot - xlns_dot));

    printf("=================================================================\n");
#ifdef GGML_USE_XLNS16
    printf("  All three ggml vec functions compiled and ran with xlns16\n");
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
