# ggml × xlns16 — Proof-of-Concept Fork

**GSoC 2026 · Project 8 · krishnamurthi-ramesh**

This is a minimal fork of [ggml](https://github.com/ggerganov/ggml) demonstrating
the `#ifdef GGML_USE_XLNS16` integration approach for Logarithmic Number System
(LNS) arithmetic, exactly as described by maintainer **@markgarnold** in
[xlnscpp Discussion #1](https://github.com/xlnsresearch/xlnscpp/discussions/1):

> *"There will be a forked copy of ggml in which you will change many instances
> of things like `ggml_vec_add_f32` to use xlns16 internally."*

---

## What was changed

All modifications are in **`src/ggml-cpu/vec.h`** only.
The xlnscpp library is added as **`xlnscpp/`**.

### Modified functions

| Function | LNS operation |
|----------|--------------|
| `ggml_vec_add_f32` | ADD via `xlns16_float` overloaded `+` |
| `ggml_vec_mul_f32` | MUL via `xlns16_float` overloaded `*` (exact in LNS) |
| `ggml_vec_scale_f32` | SCALE: scalar broadcast multiply |

### Pattern used (matches @markgarnold description exactly)

```c
inline static void ggml_vec_add_f32(const int n, float * z,
                                    const float * x, const float * y) {
#ifdef GGML_USE_XLNS16
    // Dynamic float -> xlns16 -> compute -> float, no caching
    for (int i = 0; i < n; ++i)
        z[i] = xlns16_2float(float2xlns16_(x[i]) + float2xlns16_(y[i]));
#else
    for (int i = 0; i < n; ++i) z[i] = x[i] + y[i];
#endif
}
```

---

## Key design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Where to patch | `vec.h` inline functions | Lowest-level, affects all callers automatically |
| Conversion | Dynamic (no cache) | Less RAM; `float2xlns16_` caches internally |
| Compile flag | `-DGGML_USE_XLNS16 -Dxlns16_table` | Table-driven LNS addition |
| Storage format | Unchanged (float/quantized) | No GGUF modifications needed |
| xlns32 | Not used | Only xlns16 is the target |

---

## Building

```bash
# Standard FP32 build (no change to behaviour):
cmake -B build && cmake --build build

# Build with xlns16 internal arithmetic:
cmake -B build -DGGML_USE_XLNS16=ON -DCMAKE_CXX_FLAGS="-Dxlns16_table"
cmake --build build
```

---

## Operator coverage plan

| Operator | ggml function | Status |
|----------|--------------|--------|
| ADD | `ggml_vec_add_f32` | Done |
| MUL | `ggml_vec_mul_f32` | Done |
| SCALE | `ggml_vec_scale_f32` | Done |
| MUL_MAT | `ggml_vec_dot_f32` | Next |
| SILU | `ggml_vec_silu_f32` | Planned |
| RMS_NORM | inline in ggml-cpu.c | Planned |
| SOFT_MAX | `ggml_vec_soft_max_f32` | Planned |

---

## References

- xlnscpp: https://github.com/xlnsresearch/xlnscpp
- GSoC Discussion: https://github.com/xlnsresearch/xlnscpp/discussions/1
- Code challenge: https://github.com/krishnamurthi-ramesh/Gsoc-xlnscpp-CodeChallenge
- ggml upstream: https://github.com/ggerganov/ggml
