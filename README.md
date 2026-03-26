# ggml × xlns16 — Proof-of-Concept Fork

**GSoC 2026 · Project 8 · krishnamurthi-ramesh**

This is a professional proof-of-concept fork of [ggml](https://github.com/ggerganov/ggml) demonstrating the `#ifdef GGML_USE_XLNS16` integration approach for Logarithmic Number System (LNS) arithmetic, exactly as requested by maintainer **@markgarnold** in [xlnscpp Discussion #1](https://github.com/xlnsresearch/xlnscpp/discussions/1).

---

## ⚡ What was changed

All modifications follow the **"Core-Patch" strategy**, modifying existing vectorized operations rather than creating a separate backend. This ensures maximum compatibility and minimum memory overhead.

### Key Modifications

| File | Modification | Purpose |
|------|--------------|---------|
| `src/ggml-cpu/vec.h` | Patch `add_f32`, `mul_f32`, `scale_f32` | Fundamental arithmetic in LNS |
| `src/ggml-cpu/vec.cpp` | Patch `ggml_vec_dot_f32` | Native LNS inner loop for **MUL_MAT** |
| `xlnscpp/` | Full library inclusion | Standalone buildability |

### LNS-Native Accumulation
In `ggml_vec_dot_f32`, the accumulator is kept as an `xlns16_float` throughout the loop. This ensures:
- **Exact Multiplication**: Performed as integer addition of log representations.
- **Accurate Addition**: Uses the library's table-driven Gaussian log addition.
- **Zero Memory Overhead**: Floating point inputs are converted dynamically, avoiding shadow tensors.

---

## 🏗️ Professional Standards

This fork is built to the standards of a production-ready `ggml` contribution:
- **Zero Monkeypatches**: No "linter hacks" or fallback type definitions.
- **Standard Includes**: Uses the project's native `ggml-impl.h` and relative paths.
- **No Side Effects**: Standard FP32 behavior is preserved 100% when `GGML_USE_XLNS16` is not defined.

> [!IMPORTANT]
> **IDE Note**: Because this fork depends on the global `ggml` build system, your IDE may show "Header Not Found" errors for `ggml-impl.h`. This is expected and correct behavior for a modular `ggml` component; the paths are resolved at compile-time by `cmake`.

---

## 🚀 Verification

Compiling with `-DGGML_USE_XLNS16` enables the LNS path. You can verify the numerical results vs FP32 using the included test utility:

```bash
# Build and run the verification
g++ -std=c++17 -O2 -I. -Dxlns16_table -DGGML_USE_XLNS16 verify_xlns16_vec.cpp -o verify_xlns16_vec
./verify_xlns16_vec
```

---

## 📊 Status Tracker

| Operator | Status | Implementation Detail |
|----------|--------|-----------------------|
| **ADD** | ✅ Done | `ggml_vec_add_f32` |
| **MUL** | ✅ Done | `ggml_vec_mul_f32` |
| **SCALE** | ✅ Done | `ggml_vec_scale_f32` |
| **MUL_MAT** | ✅ Done | `ggml_vec_dot_f32` (LNS-native loop) |
| **SILU** | ⏳ Planned | Vectorized SILU |
| **RMS_NORM** | ⏳ Planned | Layer norm inner loop |

---

## References

- **GSoC Discussion**: [Topic #1 (The correct approach)](https://github.com/xlnsresearch/xlnscpp/discussions/1)
- **Code challenge**: [krishnamurthi-ramesh/Gsoc-xlnscpp-CodeChallenge](https://github.com/krishnamurthi-ramesh/Gsoc-xlnscpp-CodeChallenge)
- **xlnscpp**: [Official Library](https://github.com/xlnsresearch/xlnscpp)
