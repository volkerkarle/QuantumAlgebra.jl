# SU(N) Performance Analysis

## Optimization Status

**SU(2) Fast Path Implemented** (January 2026): Direct Levi-Civita computation for SU(2) commutators and products, bypassing Dict lookups. This provides ~15-20% speedup for SU(2) operations.

---

## Benchmark Results Summary

### 1. Operator Creation: ✅ GOOD
| Operator | Time |
|----------|------|
| Boson a() | 132 ns |
| TLS σx() | 138 ns |
| SU(2) gen | 140 ns |
| SU(3) gen | 141 ns |
| SU(8) gen | 151 ns |

**Conclusion:** Minimal overhead (~10 ns) for SU(N) generators. The extra `algebra_id` and `gen_idx` fields don't significantly impact creation time.

---

### 2. Simple Commutator [A, B]: ⚠️ MODERATE OVERHEAD
| Expression | Time | vs TLS |
|------------|------|--------|
| [σx, σy] TLS | 3.06 μs | 1.0x |
| [T¹, T²] SU(2) | 4.77 μs | 1.6x |
| [λ¹, λ²] SU(3) | 6.84 μs | 2.2x |
| [G¹, G²] SU(4) | 6.86 μs | 2.2x |

**Conclusion:** SU(2) is ~1.6x slower than TLS (improved from 1.8x with fast path). The remaining overhead is from:
- ContractionResult processing in normal_order!
- Vector allocations for multi-term results

---

### 3. Product Normal Ordering: ⚠️ MODERATE OVERHEAD
| Expression | Time | vs TLS |
|------------|------|--------|
| σx * σy | 0.72 μs | 1.0x |
| T¹ * T² SU(2) | 2.20 μs | 3.1x |
| λ¹ * λ² SU(3) | 3.30 μs | 4.6x |
| G¹ * G² SU(4) | 3.11 μs | 4.3x |

**Conclusion:** Product rules are 3-4x slower (improved from 3.4x for SU(2)). The overhead is in the `normal_order!` processing of ContractionResult.

---

### 4. Triple Product: ⚠️ GROWING OVERHEAD
| Expression | Time | vs TLS |
|------------|------|--------|
| σx * σy * σz | 1.06 μs | 1.0x |
| T¹ * T² * T³ SU(2) | 3.80 μs | 3.6x |
| λ¹ * λ² * λ³ SU(3) | 8.66 μs | 8.2x |

**Conclusion:** SU(2) improved from 4.0x to 3.6x. SU(3) overhead grows with expression complexity because intermediate products generate multiple terms.

---

### 5. Quadratic Casimir: 🔴 SIGNIFICANT OVERHEAD
| Expression | Time | vs TLS |
|------------|------|--------|
| TLS (3 terms) | 0.85 μs | 1.0x |
| SU(2) (3 terms) | 4.84 μs | 5.7x |
| SU(3) (8 terms) | 23.58 μs | 28x |
| SU(4) (15 terms) | 48.44 μs | 57x |

**Conclusion:** Casimir calculations show the cost of many products. SU(2) improved from 6.2x to 5.7x.

---

### 6. Mixed Boson + Spin: ✅ GOOD
| Expression | Time |
|------------|------|
| (a† + a)σx TLS | 0.50 μs |
| (a† + a)T¹ SU(2) | 0.41 μs |
| (a† + a)λ¹ SU(3) | 0.42 μs |

**Conclusion:** When SU(N) generators don't interact with each other (just with bosons), performance is equivalent or better. This is the typical physics use case!

---

### 7. Heisenberg EOM: ✅ GOOD
| Expression | Time |
|------------|------|
| d/dt a (TLS H) | 3.65 μs |
| d/dt a (SU(2) H) | 3.65 μs |
| d/dt a (SU(3) H) | 3.57 μs |
| d/dt λ¹ (SU(3) H) | 3.35 μs |

**Conclusion:** Equations of motion are essentially identical performance. This is the primary use case and it works well!

---

### 8. Scaling (T¹ + T²)^n: 🔴 EXPONENTIAL BLOWUP
| n | TLS | SU(2) | SU(3) | SU(2)/TLS | SU(3)/TLS |
|---|-----|-------|-------|-----------|-----------|
| 2 | 2.61 μs | 7.13 μs | 10.9 μs | 2.7x | 4.2x |
| 3 | 11.4 μs | 30.1 μs | 69.6 μs | 2.6x | 6.1x |
| 4 | 37.5 μs | 178 μs | 617 μs | 4.7x | 16x |
| 5 | 114 μs | 1.18 ms | 5.91 ms | 10x | 52x |
| 6 | 304 μs | 9.62 ms | 77.5 ms | 32x | 255x |

**Conclusion:** This remains the critical problem. The multi-term results cascade exponentially.

---

### 9. Sum of Many Products: ⚠️ MODERATE
| N terms | TLS | SU(3) | Ratio |
|---------|-----|-------|-------|
| 10 | 2.59 μs | 22.1 μs | 8.5x |
| 50 | 2.58 μs | 22.2 μs | 8.6x |
| 100 | 2.57 μs | 22.8 μs | 8.9x |

**Conclusion:** Linear sums have ~9x overhead, but don't explode.

---

### 10. Structure Constant Lookup: ✅ VERY FAST
| Operation | Time |
|-----------|------|
| f[a,b] SU(2) | 1.83 ns |
| f[a,b] SU(3) | 1.80 ns |
| f[a,b] SU(4) | 1.67 ns |
| product_coefficients | 49 ns |

**Conclusion:** Dict lookup is not the bottleneck! The overhead is in the expression manipulation.

---

### 11. Memory Allocations: ⚠️ HIGHER
| Expression | Bytes | vs TLS |
|------------|-------|--------|
| σx * σy | 3,760 | 1.0x |
| T¹ * T² SU(2) | 8,208 | 2.2x |
| λ¹ * λ² SU(3) | 10,880 | 2.9x |
| C₂ SU(3) | 66,512 | - |

**Conclusion:** SU(2) memory improved from 2.5x to 2.2x with the fast path.

---

## Key Findings

### Good News ✅
1. **Typical physics use cases work well**: Hamiltonians with `(a† + a)T` terms have no overhead
2. **Heisenberg EOM is fast**: Primary use case performs identically to TLS
3. **Structure constant lookup is not a bottleneck**: 1-2 ns per lookup
4. **Operator creation is fast**: ~140 ns regardless of N
5. **SU(2) fast path provides ~15-20% improvement** over generic path

### Bad News 🔴
1. **Pure SU(N) algebra is 3-4x slower** than TLS for simple operations
2. **Scaling is exponential** for nested products like `(T¹ + T²)^n`
3. **SU(2) is still slower than TLS** due to ContractionResult processing overhead
4. **Memory allocations are 2-3x higher**

### Root Causes
1. **ContractionResult processing**: Even with fast structure constant lookup, the `normal_order!` handling of ContractionResult is slower than the legacy tuple path
2. **Vector allocations**: `ops::Vector{Tuple{ComplexF64, BaseOperator}}` for multi-term results
3. **Additive vs multiplicative convention**: TLS uses multiplicative prefactors, SU(N) uses additive terms

---

## Optimizations Implemented

### SU(2) Fast Path (January 2026)
- **`su2_commutator_result(a, b)`**: Direct Levi-Civita computation for `[T^a, T^b] = i ε_{abc} T^c`
- **`su2_product_result(a, b)`**: Direct computation for `T^a T^b = (1/4)δ_{ab}I + (i/2)ε_{abc}T^c`
- **`_exchange_lie_algebra_generators`**: Fast path for `algebra_id == SU2_ALGEBRA_ID`
- **`_contract_lie_algebra_generators`**: Fast path for `algebra_id == SU2_ALGEBRA_ID`

### Results
| Benchmark | Before | After | Improvement |
|-----------|--------|-------|-------------|
| [T¹, T²] SU(2) | 5.85 μs | 4.77 μs | 18% faster |
| T¹ * T² SU(2) | 2.71 μs | 2.20 μs | 19% faster |
| T¹ * T² * T³ SU(2) | 4.78 μs | 3.80 μs | 20% faster |
| C₂ SU(2) | 5.76 μs | 4.84 μs | 16% faster |
| Memory SU(2) | 9,584 B | 8,208 B | 14% less |

---

## Future Optimizations

### High Priority
1. **Optimize ContractionResult processing in normal_order!**: The main remaining overhead
2. **Use legacy tuple format for SU(2) off-diagonal products**: Would enable inline processing

### Medium Priority
1. **Use NTuple instead of Vector** for small fixed-size results (≤3 terms)
2. **Pre-allocate result buffers** to reduce GC pressure
3. **Add SU(3) fast path** using pre-computed 8×8 lookup tables

### Low Priority
1. **StaticArrays for structure constants**: Replace `Matrix{Dict}` with `SMatrix`
2. **Lazy evaluation**: Delay normal ordering until needed
3. **Term coalescing**: Simplify intermediate expressions early

---

## Recommendations

### For Users
- SU(N) is **suitable for typical quantum optics**: Hamiltonians, EOMs, expectation values
- **Avoid deeply nested pure SU(N) products** like `(λ¹ + λ² + λ³)^10`
- For SU(2), the Lie algebra implementation is ~3x slower than TLS for products

### For Developers
- Focus optimization efforts on `normal_order!` ContractionResult handling
- Consider unifying TLS and SU(2) code paths for better performance
- Profile memory allocations to identify GC hotspots
