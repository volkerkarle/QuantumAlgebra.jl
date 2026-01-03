# SU(N) Performance Analysis

## Benchmark Results Summary

### 1. Operator Creation: ✅ GOOD
| Operator | Time |
|----------|------|
| Boson a() | 131 ns |
| TLS σx() | 141 ns |
| SU(2) gen | 152 ns |
| SU(3) gen | 144 ns |
| SU(8) gen | 150 ns |

**Conclusion:** Minimal overhead (~10 ns) for SU(N) generators. The extra `algebra_id` and `gen_idx` fields don't significantly impact creation time.

---

### 2. Simple Commutator [A, B]: ⚠️ MODERATE OVERHEAD
| Expression | Time | vs TLS |
|------------|------|--------|
| [σx, σy] TLS | 3.18 μs | 1.0x |
| [T¹, T²] SU(2) | 5.85 μs | 1.8x |
| [λ¹, λ²] SU(3) | 6.99 μs | 2.2x |
| [G¹, G²] SU(4) | 7.14 μs | 2.2x |

**Conclusion:** SU(2) is ~2x slower than TLS due to:
- Dict lookup for structure constants
- More general code path
- SU(3)/SU(4) similar to SU(2) because commutators still produce single terms

---

### 3. Product Normal Ordering: ⚠️ MODERATE OVERHEAD
| Expression | Time | vs TLS |
|------------|------|--------|
| σx * σy | 0.79 μs | 1.0x |
| T¹ * T² SU(2) | 2.71 μs | 3.4x |
| λ¹ * λ² SU(3) | 3.39 μs | 4.3x |
| G¹ * G² SU(4) | 3.41 μs | 4.3x |

**Conclusion:** Product rules are 3-4x slower. The contraction logic with Dict lookups and multi-term handling adds overhead even when results are simple.

---

### 4. Triple Product: ⚠️ GROWING OVERHEAD
| Expression | Time | vs TLS |
|------------|------|--------|
| σx * σy * σz | 1.20 μs | 1.0x |
| T¹ * T² * T³ SU(2) | 4.78 μs | 4.0x |
| λ¹ * λ² * λ³ SU(3) | 9.62 μs | 8.0x |

**Conclusion:** SU(3) overhead grows with expression complexity because intermediate products generate multiple terms that must all be processed.

---

### 5. Quadratic Casimir: 🔴 SIGNIFICANT OVERHEAD
| Expression | Time | vs TLS |
|------------|------|--------|
| TLS (3 terms) | 0.93 μs | 1.0x |
| SU(2) (3 terms) | 5.76 μs | 6.2x |
| SU(3) (8 terms) | 23.68 μs | 25x |
| SU(4) (15 terms) | 53.96 μs | 58x |

**Conclusion:** Casimir calculations show the cost of many products. SU(4) with 15 generators is 58x slower than TLS equivalent.

---

### 6. Mixed Boson + Spin: ✅ GOOD
| Expression | Time |
|------------|------|
| (a† + a)σx TLS | 0.52 μs |
| (a† + a)T¹ SU(2) | 0.46 μs |
| (a† + a)λ¹ SU(3) | 0.43 μs |

**Conclusion:** When SU(N) generators don't interact with each other (just with bosons), performance is equivalent or better. This is the typical physics use case!

---

### 7. Heisenberg EOM: ✅ GOOD
| Expression | Time |
|------------|------|
| d/dt a (TLS H) | 3.88 μs |
| d/dt a (SU(2) H) | 3.81 μs |
| d/dt a (SU(3) H) | 3.93 μs |
| d/dt λ¹ (SU(3) H) | 3.65 μs |

**Conclusion:** Equations of motion are essentially identical performance. This is the primary use case and it works well!

---

### 8. Scaling (T¹ + T²)^n: 🔴 EXPONENTIAL BLOWUP
| n | TLS | SU(2) | SU(3) | SU(2)/TLS | SU(3)/TLS |
|---|-----|-------|-------|-----------|-----------|
| 2 | 2.96 μs | 8.87 μs | 11.3 μs | 3x | 4x |
| 3 | 14.1 μs | 34.5 μs | 71.1 μs | 2.4x | 5x |
| 4 | 40.8 μs | 219 μs | 648 μs | 5.4x | 16x |
| 5 | 120 μs | 1.46 ms | 5.69 ms | 12x | 47x |
| 6 | 302 μs | 10.9 ms | 77.6 ms | 36x | 257x |

**Conclusion:** This is the critical problem! 
- SU(2) scales much worse than TLS (36x at n=6)
- SU(3) scales catastrophically (257x at n=6)
- The multi-term results cascade exponentially

---

### 9. Sum of Many Products: ⚠️ MODERATE
| N terms | TLS | SU(3) | Ratio |
|---------|-----|-------|-------|
| 10 | 2.63 μs | 20.7 μs | 7.9x |
| 50 | 2.46 μs | 27.1 μs | 11x |
| 100 | 2.83 μs | 25.0 μs | 8.8x |

**Conclusion:** Linear sums have ~10x overhead, but don't explode. The TLS time is suspiciously constant (likely simplifying to a constant).

---

### 10. Structure Constant Lookup: ✅ VERY FAST
| Operation | Time |
|-----------|------|
| f[a,b] SU(2) | 2.58 ns |
| f[a,b] SU(3) | 1.97 ns |
| f[a,b] SU(4) | 2.88 ns |
| product_coefficients | 66 ns |

**Conclusion:** Dict lookup is not the bottleneck! The overhead is in the expression manipulation, not structure constant access.

---

### 11. Memory Allocations: ⚠️ HIGHER
| Expression | Bytes | vs TLS |
|------------|-------|--------|
| σx * σy | 3,760 | 1.0x |
| T¹ * T² SU(2) | 9,584 | 2.5x |
| λ¹ * λ² SU(3) | 10,880 | 2.9x |
| C₂ SU(3) | 66,512 | - |

**Conclusion:** 2.5-3x more memory per operation. This contributes to GC pressure in large expressions.

---

## Key Findings

### Good News ✅
1. **Typical physics use cases work well**: Hamiltonians with `(a† + a)T` terms have no overhead
2. **Heisenberg EOM is fast**: Primary use case performs identically to TLS
3. **Structure constant lookup is not a bottleneck**: 2-3 ns per lookup
4. **Operator creation is fast**: ~150 ns regardless of N

### Bad News 🔴
1. **Pure SU(N) algebra is 3-4x slower** than TLS for simple operations
2. **Scaling is exponential** for nested products like `(T¹ + T²)^n`
3. **SU(2) is slower than TLS** even though mathematically equivalent - implementation overhead
4. **Memory allocations are 2.5-3x higher**

### Root Causes
1. **Multi-term ExchangeResult/ContractionResult**: Each operation can spawn multiple terms
2. **Dict usage**: More flexible but slower than direct computation
3. **General code path**: SU(N) code is more complex than specialized TLS code
4. **Vector allocations**: `ops::Vector{Tuple{ComplexF64, BaseOperator}}` for multi-term results

---

## Recommendations

### For Users
- SU(N) is **suitable for typical quantum optics**: Hamiltonians, EOMs, expectation values
- **Avoid deeply nested pure SU(N) products** like `(λ¹ + λ² + λ³)^10`
- For large-scale computations, consider **SU(2) via TLS** for better performance

### For Future Optimization
1. **Specialize SU(2)**: Detect algebra_id=1 and use direct computation like TLS
2. **Use StaticArrays**: Replace `Vector{Tuple}` with `SVector` for small result sets
3. **Pre-compute common products**: Cache T^a T^a = 1/4 for diagonal cases
4. **Lazy evaluation**: Delay normal ordering until needed

### For PR to Johannes
- **Document performance characteristics** in README
- **Highlight that typical use cases are fast**
- **Note that nested pure algebra expressions scale poorly**
- **Suggest SU(2) specialization as future work**
