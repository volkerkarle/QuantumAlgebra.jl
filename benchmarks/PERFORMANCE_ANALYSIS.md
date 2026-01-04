# SU(N) Performance Analysis

## Optimization Status

**SU(2) Inline Optimization** (January 2026): Major performance improvement for SU(2) by using inline prefactor handling in `normal_order!`, matching TLS performance within ~20%.

---

## Benchmark Results Summary (After Inline Optimization)

### 1. Operator Creation: ✅ GOOD
| Operator | Time |
|----------|------|
| Boson a() | 132 ns |
| TLS σx() | 138 ns |
| SU(2) gen | 140 ns |
| SU(3) gen | 141 ns |
| SU(8) gen | 151 ns |

**Conclusion:** Minimal overhead (~10 ns) for SU(N) generators.

---

### 2. Simple Commutator [A, B]: ✅ NEAR PARITY
| Expression | Time | vs TLS |
|------------|------|--------|
| [σx, σy] TLS | 3.26 μs | 1.0x |
| [T¹, T²] SU(2) | 3.91 μs | **1.2x** |
| [λ¹, λ²] SU(3) | 6.78 μs | 2.1x |

**Conclusion:** SU(2) is now only ~1.2x slower than TLS (was 1.6x before optimization).

---

### 3. Product Normal Ordering: ✅ NEAR PARITY
| Expression | Time | vs TLS |
|------------|------|--------|
| σx * σy | 0.93 μs | 1.0x |
| T¹ * T² SU(2) | 1.1 μs | **1.2x** |
| λ¹ * λ² SU(3) | 3.21 μs | 3.4x |

**Conclusion:** SU(2) products are now only ~1.2x slower than TLS (was 3.1x before optimization).

---

### 4. Triple Product: ✅ NEAR PARITY
| Expression | Time | vs TLS |
|------------|------|--------|
| σx * σy * σz | 1.41 μs | 1.0x |
| T¹ * T² * T³ SU(2) | 1.64 μs | **1.2x** |
| λ¹ * λ² * λ³ SU(3) | 9.59 μs | 6.8x |

**Conclusion:** SU(2) triple products are now ~1.2x slower (was 3.6x before optimization).

---

### 5. Quadratic Casimir: ✅ PARITY FOR SU(2)
| Expression | Time | vs TLS |
|------------|------|--------|
| TLS (3 terms) | 2.91 μs | 1.0x |
| SU(2) (3 terms) | 2.97 μs | **1.0x** |
| SU(3) (8 terms) | 27.82 μs | 9.6x |

**Conclusion:** SU(2) Casimir is now at parity with TLS (was 5.7x before optimization)!

---

### 6. Mixed Boson + Spin: ✅ GOOD
| Expression | Time |
|------------|------|
| (a† + a)σx TLS | 0.50 μs |
| (a† + a)T¹ SU(2) | 0.41 μs |
| (a† + a)λ¹ SU(3) | 0.42 μs |

**Conclusion:** When SU(N) generators don't interact with each other, performance is equivalent or better.

---

### 7. Heisenberg EOM: ✅ GOOD
| Expression | Time |
|------------|------|
| d/dt a (TLS H) | 3.65 μs |
| d/dt a (SU(2) H) | 3.65 μs |
| d/dt a (SU(3) H) | 3.57 μs |
| d/dt λ¹ (SU(3) H) | 3.35 μs |

**Conclusion:** Equations of motion have identical performance. This is the primary use case!

---

### 8. Scaling (T¹ + T²)^n: ⚠️ IMPROVED
| n | TLS | SU(2) | SU(3) | SU(2)/TLS | SU(3)/TLS |
|---|-----|-------|-------|-----------|-----------|
| 2 | 4.4 μs | 5.1 μs | 12.8 μs | **1.2x** | 2.9x |
| 3 | 15.7 μs | 24.0 μs | 72.3 μs | 1.5x | 4.6x |
| 4 | 48.6 μs | 115.2 μs | 719.4 μs | 2.4x | 14.8x |

**Conclusion:** SU(2) scaling significantly improved. Low-order expressions (n≤2) are near parity.

---

### 9. Memory Allocations: ✅ NEAR PARITY
| Expression | Bytes | vs TLS |
|------------|-------|--------|
| σx * σy | 3,888 | 1.0x |
| T¹ * T² SU(2) | 4,384 | **1.1x** |
| λ¹ * λ² SU(3) | 11,152 | 2.9x |

**Conclusion:** SU(2) memory is now only 1.1x TLS (was 2.2x before optimization).

---

## Performance Improvement Summary

### SU(2) Before vs After Inline Optimization

| Benchmark | Before | After | Improvement |
|-----------|--------|-------|-------------|
| T¹ * T² product | 2.20 μs | **1.1 μs** | **2.0x faster** |
| [T¹, T²] commutator | 4.77 μs | **3.91 μs** | **1.2x faster** |
| T¹ * T² * T³ triple | 3.80 μs | **1.64 μs** | **2.3x faster** |
| C₂ Casimir (3 terms) | 4.84 μs | **2.97 μs** | **1.6x faster** |
| Memory (T¹ * T²) | 8,208 B | **4,384 B** | **1.9x less** |

### SU(2) vs TLS Overhead (After Optimization)

| Operation | Overhead |
|-----------|----------|
| Products | 1.2x |
| Commutators | 1.2x |
| Triple products | 1.2x |
| Casimir | **1.0x (parity!)** |
| Memory | 1.1x |

---

## Key Findings

### Good News ✅
1. **SU(2) is now at near-parity with TLS** for most operations
2. **Casimir calculations are at full parity** for SU(2)
3. **Typical physics use cases work excellently**: Hamiltonians, EOMs, expectation values
4. **Memory allocations significantly reduced** for SU(2)

### Remaining Challenges 🔴
1. **SU(3) and higher still have overhead** due to multi-term products
2. **Scaling for high-order expressions** still grows faster than TLS
3. **SU(N>2) uses Dict-based structure constants** with some overhead

---

## Optimizations Implemented

### Phase 1: SU(2) Fast Path (January 2026)
- Direct Levi-Civita computation for structure constants
- Bypassed Dict lookups for SU(2)
- Result: ~15-20% improvement

### Phase 2: Inline Prefactor Handling (January 2026)
- Changed `normal_order!` prefactor from `Complex{Int}` to `ComplexF64`
- SU(2) contractions now use inline processing like TLS
- Added `simplify_number` for automatic rational conversion
- **Result: ~2x improvement for SU(2), near-parity with TLS**

---

## Future Optimizations

### Medium Priority
1. **Add SU(3) fast path** using pre-computed 8×8 lookup tables
2. **Optimize multi-term ContractionResult handling** for SU(N>2)

### Low Priority
1. **StaticArrays for structure constants**: Replace `Matrix{Dict}` with `SMatrix`
2. **Lazy evaluation**: Delay normal ordering until needed
3. **Term coalescing**: Simplify intermediate expressions early

---

## Recommendations

### For Users
- **SU(2) is now recommended** for two-level system problems requiring Lie algebra structure
- **SU(N) is suitable for typical quantum optics**: Hamiltonians, EOMs, expectation values
- **Avoid deeply nested pure SU(N>2) products** like `(λ¹ + λ² + λ³)^10`

### For Developers
- SU(2) optimization is essentially complete
- Focus remaining efforts on SU(3) and higher
- Consider profiling typical physics workflows to identify hotspots
