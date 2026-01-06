# Implementation Notes: Symbolic Sums and Products for QuantumAlgebra

**Date:** January 6, 2026  
**Task:** Add proper symbolic sum support with correct commutator semantics

---

## Background and Motivation

### The Problem

QuantumAlgebra's existing `sumindex(n)` function creates dummy indices like `#₁`, `#₂` for representing sums over sites (e.g., `Σᵢ σz(i)`). However, when computing commutators of expressions with the same sum index, it treats them as operators at the **same site only**.

Mathematically, `[Σᵢ Aᵢ, Σⱼ Bⱼ]` should produce both:
- **Same-site contribution:** `Σᵢ [Aᵢ, Bᵢ]`
- **Cross-site contribution:** `Σᵢ Σⱼ≠ᵢ [Aᵢ, Bⱼ]`

### Physics Context

This is critical for the Schrieffer-Wolff transformation in multi-atom/multi-qubit systems. For example, the Tavis-Cummings model (N atoms in a cavity) should produce the exchange interaction:

```
H_exchange = Σᵢ≠ⱼ (g²/Δ) σ⁺(i)σ⁻(j)
```

This term represents cavity-mediated spin-spin coupling, essential for:
- Quantum gates between qubits
- Collective spin dynamics
- Superradiance

---

## Design Decisions

### Key Insight: Bound Variables

Sum indices are **bound variables** (like in lambda calculus). This requires careful handling:

| Operation | Rule |
|-----------|------|
| `(Σᵢ Xᵢ) * Xᵢ` | Rename bound index: `Σⱼ Xⱼ Xᵢ` |
| `Σᵢ Xᵢ + Σⱼ Xⱼ` | Alpha-equivalent: `2 Σᵢ Xᵢ` |
| `(Σᵢ Xᵢ)³` | Fresh indices: `Σᵢ Σⱼ Σₖ Xᵢ Xⱼ Xₖ` |

### Architecture Choices

1. **Wrapper type approach:** `SymSum` wraps a `QuExpr` rather than being a subtype. This is less invasive to existing code.

2. **`SymExpr` container:** A general expression type that can hold multiple symbolic sums plus regular terms. Provides a closed algebra where all operations return well-defined types.

3. **Lazy index renaming:** Only rename bound variables when there's an actual conflict (detected via `free_indices`).

4. **Separate from existing `∑`:** The new system is parallel to the existing `∑` infrastructure, allowing gradual adoption.

5. **Canonicalization for display:** Indices are renumbered for readable output, but internal representation uses actual indices.

---

## Implementation Details

### Core Types

```julia
# Single symbolic sum
struct SymSum <: AbstractSymbolicAggregate
    expr::Union{QuExpr, AbstractSymbolicAggregate}  # Summand (can nest)
    index::QuIndex                                   # Bound index
    excluded::Vector{QuIndex}                        # For i≠j constraints
end

# Single symbolic product  
struct SymProd <: AbstractSymbolicAggregate
    expr::Union{QuExpr, AbstractSymbolicAggregate}
    index::QuIndex
    excluded::Vector{QuIndex}
end

# General container for sums of symbolic aggregates
struct SymExpr
    terms::Vector{Tuple{Number, AbstractSymbolicAggregate}}
    scalar::QuExpr
end
```

### Fresh Index Generation

Thread-safe counter starting at 1000 to avoid conflicts:

```julia
const _FRESH_SUMINDEX_COUNTER = Threads.Atomic{Int}(1000)

function fresh_sumindex()
    n = Threads.atomic_add!(_FRESH_SUMINDEX_COUNTER, 1)
    sumindex(n)
end
```

### Free Index Analysis

Critical for detecting when bound variables need renaming:

```julia
function free_indices(expr::QuExpr)
    # Get all indices, filter out those bound by QuTerm.nsuminds
    all_inds = Set{QuIndex}()
    for (term, _) in expr.terms
        bound_by_term = Set(sumindex(i) for i in 1:term.nsuminds)
        term_inds = Set(indices(term))
        union!(all_inds, setdiff(term_inds, bound_by_term))
    end
    return all_inds
end

function free_indices(s::SymSum)
    inner_free = free_indices(s.expr)
    delete!(inner_free, s.index)  # Bound by this sum
    union!(inner_free, Set(s.excluded))  # Excluded refer to outer scope
    return inner_free
end
```

### Multiplication with Bound Variable Handling

```julia
function Base.:*(s::SymSum, b::QuExpr)
    if has_free_index(b, s.index)
        # Rename bound index to avoid capture
        s = ensure_fresh_index(s, Set([s.index]))
    end
    SymSum(s.expr * b, s.index, s.excluded)
end

function Base.:*(a::SymSum, b::SymSum)
    # Always use fresh indices to avoid conflicts
    all_free = union(free_indices(a), free_indices(b))
    a = ensure_fresh_index(a, union(all_free, Set([b.index])))
    b = ensure_fresh_index(b, union(all_free, Set([a.index])))
    
    # Nested sum: Σᵢ Σⱼ (Aᵢ * Bⱼ)
    inner = SymSum(a.expr * b.expr, b.index, b.excluded)
    SymSum(inner, a.index, a.excluded)
end
```

### Commutator: Same-Site + Cross-Site

The key physics function:

```julia
function comm(A::SymSum, B::SymSum)
    # Ensure distinct indices
    all_free = union(free_indices(A), free_indices(B))
    A = ensure_fresh_index(A, union(all_free, Set([B.index])))
    
    # Same-site: Σᵢ [Aᵢ, Bᵢ]
    B_same = replace_sym_index(B.expr, B.index, A.index)
    same_site_comm = comm(A.expr, B_same)
    same_site = SymSum(same_site_comm, A.index, A.excluded)
    
    # Cross-site: Σᵢ Σⱼ≠ᵢ [Aᵢ, Bⱼ]
    j = fresh_sumindex()
    B_cross = replace_sym_index(B.expr, B.index, j)
    cross_site_comm = comm(A.expr, B_cross)
    inner_sum = SymSum(cross_site_comm, j, [A.index])  # j≠i constraint
    cross_site = SymSum(inner_sum, A.index, A.excluded)
    
    SymExpr([(1, same_site), (1, cross_site)], zero(QuExpr))
end
```

### Expansion with Exclusion Constraints

```julia
function expand_symbolic(s::SymSum, range; excluded_values::Dict{QuIndex,Any}=Dict())
    result = zero(QuExpr)
    
    for val in range
        # Check exclusions
        skip = false
        for excl in s.excluded
            if isintindex(excl) && excl.num == val
                skip = true; break
            elseif haskey(excluded_values, excl) && excluded_values[excl] == val
                skip = true; break
            end
        end
        skip && continue
        
        # Replace and recurse
        concrete_idx = QuIndex(val)
        term = replace_sym_index(s.expr, s.index, concrete_idx)
        
        if term isa SymSum
            new_excluded = copy(excluded_values)
            new_excluded[s.index] = val  # Track for nested exclusions
            term = expand_symbolic(term, range; excluded_values=new_excluded)
        end
        
        result = result + term
    end
    return result
end
```

### Alpha-Equivalence

```julia
function alpha_equivalent(a::SymSum, b::SymSum)
    length(a.excluded) != length(b.excluded) && return false
    
    # Canonicalize both and compare
    a_canon = canonicalize_indices(a)
    b_canon = canonicalize_indices(b)
    
    return a_canon.expr == b_canon.expr && a_canon.excluded == b_canon.excluded
end
```

---

## Files Created/Modified

| File | Action | Lines |
|------|--------|-------|
| `src/symbolic_sums.jl` | Created | ~700 |
| `src/QuantumAlgebra.jl` | Modified | +1 |
| `test/test_symbolic_sums.jl` | Created | ~500 |
| `test/runtests.jl` | Modified | +3 |

---

## Exported API

```julia
# Types
SymSum, SymProd, SymExpr

# Constructors
symbolic_sum(expr, index, [excluded])
symbolic_prod(expr, index, [excluded])

# Functions
expand_symbolic(s, range)       # Expand to explicit sum/product
free_indices(expr)              # Get free (unbound) indices
canonicalize_indices(s)         # Renumber indices canonically
simplify_sums(e::SymExpr)       # Combine alpha-equivalent terms
```

---

## Test Results

### Symbolic Sums Tests: 96/96 passing

- Basic construction
- SymExpr construction
- Free indices detection
- Index replacement
- Negation, scalar multiplication
- Addition, subtraction
- Multiplication with bound variable handling
- Power with fresh indices
- Expansion (basic, with exclusion, products, SymExpr)
- Canonicalization
- Alpha-equivalence
- Simplify sums
- Commutator (SymSum with QuExpr, SymSum with SymSum)
- Tavis-Cummings exchange interaction
- Display and LaTeX
- Zero and one
- Normal form through SymSum
- SymExpr equality
- Symbolic algebra integration

### Full Package Tests: 2196/2196 passing (4 pre-existing broken)

---

## Physics Validation

### Tavis-Cummings Exchange Interaction

```julia
i = sumindex(1)
V = SymSum(a'() * σm(i) + a() * σp(i), i)  # Σᵢ (a†σ⁻ + aσ⁺)
S = SymSum(a() * σp(i) - a'() * σm(i), i)  # Σᵢ (aσ⁺ - a†σ⁻)

result = comm(S, V)
expanded = normal_form(expand_symbolic(result, 1:2))
```

**Result:**
```
2 + σᶻ(1) + σᶻ(2) + σˣ(1)σˣ(2) + σʸ(1)σʸ(2) + 2a†()σᶻ(1)a() + 2a†()σᶻ(2)a()
```

**Verification:**
- `σˣ(1)σˣ(2) + σʸ(1)σʸ(2) = 2(σ⁺(1)σ⁻(2) + σ⁺(2)σ⁻(1))` ✅
- This IS the exchange interaction in the σx/σy basis!

---

## Usage Examples

### Basic Sum

```julia
using QuantumAlgebra
using QuantumAlgebra: SymSum, expand_symbolic, sumindex

i = sumindex(1)
s = SymSum(σz(i), i)  # Σᵢ σz(i)

expand_symbolic(s, 1:3)  # → σz(1) + σz(2) + σz(3)
```

### Sum Squared

```julia
s = SymSum(a(i), i)
s^2  # → Σⱼ Σₖ a(j)a(k) with fresh indices

expand_symbolic(s^2, 1:2)  # → a(1)² + a(1)a(2) + a(2)a(1) + a(2)²
```

### Combining Equivalent Sums

```julia
using QuantumAlgebra: simplify_sums

i, j = sumindex(1), sumindex(2)
s1 = SymSum(σz(i), i)
s2 = SymSum(σz(j), j)

combined = s1 + s2  # SymExpr with 2 terms
simplified = simplify_sums(combined)  # → 2 * Σᵢ σz(i)
```

### Commutator with Cross-Site Terms

```julia
i = sumindex(1)
S = SymSum(σp(i), i)
V = SymSum(σm(i), i)

result = comm(S, V)
# Returns SymExpr with:
# - Same-site: Σᵢ [σ⁺(i), σ⁻(i)] = Σᵢ σz(i)
# - Cross-site: Σᵢ Σⱼ≠ᵢ [σ⁺(i), σ⁻(j)] = 0 (different sites commute)

expand_symbolic(result, 1:2)  # → σz(1) + σz(2)
```

### Exclusion Constraints

```julia
i, j = sumindex(1), sumindex(2)

# Σᵢ Σⱼ≠ᵢ σz(j)
inner = SymSum(σz(j), j, [i])  # j ≠ i
outer = SymSum(inner, i)

expand_symbolic(outer, 1:3)  # → 2σz(1) + 2σz(2) + 2σz(3)
```

---

## Future Enhancements

1. **Integration with existing `∑`:** Add `to_symbolic_sum(expr::QuExpr)` to convert expressions using `QuTerm.nsuminds`.

2. **Symbolic range:** Allow `expand_symbolic(s, N)` where N is symbolic, returning expressions with symbolic sums.

3. **Automatic simplification:** Recognize when cross-site terms vanish for commuting operators.

4. **Pretty printing:** Improve display for deeply nested sums.

5. **Performance:** Optimize for large expressions with many sum indices.

---

## References

- Task specification: `TASK_symbolic_sums.md`
- Lambda calculus bound variable semantics
- Schrieffer-Wolff transformation
- Tavis-Cummings model
