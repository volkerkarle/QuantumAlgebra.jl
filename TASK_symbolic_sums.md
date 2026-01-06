# Task: Add Proper Symbolic Sum Support to QuantumAlgebra

## Background

QuantumAlgebra currently has a `sumindex(n)` function that creates dummy indices like `#₁`, `#₂`. These are used to represent sums over sites, e.g., `Σᵢ σz(i)` is written as `σz(sumindex(1))`.

**The problem**: When computing commutators of expressions with the same sum index, QuantumAlgebra treats them as operators at the **same site**. But mathematically, `[Σᵢ Aᵢ, Σⱼ Bⱼ]` should produce both same-site AND cross-site contributions:

```
[Σᵢ Aᵢ, Σⱼ Bⱼ] = Σᵢ [Aᵢ, Bᵢ]  +  Σᵢ Σⱼ≠ᵢ [Aᵢ, Bⱼ]
                     ↑                  ↑
                same-site          cross-site
```

## Current Behavior

```julia
using QuantumAlgebra
i = sumindex(1)

S = a() * σp(i)      # Represents Σᵢ a σ⁺(i)
V = a'() * σm(i)     # Represents Σᵢ a† σ⁻(i)

comm(S, V)  # Only gives same-site [Sᵢ, Vᵢ] contribution
```

The cross-site term `σ⁺(i)σ⁻(j)` (which represents exchange interactions like `J_ij ∝ gᵢgⱼ` in multi-atom physics) is missing.

## Desired Behavior

We need a way to represent symbolic sums explicitly, so that commutators correctly produce both contributions.

## Proposed Design

Introduce a `SymbolicSum` type (or similar) that wraps an expression and tracks the summation:

```julia
# Option 1: Explicit sum wrapper
struct SymbolicSum
    expr::QuExpr           # The summand
    index::QuIndex         # The dummy index being summed over
    label::Symbol          # Optional label (e.g., :atoms)
end

# Constructor
Σ(expr, index) = SymbolicSum(expr, index, :default)

# Usage
i = sumindex(1)
H_int = Σ(g * a'() * σm(i), i)  # Explicit: Σᵢ g a† σ⁻(i)
```

Then implement commutator rules:

```julia
# [Σᵢ Aᵢ, B] where B has no sum over i
comm(A::SymbolicSum, B::QuExpr) = Σ(comm(A.expr, B), A.index)

# [Σᵢ Aᵢ, Σᵢ Bᵢ] - same summation variable
function comm(A::SymbolicSum, B::SymbolicSum)
    if same_sum(A, B)
        # Same-site: Σᵢ [Aᵢ, Bᵢ]
        same_site = Σ(comm(A.expr, B.expr), A.index)
        
        # Cross-site: Σᵢ Σⱼ≠ᵢ [Aᵢ, Bⱼ]
        j = fresh_sumindex()
        B_j = replace_index(B.expr, B.index, j)
        cross_site = ΣΣneq(comm(A.expr, B_j), A.index, j)  # Double sum with i≠j
        
        return same_site + cross_site
    else
        # Different sums: Σᵢ Σⱼ [Aᵢ, Bⱼ]
        return ΣΣ(comm(A.expr, B.expr), A.index, B.index)
    end
end
```

## Key Implementation Details

1. **Index replacement utility**: Need a function to replace one `QuIndex` with another in a `QuExpr`:
   ```julia
   replace_index(expr::QuExpr, old::QuIndex, new::QuIndex) -> QuExpr
   ```

2. **Double sum types**: May need `SymbolicDoubleSum` or nested sums to represent `Σᵢ Σⱼ` and `Σᵢ Σⱼ≠ᵢ`

3. **Integration with existing arithmetic**: `SymbolicSum + QuExpr`, `SymbolicSum * scalar`, etc. should work naturally

4. **Optional**: A way to "expand" a symbolic sum into explicit indices for small N:
   ```julia
   expand(Σ(σz(i), i), 1:3)  # → σz(1) + σz(2) + σz(3)
   ```

## Physics Context

This is needed for the Schrieffer-Wolff transformation in multi-atom/multi-qubit systems. For example, the Tavis-Cummings model (N atoms in a cavity) should produce the exchange interaction:

```
H_exchange = Σᵢ≠ⱼ (g²/Δ) σ⁺(i)σ⁻(j)
```

This term represents cavity-mediated spin-spin coupling and is essential for:
- Quantum gates between qubits
- Collective spin dynamics
- Superradiance

## Questions to Consider

1. Should `SymbolicSum` be a subtype of `QuExpr`, or a separate type that contains a `QuExpr`?

2. How to handle the `nsuminds` field in `QuTerm`? It's currently always 0 — maybe it should track the number of active sums?

3. Should there be syntax sugar like `@sum i expr` or `Σ[i] expr`?

4. How to print symbolic sums nicely? e.g., `Σ_#₁ g a†() σ⁻(#₁)`

## Reference: QuIndex Structure

The current `QuIndex` type has two fields:
- `sym::Char` - a symbol character
- `num::Int` - a number

| Type | Example | sym | num | Meaning |
|------|---------|-----|-----|---------|
| Integer | `σz(1)` | `'\0'` | `1` | Explicit site 1 |
| Symbol | `σz(:a)` | `'a'` | `-2147483648` | Named site "a" |
| Sum index | `σz(sumindex(1))` | `'#'` | `1` | Dummy index `#₁` |

## Reference: Working Index Replacement Code

This code works for replacing indices in expressions:

```julia
function replace_index(expr::QuExpr, old_idx::QuIndex, new_idx::QuIndex)
    result = QuExpr()
    for (term, coeff) in expr.terms
        new_ops = BaseOperator[]
        for op in term.bares.v
            new_inds = [ind == old_idx ? new_idx : ind for ind in op.inds]
            push!(new_ops, BaseOperator(op.t, op.name, new_inds, op.algebra_id, op.gen_idx))
        end
        new_term = QuTerm(
            term.nsuminds,
            term.δs,
            term.params,
            term.expvals,
            term.corrs,
            BaseOpProduct(new_ops)
        )
        result = result + coeff * QuExpr(new_term)
    end
    return result
end
```

## Test Case

After implementation, this should work:

```julia
using QuantumAlgebra
using Symbolics

@variables g Δ

i = sumindex(1)

# Define sums explicitly
S = Σ(g/Δ * (a()*σp(i) - a'()*σm(i)), i)
V = Σ(g * (a'()*σm(i) + a()*σp(i)), i)

# Commutator should produce both same-site and cross-site terms
result = comm(S, V)

# Should contain:
# - Same-site: terms like σ⁺(#₁)σ⁻(#₁), a†a, etc.
# - Cross-site: terms like σ⁺(#₁)σ⁻(#₂), σ⁺(#₂)σ⁻(#₁)  ← THE EXCHANGE INTERACTION
```
