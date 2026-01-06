# ============================================================================
# Symbolic Sums and Products for QuantumAlgebra
# ============================================================================
#
# This module provides explicit symbolic sum (Σ) and product (Π) types that
# correctly handle commutator semantics, including same-site and cross-site
# contributions.
#
# Key insight: Sum indices are BOUND VARIABLES (like in lambda calculus).
# We must handle:
# - (Σᵢ Xᵢ) * Xᵢ → Σⱼ Xⱼ Xᵢ  (rename bound index to avoid capture)
# - Σᵢ Xᵢ + Σⱼ Xⱼ → 2 Σᵢ Xᵢ  (alpha-equivalence)
# - (Σᵢ Xᵢ)³ → Σᵢⱼₖ Xᵢ Xⱼ Xₖ  (fresh indices for each factor)
# ============================================================================

export SymSum, SymProd, SymExpr
export symbolic_sum, symbolic_prod
export expand_symbolic, free_indices, canonicalize_indices
export simplify_sums

# ============================================================================
# Fresh Index Counter (Thread-Safe)
# ============================================================================

const _FRESH_SUMINDEX_COUNTER = Threads.Atomic{Int}(1000)

"""
    fresh_sumindex()

Generate a new unique sum index that won't conflict with any existing indices.
Uses a high starting number to avoid conflicts with user-created indices.
"""
function fresh_sumindex()
    n = Threads.atomic_add!(_FRESH_SUMINDEX_COUNTER, 1)
    sumindex(n)
end

"""
    reset_fresh_sumindex_counter!(n=1000)

Reset the fresh sum index counter. Useful for testing.
"""
function reset_fresh_sumindex_counter!(n::Int=1000)
    Threads.atomic_xchg!(_FRESH_SUMINDEX_COUNTER, n)
    nothing
end

# ============================================================================
# Core Types
# ============================================================================

# Forward declarations for mutual recursion
abstract type AbstractSymbolicAggregate end

"""
    SymSum

Represents a symbolic sum Σᵢ expr(i) over a bound index i.

Fields:
- `expr`: The summand (can be QuExpr, SymSum, or SymProd for nested sums)
- `index`: The bound summation index (must be a sum index)
- `excluded`: Indices that must differ from this one (for i≠j constraints)

# Example
```julia
i = sumindex(1)
s = SymSum(σz(i), i)  # Represents Σᵢ σz(i)
```
"""
struct SymSum <: AbstractSymbolicAggregate
    expr::Union{QuExpr, AbstractSymbolicAggregate}
    index::QuIndex
    excluded::Vector{QuIndex}
    
    function SymSum(expr, index::QuIndex, excluded::Vector{QuIndex}=QuIndex[])
        issumindex(index) || error("SymSum index must be a sum index (created with sumindex(n)), got $index")
        new(expr, index, excluded)
    end
end

"""
    SymProd

Represents a symbolic product Πᵢ expr(i) over a bound index i.

Fields:
- `expr`: The factor (can be QuExpr, SymSum, or SymProd)
- `index`: The bound product index (must be a sum index)
- `excluded`: Indices that must differ from this one (for i≠j constraints)
"""
struct SymProd <: AbstractSymbolicAggregate
    expr::Union{QuExpr, AbstractSymbolicAggregate}
    index::QuIndex
    excluded::Vector{QuIndex}
    
    function SymProd(expr, index::QuIndex, excluded::Vector{QuIndex}=QuIndex[])
        issumindex(index) || error("SymProd index must be a sum index (created with sumindex(n)), got $index")
        new(expr, index, excluded)
    end
end

"""
    SymExpr

A general symbolic expression containing:
- Multiple symbolic sum/product terms with coefficients
- Regular QuExpr terms without symbolic aggregation

This provides a closed algebra where all operations return a SymExpr.
"""
struct SymExpr
    # (coefficient, symbolic aggregate) pairs
    terms::Vector{Tuple{Number, AbstractSymbolicAggregate}}
    # Regular (non-symbolic-aggregate) terms
    scalar::QuExpr
    
    function SymExpr(terms::Vector{<:Tuple{Number, AbstractSymbolicAggregate}}, scalar::QuExpr)
        # Filter out zero coefficients
        filtered = filter(t -> !iszero(t[1]), terms)
        new(filtered, scalar)
    end
end

# Convenience constructors
SymExpr() = SymExpr(Tuple{Number, AbstractSymbolicAggregate}[], zero(QuExpr))
SymExpr(s::AbstractSymbolicAggregate) = SymExpr([(1, s)], zero(QuExpr))
SymExpr(e::QuExpr) = SymExpr(Tuple{Number, AbstractSymbolicAggregate}[], e)
SymExpr(n::Number) = SymExpr(QuExpr(n))

# Unicode aliases
const Σₛ = SymSum  # Σₛ to distinguish from existing ∑
const Πₛ = SymProd

"""
    symbolic_sum(expr, index, [excluded])

Create a symbolic sum Σᵢ expr over the given index.
ASCII alternative to SymSum constructor.
"""
symbolic_sum(expr, index::QuIndex, excluded::Vector{QuIndex}=QuIndex[]) = SymSum(expr, index, excluded)
symbolic_sum(expr::Number, index::QuIndex, excluded::Vector{QuIndex}=QuIndex[]) = SymSum(QuExpr(expr), index, excluded)

"""
    symbolic_prod(expr, index, [excluded])

Create a symbolic product Πᵢ expr over the given index.
ASCII alternative to SymProd constructor.
"""
symbolic_prod(expr, index::QuIndex, excluded::Vector{QuIndex}=QuIndex[]) = SymProd(expr, index, excluded)
symbolic_prod(expr::Number, index::QuIndex, excluded::Vector{QuIndex}=QuIndex[]) = SymProd(QuExpr(expr), index, excluded)

# ============================================================================
# Free and Bound Index Analysis
# ============================================================================

"""
    free_indices(expr) -> Set{QuIndex}

Return the set of indices that appear FREE (not bound by any symbolic sum/product)
in the expression. Sum indices that are bound by QuTerm.nsuminds are NOT considered
free by this function.
"""
function free_indices(expr::QuExpr)
    # Get all indices, but filter out sum indices that are bound by nsuminds
    all_inds = Set{QuIndex}()
    for (term, _) in expr.terms
        # Indices bound by nsuminds are #1, #2, ..., #nsuminds
        bound_by_term = Set(sumindex(i) for i in 1:term.nsuminds)
        term_inds = Set(indices(term))
        # Free = all indices minus those bound by nsuminds
        union!(all_inds, setdiff(term_inds, bound_by_term))
    end
    return all_inds
end

function free_indices(s::SymSum)
    inner_free = free_indices(s.expr)
    delete!(inner_free, s.index)  # This index is bound by the sum
    # Excluded indices refer to outer scope, so they're free from this sum's perspective
    union!(inner_free, Set(s.excluded))
    return inner_free
end

function free_indices(s::SymProd)
    inner_free = free_indices(s.expr)
    delete!(inner_free, s.index)  # This index is bound by the product
    union!(inner_free, Set(s.excluded))
    return inner_free
end

function free_indices(e::SymExpr)
    result = free_indices(e.scalar)
    for (_, agg) in e.terms
        union!(result, free_indices(agg))
    end
    return result
end

"""
    has_free_index(expr, idx::QuIndex) -> Bool

Check if idx appears as a FREE variable in expr.
"""
has_free_index(expr, idx::QuIndex) = idx in free_indices(expr)

# ============================================================================
# Index Replacement
# ============================================================================

"""
    replace_sym_index(expr, old_idx::QuIndex, new_idx::QuIndex)

Replace all occurrences of `old_idx` with `new_idx` in the expression.
This is a deep replacement that handles nested structures.
"""
function replace_sym_index(expr::QuExpr, old_idx::QuIndex, new_idx::QuIndex)
    old_idx == new_idx && return expr
    f = replace_inds(old_idx => new_idx)
    f(expr)
end

function replace_sym_index(s::SymSum, old_idx::QuIndex, new_idx::QuIndex)
    old_idx == new_idx && return s
    new_expr = replace_sym_index(s.expr, old_idx, new_idx)
    new_index = s.index == old_idx ? new_idx : s.index
    new_excluded = [idx == old_idx ? new_idx : idx for idx in s.excluded]
    SymSum(new_expr, new_index, new_excluded)
end

function replace_sym_index(s::SymProd, old_idx::QuIndex, new_idx::QuIndex)
    old_idx == new_idx && return s
    new_expr = replace_sym_index(s.expr, old_idx, new_idx)
    new_index = s.index == old_idx ? new_idx : s.index
    new_excluded = [idx == old_idx ? new_idx : idx for idx in s.excluded]
    SymProd(new_expr, new_index, new_excluded)
end

function replace_sym_index(e::SymExpr, old_idx::QuIndex, new_idx::QuIndex)
    old_idx == new_idx && return e
    new_terms = [(c, replace_sym_index(agg, old_idx, new_idx)) for (c, agg) in e.terms]
    new_scalar = replace_sym_index(e.scalar, old_idx, new_idx)
    SymExpr(new_terms, new_scalar)
end

# ============================================================================
# Ensure Fresh Index (Rename if Conflict)
# ============================================================================

"""
    ensure_fresh_index(s::AbstractSymbolicAggregate, conflicting_indices::Set{QuIndex})

If s.index is in conflicting_indices, rename it to a fresh index.
Returns the (possibly renamed) aggregate.
"""
function ensure_fresh_index(s::SymSum, conflicting::Set{QuIndex})
    if s.index in conflicting
        new_idx = fresh_sumindex()
        return replace_sym_index(s, s.index, new_idx)
    end
    return s
end

function ensure_fresh_index(s::SymProd, conflicting::Set{QuIndex})
    if s.index in conflicting
        new_idx = fresh_sumindex()
        return replace_sym_index(s, s.index, new_idx)
    end
    return s
end

# ============================================================================
# Negation
# ============================================================================

Base.:-(s::SymSum) = SymSum(-s.expr, s.index, s.excluded)
Base.:-(s::SymProd) = SymProd(-s.expr, s.index, s.excluded)
Base.:-(e::SymExpr) = SymExpr([(-c, agg) for (c, agg) in e.terms], -e.scalar)

# ============================================================================
# Scalar Multiplication
# ============================================================================

Base.:*(c::Number, s::SymSum) = iszero(c) ? SymExpr() : SymSum(c * s.expr, s.index, s.excluded)
Base.:*(s::SymSum, c::Number) = c * s
Base.:*(c::Number, s::SymProd) = iszero(c) ? SymExpr() : SymProd(c * s.expr, s.index, s.excluded)
Base.:*(s::SymProd, c::Number) = c * s

function Base.:*(c::Number, e::SymExpr)
    iszero(c) && return SymExpr()
    SymExpr([(c * coeff, agg) for (coeff, agg) in e.terms], c * e.scalar)
end
Base.:*(e::SymExpr, c::Number) = c * e

# ============================================================================
# Addition
# ============================================================================

# SymSum + SymSum
function Base.:+(a::SymSum, b::SymSum)
    # For now, just collect into SymExpr
    # simplify_sums() can combine alpha-equivalent terms later
    SymExpr([(1, a), (1, b)], zero(QuExpr))
end

# SymProd + SymProd
function Base.:+(a::SymProd, b::SymProd)
    SymExpr([(1, a), (1, b)], zero(QuExpr))
end

# Mixed aggregates
Base.:+(a::SymSum, b::SymProd) = SymExpr([(1, a), (1, b)], zero(QuExpr))
Base.:+(a::SymProd, b::SymSum) = b + a

# Aggregate + QuExpr
Base.:+(a::AbstractSymbolicAggregate, b::QuExpr) = SymExpr([(1, a)], b)
Base.:+(a::QuExpr, b::AbstractSymbolicAggregate) = b + a

# Aggregate + Number
Base.:+(a::AbstractSymbolicAggregate, b::Number) = a + QuExpr(b)
Base.:+(a::Number, b::AbstractSymbolicAggregate) = b + a

# SymExpr + SymExpr
function Base.:+(a::SymExpr, b::SymExpr)
    new_terms = vcat(a.terms, b.terms)
    new_scalar = a.scalar + b.scalar
    SymExpr(new_terms, new_scalar)
end

# SymExpr + Aggregate
function Base.:+(a::SymExpr, b::AbstractSymbolicAggregate)
    SymExpr(vcat(a.terms, [(1, b)]), a.scalar)
end
Base.:+(a::AbstractSymbolicAggregate, b::SymExpr) = b + a

# SymExpr + QuExpr
Base.:+(a::SymExpr, b::QuExpr) = SymExpr(a.terms, a.scalar + b)
Base.:+(a::QuExpr, b::SymExpr) = b + a

# SymExpr + Number
Base.:+(a::SymExpr, b::Number) = a + QuExpr(b)
Base.:+(a::Number, b::SymExpr) = b + a

# ============================================================================
# Subtraction
# ============================================================================

Base.:-(a::AbstractSymbolicAggregate, b::AbstractSymbolicAggregate) = a + (-b)
Base.:-(a::AbstractSymbolicAggregate, b::QuExpr) = a + (-b)
Base.:-(a::QuExpr, b::AbstractSymbolicAggregate) = a + (-b)
Base.:-(a::AbstractSymbolicAggregate, b::Number) = a + (-b)
Base.:-(a::Number, b::AbstractSymbolicAggregate) = a + (-b)
Base.:-(a::SymExpr, b::SymExpr) = a + (-b)
Base.:-(a::SymExpr, b::AbstractSymbolicAggregate) = a + (-b)
Base.:-(a::AbstractSymbolicAggregate, b::SymExpr) = a + (-b)
Base.:-(a::SymExpr, b::QuExpr) = a + (-b)
Base.:-(a::QuExpr, b::SymExpr) = a + (-b)
Base.:-(a::SymExpr, b::Number) = a + (-b)
Base.:-(a::Number, b::SymExpr) = a + (-b)

# ============================================================================
# Multiplication: The Key Part with Bound Variable Handling
# ============================================================================

"""
    *(s::SymSum, b::QuExpr)

Multiply a symbolic sum by an expression.
If b contains the sum index as a FREE variable, rename the bound index first.

(Σᵢ Aᵢ) * B = Σᵢ (Aᵢ * B)  if i ∉ free(B)
(Σᵢ Aᵢ) * Bᵢ = Σⱼ (Aⱼ * Bᵢ)  if i ∈ free(B) (rename to avoid capture)
"""
function Base.:*(s::SymSum, b::QuExpr)
    if has_free_index(b, s.index)
        # Rename bound index to avoid capture
        s = ensure_fresh_index(s, Set([s.index]))
    end
    SymSum(s.expr * b, s.index, s.excluded)
end

function Base.:*(a::QuExpr, s::SymSum)
    if has_free_index(a, s.index)
        s = ensure_fresh_index(s, Set([s.index]))
    end
    SymSum(a * s.expr, s.index, s.excluded)
end

# Same for SymProd
function Base.:*(s::SymProd, b::QuExpr)
    if has_free_index(b, s.index)
        s = ensure_fresh_index(s, Set([s.index]))
    end
    SymProd(s.expr * b, s.index, s.excluded)
end

function Base.:*(a::QuExpr, s::SymProd)
    if has_free_index(a, s.index)
        s = ensure_fresh_index(s, Set([s.index]))
    end
    SymProd(a * s.expr, s.index, s.excluded)
end

"""
    *(a::SymSum, b::SymSum)

Multiply two symbolic sums. Always use fresh indices to avoid any conflicts.

(Σᵢ Aᵢ) * (Σⱼ Bⱼ) = Σₖ Σₗ (Aₖ * Bₗ)
"""
function Base.:*(a::SymSum, b::SymSum)
    # Get all free indices from both expressions to avoid conflicts
    all_free = union(free_indices(a), free_indices(b))
    
    # Ensure both have fresh indices
    a = ensure_fresh_index(a, union(all_free, Set([b.index])))
    b = ensure_fresh_index(b, union(all_free, Set([a.index])))
    
    # Nested sum: Σᵢ Σⱼ (Aᵢ * Bⱼ)
    inner = SymSum(a.expr * b.expr, b.index, b.excluded)
    SymSum(inner, a.index, a.excluded)
end

# SymSum * SymProd -> SymSum containing SymProd
function Base.:*(a::SymSum, b::SymProd)
    all_free = union(free_indices(a), free_indices(b))
    a = ensure_fresh_index(a, union(all_free, Set([b.index])))
    b = ensure_fresh_index(b, union(all_free, Set([a.index])))
    
    inner = SymProd(a.expr * b.expr, b.index, b.excluded)
    SymSum(inner, a.index, a.excluded)
end

function Base.:*(a::SymProd, b::SymSum)
    all_free = union(free_indices(a), free_indices(b))
    a = ensure_fresh_index(a, union(all_free, Set([b.index])))
    b = ensure_fresh_index(b, union(all_free, Set([a.index])))
    
    inner = SymSum(a.expr * b.expr, b.index, b.excluded)
    SymProd(inner, a.index, a.excluded)
end

function Base.:*(a::SymProd, b::SymProd)
    all_free = union(free_indices(a), free_indices(b))
    a = ensure_fresh_index(a, union(all_free, Set([b.index])))
    b = ensure_fresh_index(b, union(all_free, Set([a.index])))
    
    inner = SymProd(a.expr * b.expr, b.index, b.excluded)
    SymProd(inner, a.index, a.excluded)
end

# SymExpr multiplication - distribute
function Base.:*(a::SymExpr, b::QuExpr)
    new_terms = [(c, agg * b) for (c, agg) in a.terms]
    new_scalar = a.scalar * b
    SymExpr(new_terms, new_scalar)
end
Base.:*(a::QuExpr, b::SymExpr) = b * a

function Base.:*(a::SymExpr, b::AbstractSymbolicAggregate)
    # (Σ terms + scalar) * b = Σ(terms * b) + scalar * b
    new_terms = Tuple{Number, AbstractSymbolicAggregate}[]
    
    # Each term * b
    for (c, agg) in a.terms
        result = agg * b
        if result isa AbstractSymbolicAggregate
            push!(new_terms, (c, result))
        elseif result isa SymExpr
            for (c2, agg2) in result.terms
                push!(new_terms, (c * c2, agg2))
            end
            # Note: result.scalar should be zero for aggregate * aggregate
        end
    end
    
    # scalar * b
    if !iszero(a.scalar)
        scalar_result = a.scalar * b
        if scalar_result isa AbstractSymbolicAggregate
            push!(new_terms, (1, scalar_result))
        elseif scalar_result isa SymExpr
            append!(new_terms, scalar_result.terms)
        end
    end
    
    SymExpr(new_terms, zero(QuExpr))
end

Base.:*(a::AbstractSymbolicAggregate, b::SymExpr) = b * a

function Base.:*(a::SymExpr, b::SymExpr)
    result = SymExpr()
    
    # Distribute: (terms_a + scalar_a) * (terms_b + scalar_b)
    for (ca, agga) in a.terms
        for (cb, aggb) in b.terms
            prod = agga * aggb
            if prod isa AbstractSymbolicAggregate
                result = result + SymExpr([(ca * cb, prod)], zero(QuExpr))
            else
                result = result + (ca * cb) * prod
            end
        end
        if !iszero(b.scalar)
            prod = agga * b.scalar
            if prod isa AbstractSymbolicAggregate
                result = result + SymExpr([(ca, prod)], zero(QuExpr))
            end
        end
    end
    
    if !iszero(a.scalar)
        for (cb, aggb) in b.terms
            prod = a.scalar * aggb
            if prod isa AbstractSymbolicAggregate
                result = result + SymExpr([(cb, prod)], zero(QuExpr))
            end
        end
        result = result + SymExpr(a.scalar * b.scalar)
    end
    
    result
end

# ============================================================================
# Power
# ============================================================================

function Base.:^(s::AbstractSymbolicAggregate, n::Integer)
    n < 0 && error("Negative powers not supported for symbolic sums/products")
    n == 0 && return SymExpr(one(QuExpr))
    n == 1 && return s
    
    # (Σᵢ Xᵢ)^n = Σᵢ₁ Σᵢ₂ ... Σᵢₙ (Xᵢ₁ * Xᵢ₂ * ... * Xᵢₙ)
    result = s
    for _ in 2:n
        result = result * s
    end
    result
end

function Base.:^(e::SymExpr, n::Integer)
    n < 0 && error("Negative powers not supported")
    n == 0 && return SymExpr(one(QuExpr))
    n == 1 && return e
    
    result = e
    for _ in 2:n
        result = result * e
    end
    result
end

# ============================================================================
# Commutator
# ============================================================================

"""
    comm(A::SymSum, B::QuExpr)

Commutator [Σᵢ Aᵢ, B] = Σᵢ [Aᵢ, B]
"""
function comm(A::SymSum, B::QuExpr)
    if has_free_index(B, A.index)
        A = ensure_fresh_index(A, Set([A.index]))
    end
    SymSum(comm(A.expr, B), A.index, A.excluded)
end

comm(A::QuExpr, B::SymSum) = -comm(B, A)

"""
    comm(A::SymSum, B::SymSum)

Commutator [Σᵢ Aᵢ, Σⱼ Bⱼ].

If the sums are over the same index (same bound variable), produces:
- Same-site term: Σᵢ [Aᵢ, Bᵢ]
- Cross-site term: Σᵢ Σⱼ≠ᵢ [Aᵢ, Bⱼ]

If the sums are over different indices:
- Σᵢ Σⱼ [Aᵢ, Bⱼ]
"""
function comm(A::SymSum, B::SymSum)
    # We always treat the sums as "conceptually the same" and split into same/cross
    # The user created both sums to represent sums over the same physical set
    
    # Ensure both have distinct indices
    all_free = union(free_indices(A), free_indices(B))
    A = ensure_fresh_index(A, union(all_free, Set([B.index])))
    
    # Now A.index and B.index are different (or were different to begin with)
    
    # Same-site contribution: Σᵢ [Aᵢ, Bᵢ]
    # We rename B's index to match A's for the same-site term
    B_same = replace_sym_index(B.expr, B.index, A.index)
    same_site_comm = comm(A.expr, B_same)
    same_site = SymSum(same_site_comm, A.index, A.excluded)
    
    # Cross-site contribution: Σᵢ Σⱼ≠ᵢ [Aᵢ, Bⱼ]
    j = fresh_sumindex()
    B_cross = replace_sym_index(B.expr, B.index, j)
    cross_site_comm = comm(A.expr, B_cross)
    
    # Inner sum over j with j≠i constraint
    inner_sum = SymSum(cross_site_comm, j, [A.index])
    # Outer sum over i
    cross_site = SymSum(inner_sum, A.index, A.excluded)
    
    # Return both contributions as a SymExpr
    SymExpr([(1, same_site), (1, cross_site)], zero(QuExpr))
end

# Commutator with SymExpr - distribute
function comm(A::SymExpr, B::QuExpr)
    result = SymExpr()
    for (c, agg) in A.terms
        result = result + c * comm(agg, B)
    end
    if !iszero(A.scalar)
        result = result + SymExpr(comm(A.scalar, B))
    end
    result
end

comm(A::QuExpr, B::SymExpr) = -comm(B, A)

function comm(A::SymExpr, B::SymExpr)
    result = SymExpr()
    
    for (ca, agga) in A.terms
        for (cb, aggb) in B.terms
            c_result = comm(agga, aggb)
            result = result + (ca * cb) * c_result
        end
        if !iszero(B.scalar)
            result = result + ca * comm(agga, B.scalar)
        end
    end
    
    if !iszero(A.scalar)
        for (cb, aggb) in B.terms
            result = result + cb * comm(A.scalar, aggb)
        end
        if !iszero(B.scalar)
            result = result + SymExpr(comm(A.scalar, B.scalar))
        end
    end
    
    result
end

function comm(A::AbstractSymbolicAggregate, B::SymExpr)
    comm(SymExpr(A), B)
end

function comm(A::SymExpr, B::AbstractSymbolicAggregate)
    comm(A, SymExpr(B))
end

# ============================================================================
# Expansion
# ============================================================================

"""
    expand_symbolic(s::SymSum, range) -> QuExpr

Expand a symbolic sum into an explicit sum over the given range.

# Example
```julia
i = sumindex(1)
expand_symbolic(SymSum(σz(i), i), 1:3)  # → σz(1) + σz(2) + σz(3)
```
"""
function expand_symbolic(s::SymSum, range; excluded_values::Dict{QuIndex,Any}=Dict{QuIndex,Any}())
    result = zero(QuExpr)
    
    for val in range
        # Check static exclusions (integer indices)
        skip = false
        for excl in s.excluded
            if isintindex(excl) && excl.num == val
                skip = true
                break
            elseif haskey(excluded_values, excl) && excluded_values[excl] == val
                skip = true
                break
            end
        end
        skip && continue
        
        # Replace sum index with concrete value
        concrete_idx = QuIndex(val)
        term = replace_sym_index(s.expr, s.index, concrete_idx)
        
        # Recursively expand nested symbolic sums
        if term isa SymSum
            new_excluded = copy(excluded_values)
            new_excluded[s.index] = val
            term = expand_symbolic(term, range; excluded_values=new_excluded)
        elseif term isa SymProd
            new_excluded = copy(excluded_values)
            new_excluded[s.index] = val
            term = expand_symbolic(term, range; excluded_values=new_excluded)
        elseif term isa SymExpr
            new_excluded = copy(excluded_values)
            new_excluded[s.index] = val
            term = expand_symbolic(term, range; excluded_values=new_excluded)
        end
        
        result = result + term
    end
    
    return result
end

"""
    expand_symbolic(s::SymProd, range) -> QuExpr

Expand a symbolic product into an explicit product over the given range.
"""
function expand_symbolic(s::SymProd, range; excluded_values::Dict{QuIndex,Any}=Dict{QuIndex,Any}())
    result = one(QuExpr)
    
    for val in range
        # Check exclusions
        skip = false
        for excl in s.excluded
            if isintindex(excl) && excl.num == val
                skip = true
                break
            elseif haskey(excluded_values, excl) && excluded_values[excl] == val
                skip = true
                break
            end
        end
        skip && continue
        
        # Replace index with concrete value
        concrete_idx = QuIndex(val)
        term = replace_sym_index(s.expr, s.index, concrete_idx)
        
        # Recursively expand
        if term isa AbstractSymbolicAggregate || term isa SymExpr
            new_excluded = copy(excluded_values)
            new_excluded[s.index] = val
            term = expand_symbolic(term, range; excluded_values=new_excluded)
        end
        
        result = result * term
    end
    
    return result
end

"""
    expand_symbolic(e::SymExpr, range) -> QuExpr

Expand all symbolic sums/products in a SymExpr.
"""
function expand_symbolic(e::SymExpr, range; excluded_values::Dict{QuIndex,Any}=Dict{QuIndex,Any}())
    result = e.scalar
    for (c, agg) in e.terms
        expanded = expand_symbolic(agg, range; excluded_values=excluded_values)
        result = result + c * expanded
    end
    return result
end

# ============================================================================
# Canonicalization and Alpha-Equivalence
# ============================================================================

"""
    canonicalize_indices(s::AbstractSymbolicAggregate) -> AbstractSymbolicAggregate

Rename all bound indices to a canonical form (sumindex(1), sumindex(2), ...).
This makes comparison of alpha-equivalent expressions easier.
"""
function canonicalize_indices(s::SymSum, next_idx::Ref{Int}=Ref(1))
    new_idx = sumindex(next_idx[])
    next_idx[] += 1
    
    # Replace index in expression
    new_expr = replace_sym_index(s.expr, s.index, new_idx)
    
    # Recursively canonicalize nested aggregates
    if new_expr isa AbstractSymbolicAggregate
        new_expr = canonicalize_indices(new_expr, next_idx)
    end
    
    # Update excluded indices too (they refer to outer scopes, but we canonicalize consistently)
    new_excluded = s.excluded  # Keep as-is for now; outer scope will rename
    
    SymSum(new_expr, new_idx, new_excluded)
end

function canonicalize_indices(s::SymProd, next_idx::Ref{Int}=Ref(1))
    new_idx = sumindex(next_idx[])
    next_idx[] += 1
    
    new_expr = replace_sym_index(s.expr, s.index, new_idx)
    
    if new_expr isa AbstractSymbolicAggregate
        new_expr = canonicalize_indices(new_expr, next_idx)
    end
    
    SymProd(new_expr, new_idx, s.excluded)
end

"""
    alpha_equivalent(a::SymSum, b::SymSum) -> Bool

Check if two symbolic sums are equal up to renaming of bound variables.
"""
function alpha_equivalent(a::SymSum, b::SymSum)
    # Must have same exclusion structure
    length(a.excluded) != length(b.excluded) && return false
    
    # Canonicalize both and compare
    a_canon = canonicalize_indices(a)
    b_canon = canonicalize_indices(b)
    
    return a_canon.expr == b_canon.expr && a_canon.excluded == b_canon.excluded
end

function alpha_equivalent(a::SymProd, b::SymProd)
    length(a.excluded) != length(b.excluded) && return false
    a_canon = canonicalize_indices(a)
    b_canon = canonicalize_indices(b)
    return a_canon.expr == b_canon.expr && a_canon.excluded == b_canon.excluded
end

# Different types are never alpha-equivalent
alpha_equivalent(a::SymSum, b::SymProd) = false
alpha_equivalent(a::SymProd, b::SymSum) = false

# ============================================================================
# Simplification
# ============================================================================

"""
    simplify_sums(e::SymExpr) -> SymExpr

Combine alpha-equivalent terms in a SymExpr.
For example: Σᵢ Xᵢ + Σⱼ Xⱼ → 2 Σᵢ Xᵢ
"""
function simplify_sums(e::SymExpr)
    isempty(e.terms) && return e
    
    # Group alpha-equivalent terms
    groups = Vector{Tuple{Number, AbstractSymbolicAggregate}}[]
    
    for (c, agg) in e.terms
        found = false
        for group in groups
            if alpha_equivalent(group[1][2], agg)
                push!(group, (c, agg))
                found = true
                break
            end
        end
        if !found
            push!(groups, [(c, agg)])
        end
    end
    
    # Sum coefficients within each group
    new_terms = Tuple{Number, AbstractSymbolicAggregate}[]
    for group in groups
        total_coeff = sum(c for (c, _) in group)
        if !iszero(total_coeff)
            # Use the first (canonical) representative
            push!(new_terms, (total_coeff, group[1][2]))
        end
    end
    
    SymExpr(new_terms, e.scalar)
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, s::SymSum)
    # Canonicalize for display
    s_display = canonicalize_indices(s)
    
    print(io, "Σ")
    print(io, s_display.index)
    if !isempty(s_display.excluded)
        print(io, "≠")
        join(io, s_display.excluded, ",")
    end
    print(io, "(")
    show(io, s_display.expr)
    print(io, ")")
end

function Base.show(io::IO, s::SymProd)
    s_display = canonicalize_indices(s)
    
    print(io, "Π")
    print(io, s_display.index)
    if !isempty(s_display.excluded)
        print(io, "≠")
        join(io, s_display.excluded, ",")
    end
    print(io, "(")
    show(io, s_display.expr)
    print(io, ")")
end

function Base.show(io::IO, e::SymExpr)
    first = true
    
    for (c, agg) in e.terms
        if first
            if c == 1
                show(io, agg)
            elseif c == -1
                print(io, "-")
                show(io, agg)
            else
                print(io, c, "*")
                show(io, agg)
            end
            first = false
        else
            if c == 1
                print(io, " + ")
                show(io, agg)
            elseif c == -1
                print(io, " - ")
                show(io, agg)
            elseif real(c) < 0
                print(io, " - ", -c, "*")
                show(io, agg)
            else
                print(io, " + ", c, "*")
                show(io, agg)
            end
        end
    end
    
    if !iszero(e.scalar)
        if first
            show(io, e.scalar)
        else
            print(io, " + ")
            show(io, e.scalar)
        end
    elseif first
        print(io, "0")
    end
end

# ============================================================================
# LaTeX Output
# ============================================================================

function latex(s::SymSum)
    s_display = canonicalize_indices(s)
    idx_str = latex(s_display.index)
    
    excl_str = ""
    if !isempty(s_display.excluded)
        excl_strs = [latex(e) for e in s_display.excluded]
        excl_str = "_{\\neq " * join(excl_strs, ",") * "}"
    end
    
    expr_str = latex(s_display.expr)
    
    "\\sum_{$idx_str$excl_str} $expr_str"
end

function latex(s::SymProd)
    s_display = canonicalize_indices(s)
    idx_str = latex(s_display.index)
    
    excl_str = ""
    if !isempty(s_display.excluded)
        excl_strs = [latex(e) for e in s_display.excluded]
        excl_str = "_{\\neq " * join(excl_strs, ",") * "}"
    end
    
    expr_str = latex(s_display.expr)
    
    "\\prod_{$idx_str$excl_str} $expr_str"
end

function latex(e::SymExpr)
    parts = String[]
    
    for (i, (c, agg)) in enumerate(e.terms)
        agg_latex = latex(agg)
        if i == 1
            if c == 1
                push!(parts, agg_latex)
            elseif c == -1
                push!(parts, "-$agg_latex")
            else
                push!(parts, "$(numlatex(c)) $agg_latex")
            end
        else
            if c == 1
                push!(parts, " + $agg_latex")
            elseif c == -1
                push!(parts, " - $agg_latex")
            elseif real(c) < 0
                push!(parts, " - $(numlatex(-c)) $agg_latex")
            else
                push!(parts, " + $(numlatex(c)) $agg_latex")
            end
        end
    end
    
    if !iszero(e.scalar)
        scalar_latex = latex(e.scalar)
        if isempty(parts)
            push!(parts, scalar_latex)
        else
            push!(parts, " + $scalar_latex")
        end
    elseif isempty(parts)
        return "0"
    end
    
    join(parts)
end

# ============================================================================
# Symbolic Algebra Integration
# ============================================================================

"""
    map_scalar_function(f, s::AbstractSymbolicAggregate)

Apply a scalar function to all scalar coefficients in the expression.
This enables integration with Symbolics.jl, SymPy, etc.
"""
function map_scalar_function(f, s::SymSum)
    new_expr = map_scalar_function(f, s.expr)
    SymSum(new_expr, s.index, s.excluded)
end

function map_scalar_function(f, s::SymProd)
    new_expr = map_scalar_function(f, s.expr)
    SymProd(new_expr, s.index, s.excluded)
end

function map_scalar_function(f, e::SymExpr)
    new_terms = [(f(c), agg isa AbstractSymbolicAggregate ? map_scalar_function(f, agg) : agg) for (c, agg) in e.terms]
    new_scalar = map_scalar_function(f, e.scalar)
    SymExpr(new_terms, new_scalar)
end

# ============================================================================
# Conversion Utilities
# ============================================================================

"""
    normal_form(s::AbstractSymbolicAggregate)

Apply normal_form to the expression inside a symbolic aggregate.
"""
normal_form(s::SymSum) = SymSum(normal_form(s.expr), s.index, s.excluded)
normal_form(s::SymProd) = SymProd(normal_form(s.expr), s.index, s.excluded)

function normal_form(e::SymExpr)
    new_terms = [(c, normal_form(agg)) for (c, agg) in e.terms]
    new_scalar = normal_form(e.scalar)
    SymExpr(new_terms, new_scalar)
end

# ============================================================================
# Equality
# ============================================================================

function Base.:(==)(a::SymSum, b::SymSum)
    alpha_equivalent(a, b)
end

function Base.:(==)(a::SymProd, b::SymProd)
    alpha_equivalent(a, b)
end

Base.:(==)(a::SymSum, b::SymProd) = false
Base.:(==)(a::SymProd, b::SymSum) = false

function Base.:(==)(a::SymExpr, b::SymExpr)
    # Simplify both and compare
    a_simp = simplify_sums(a)
    b_simp = simplify_sums(b)
    
    # Compare scalars
    a_simp.scalar != b_simp.scalar && return false
    
    # Compare number of terms
    length(a_simp.terms) != length(b_simp.terms) && return false
    
    # For each term in a, find a matching term in b
    b_used = falses(length(b_simp.terms))
    for (ca, agga) in a_simp.terms
        found = false
        for (j, (cb, aggb)) in enumerate(b_simp.terms)
            if !b_used[j] && ca == cb && alpha_equivalent(agga, aggb)
                b_used[j] = true
                found = true
                break
            end
        end
        !found && return false
    end
    
    return true
end

# ============================================================================
# Zero and One
# ============================================================================

Base.iszero(s::AbstractSymbolicAggregate) = iszero(s.expr)
Base.iszero(e::SymExpr) = isempty(e.terms) && iszero(e.scalar)

Base.zero(::Type{SymExpr}) = SymExpr()
Base.one(::Type{SymExpr}) = SymExpr(one(QuExpr))
