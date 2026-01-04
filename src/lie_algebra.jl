# Lie algebra infrastructure for QuantumAlgebra.jl
# Supports SU(N) algebras with structure constants and generator products

using LinearAlgebra: tr

"""
    AbstractLieAlgebra

Abstract type for Lie algebras. Concrete subtypes must implement:
- `algebra_dim(alg)`: dimension N of the algebra (e.g., N for SU(N))
- `num_generators(alg)`: number of generators (N²-1 for SU(N))
- `structure_constants(alg, a, b)`: returns vector of (c, f_abc) pairs for [T^a, T^b] = i f^{abc} T^c
- `product_coefficients(alg, a, b)`: returns (identity_coeff, [(c, coeff), ...]) for T^a T^b
"""
abstract type AbstractLieAlgebra end

"""
    SUAlgebra{N}

Represents the SU(N) Lie algebra with N²-1 generators.
Uses the generalized Gell-Mann matrix convention with normalization Tr(T^a T^b) = δ_{ab}/2.

The generators satisfy:
- [T^a, T^b] = i f^{abc} T^c  (structure constants, antisymmetric)
- T^a T^b = (1/2N) δ_{ab} I + (1/2)(d^{abc} + i f^{abc}) T^c

Note: Structure constants are stored as Float64 since they can be irrational (e.g., sqrt(3)/2 for SU(3)).
"""
struct SUAlgebra{N} <: AbstractLieAlgebra
    # Cached structure constants: f[a,b] = Dict(c => f_abc) for non-zero entries
    f::Matrix{Dict{Int, Float64}}
    # Cached symmetric structure constants: d[a,b] = Dict(c => d_abc) for non-zero entries  
    d::Matrix{Dict{Int, Float64}}
    
    function SUAlgebra{N}() where N
        N >= 2 || throw(ArgumentError("SU(N) requires N >= 2, got N=$N"))
        ngen = N^2 - 1
        f = [Dict{Int, Float64}() for _ in 1:ngen, _ in 1:ngen]
        d = [Dict{Int, Float64}() for _ in 1:ngen, _ in 1:ngen]
        alg = new{N}(f, d)
        _compute_structure_constants!(alg)
        alg
    end
end

algebra_dim(::SUAlgebra{N}) where N = N
num_generators(::SUAlgebra{N}) where N = N^2 - 1

"""
    gellmann_matrix(N, k)

Compute the k-th generalized Gell-Mann matrix for SU(N).
Returns an N×N complex Float64 matrix with normalization Tr(T^a T^b) = δ_{ab}/2.

The generators are ordered as:
1. Symmetric off-diagonal: (N-1)*N/2 matrices
2. Antisymmetric off-diagonal: (N-1)*N/2 matrices  
3. Diagonal: N-1 matrices

For SU(2), this gives the Pauli matrices σ_x, σ_y, σ_z (divided by 2).
For SU(3), this gives the Gell-Mann matrices λ_1, ..., λ_8 (divided by 2).
"""
function gellmann_matrix(N::Int, k::Int)
    ngen = N^2 - 1
    1 <= k <= ngen || throw(ArgumentError("Generator index k=$k out of range 1:$ngen for SU($N)"))
    
    T = zeros(ComplexF64, N, N)
    
    # Count off-diagonal matrices
    n_offdiag = div(N * (N - 1), 2)
    
    if k <= n_offdiag
        # Symmetric off-diagonal: (|j><k| + |k><j|) / 2
        # Enumerate pairs (j,k) with j < k
        idx = 0
        for col in 2:N
            for row in 1:col-1
                idx += 1
                if idx == k
                    T[row, col] = 0.5
                    T[col, row] = 0.5
                    return T
                end
            end
        end
    elseif k <= 2 * n_offdiag
        # Antisymmetric off-diagonal: -i(|j><k| - |k><j|) / 2
        k2 = k - n_offdiag
        idx = 0
        for col in 2:N
            for row in 1:col-1
                idx += 1
                if idx == k2
                    T[row, col] = -0.5im
                    T[col, row] = 0.5im
                    return T
                end
            end
        end
    else
        # Diagonal: sqrt(1/(2l(l+1))) * (sum_{j=1}^l |j><j| - l|l+1><l+1|)
        l = k - 2 * n_offdiag  # l goes from 1 to N-1
        norm = sqrt(1.0 / (2 * l * (l + 1)))
        for j in 1:l
            T[j, j] = norm
        end
        T[l+1, l+1] = -l * norm
        return T
    end
    
    error("Should not reach here")
end

"""
    _to_rational(x::Real, tol=1e-10)

Convert a real number to a rational if it's close to a simple fraction.
Returns the rational if successful, otherwise returns the original Float64.
"""
function _to_rational(x::Real, tol::Float64=1e-10)
    abs(x) < tol && return Rational{Int}(0)
    
    # Try common simple fractions first (up to denominator 12)
    for denom in 1:12
        numer = round(Int, x * denom)
        if abs(x - numer / denom) < tol
            return Rational{Int}(numer, denom)
        end
    end
    
    # Not a simple rational, return nothing to indicate failure
    return nothing
end

const STRUCTURE_CONSTANT_TOL = 1e-12

"""
    _compute_structure_constants!(alg::SUAlgebra{N})

Compute and cache the structure constants f^{abc} and d^{abc} for SU(N).
Uses the relation:
- [T^a, T^b] = i f^{abc} T^c
- {T^a, T^b} = (1/N) δ_{ab} I + d^{abc} T^c

The structure constants are computed from:
- f^{abc} = -2i Tr([T^a, T^b] T^c)
- d^{abc} = 2 Tr({T^a, T^b} T^c)
"""
function _compute_structure_constants!(alg::SUAlgebra{N}) where N
    ngen = N^2 - 1
    
    # Precompute all generators (as Float64 matrices)
    generators = [gellmann_matrix(N, k) for k in 1:ngen]
    
    for a in 1:ngen
        Ta = generators[a]
        for b in 1:ngen
            Tb = generators[b]
            
            # Commutator and anticommutator
            comm = Ta * Tb - Tb * Ta      # [T^a, T^b]
            anticomm = Ta * Tb + Tb * Ta  # {T^a, T^b}
            
            for c in 1:ngen
                Tc = generators[c]
                
                # f^{abc} = -2i Tr([T^a, T^b] T^c)
                f_abc_raw = -2im * tr(comm * Tc)
                f_abc = real(f_abc_raw)
                if abs(f_abc) > STRUCTURE_CONSTANT_TOL
                    alg.f[a, b][c] = f_abc
                end
                
                # d^{abc} = 2 Tr({T^a, T^b} T^c)
                d_abc_raw = 2 * tr(anticomm * Tc)
                d_abc = real(d_abc_raw)
                if abs(d_abc) > STRUCTURE_CONSTANT_TOL
                    alg.d[a, b][c] = d_abc
                end
            end
        end
    end
end

"""
    structure_constants(alg::SUAlgebra, a, b)

Returns the non-zero structure constants for [T^a, T^b] = i f^{abc} T^c.
Returns a Dict mapping c => f^{abc} for all c where f^{abc} ≠ 0.
"""
function structure_constants(alg::SUAlgebra, a::Int, b::Int)
    alg.f[a, b]
end

"""
    symmetric_structure_constants(alg::SUAlgebra, a, b)

Returns the non-zero symmetric structure constants for the product rule.
Returns a Dict mapping c => d^{abc} for all c where d^{abc} ≠ 0.
"""
function symmetric_structure_constants(alg::SUAlgebra, a::Int, b::Int)
    alg.d[a, b]
end

"""
    identity_coefficient(alg::SUAlgebra{N}, a, b)

Returns the coefficient of the identity in T^a T^b = coeff * I + ...
For SU(N): T^a T^b = (1/2N) δ_{ab} I + (1/2)(d^{abc} + i f^{abc}) T^c
"""
function identity_coefficient(alg::SUAlgebra{N}, a::Int, b::Int) where N
    a == b ? 1.0 / (2N) : 0.0
end

"""
    product_coefficients(alg::SUAlgebra, a, b)

Compute the full expansion of T^a T^b.
Returns (id_coeff, gen_coeffs) where:
- id_coeff is the coefficient of the identity (Float64)
- gen_coeffs is a Dict mapping c => coeff for the generator T^c (ComplexF64)

T^a T^b = id_coeff * I + sum_c gen_coeffs[c] * T^c
"""
function product_coefficients(alg::SUAlgebra{N}, a::Int, b::Int) where N
    id_coeff = identity_coefficient(alg, a, b)
    
    gen_coeffs = Dict{Int, ComplexF64}()
    
    # Contribution from d^{abc}: coefficient is d^{abc}/2
    for (c, d_abc) in alg.d[a, b]
        gen_coeffs[c] = get(gen_coeffs, c, 0.0im) + d_abc / 2
    end
    
    # Contribution from f^{abc}: coefficient is i*f^{abc}/2
    for (c, f_abc) in alg.f[a, b]
        gen_coeffs[c] = get(gen_coeffs, c, 0.0im) + im * f_abc / 2
    end
    
    (id_coeff, gen_coeffs)
end

"""
    commutator_coefficients(alg::SUAlgebra, a, b)

Compute the expansion of [T^a, T^b] = i f^{abc} T^c.
Returns a Dict mapping c => i*f^{abc} for all c where f^{abc} ≠ 0.
"""
function commutator_coefficients(alg::SUAlgebra, a::Int, b::Int)
    result = Dict{Int, ComplexF64}()
    for (c, f_abc) in alg.f[a, b]
        result[c] = im * f_abc
    end
    result
end

# ============================================================================
# Global algebra registry
# ============================================================================

"""
    AlgebraRegistry

Global registry for Lie algebras used in the current session.
Each algebra is assigned a unique ID (UInt16) that can be stored in BaseOperator.
"""
struct AlgebraRegistry
    algebras::Vector{AbstractLieAlgebra}
    # Map from (type, dim) to ID for quick lookup
    type_dim_to_id::Dict{Tuple{Symbol, Int}, UInt16}
end

const ALGEBRA_REGISTRY = AlgebraRegistry(AbstractLieAlgebra[], Dict{Tuple{Symbol, Int}, UInt16}())

"""
    register_algebra!(alg::AbstractLieAlgebra, type_sym::Symbol)

Register an algebra in the global registry and return its ID.
If an algebra of the same type and dimension is already registered, return its ID.
"""
function register_algebra!(alg::AbstractLieAlgebra, type_sym::Symbol)
    dim = algebra_dim(alg)
    key = (type_sym, dim)
    
    if haskey(ALGEBRA_REGISTRY.type_dim_to_id, key)
        return ALGEBRA_REGISTRY.type_dim_to_id[key]
    end
    
    push!(ALGEBRA_REGISTRY.algebras, alg)
    id = UInt16(length(ALGEBRA_REGISTRY.algebras))
    ALGEBRA_REGISTRY.type_dim_to_id[key] = id
    id
end

"""
    get_algebra(id::UInt16)

Retrieve an algebra from the registry by its ID.
"""
function get_algebra(id::UInt16)
    id == 0 && throw(ArgumentError("Algebra ID 0 is reserved for non-Lie-algebra operators"))
    1 <= id <= length(ALGEBRA_REGISTRY.algebras) || throw(ArgumentError("Invalid algebra ID: $id"))
    ALGEBRA_REGISTRY.algebras[id]
end

"""
    get_or_create_su(N::Int)

Get or create an SU(N) algebra and return its registry ID.
"""
function get_or_create_su(N::Int)
    key = (:SU, N)
    if haskey(ALGEBRA_REGISTRY.type_dim_to_id, key)
        return ALGEBRA_REGISTRY.type_dim_to_id[key]
    end
    alg = SUAlgebra{N}()
    register_algebra!(alg, :SU)
end

# Pre-register SU(2) with ID 1 for backwards compatibility with TLS
function __init_algebra_registry__()
    # Clear registry (in case of re-initialization)
    empty!(ALGEBRA_REGISTRY.algebras)
    empty!(ALGEBRA_REGISTRY.type_dim_to_id)
    # Register SU(2) as ID 1
    get_or_create_su(2)
end

# ============================================================================
# Pauli matrix mapping for SU(2) compatibility
# ============================================================================

# For SU(2), the generalized Gell-Mann matrices are σ_x/2, σ_y/2, σ_z/2
# Our ordering from gellmann_matrix is:
#   k=1: symmetric off-diag (1,2) -> σ_x/2
#   k=2: antisymmetric off-diag (1,2) -> σ_y/2  
#   k=3: diagonal l=1 -> σ_z/2

const SU2_SIGMA_X_INDEX = 1
const SU2_SIGMA_Y_INDEX = 2
const SU2_SIGMA_Z_INDEX = 3

# The algebra ID for SU(2), which is always registered first
const SU2_ALGEBRA_ID = UInt16(1)

# ============================================================================
# SU(2) Fast Path: Pre-computed Levi-Civita structure constants
# ============================================================================
# 
# For SU(2), [T^a, T^b] = i ε^{abc} T^c (with our normalization T = σ/2)
# The structure constants are f^{abc} = ε^{abc} (Levi-Civita symbol)
#
# su2_levicivita[a, b] = (c, ε_{abc}) where c = 6 - a - b for the non-zero case
# This matches the TLS implementation's levicivita_lut

"""
    su2_commutator_result(a::Int, b::Int)

Fast path for SU(2) commutator structure constants.
Returns (c, f_abc) where [T^a, T^b] = i f^{abc} T^c.
For a == b, returns (0, 0) indicating zero commutator.

Uses the same Levi-Civita lookup as the TLS implementation.
"""
@inline function su2_commutator_result(a::Int, b::Int)
    # Levi-Civita: ε_{123} = ε_{231} = ε_{312} = 1
    #              ε_{321} = ε_{213} = ε_{132} = -1
    #              ε_{aab} = 0 for any repeated index
    a == b && return (0, 0)
    c = 6 - a - b  # The third index (since 1+2+3=6)
    # Sign from Levi-Civita symbol: cyclic permutations of (1,2,3) are +1
    # (1,2)->3: +1, (2,3)->1: +1, (3,1)->2: +1
    # (2,1)->3: -1, (3,2)->1: -1, (1,3)->2: -1
    # Cyclic: (a,b,c) where b = a%3 + 1 gives +1
    s = (a % 3 + 1 == b) ? 1 : -1
    return (c, s)
end

"""
    su2_product_result(a::Int, b::Int)

Fast path for SU(2) product rule: T^a T^b = (1/4)δ_{ab} + (i/2)ε_{abc}T^c
Returns (id_coeff, c, gen_coeff) where:
- id_coeff is the coefficient of identity
- c is the generator index (0 if none)
- gen_coeff is the coefficient of T^c

For SU(2) with normalization Tr(T^a T^b) = δ_{ab}/2:
T^a T^b = (1/4)δ_{ab}I + (1/2)(d_{abc} + i f_{abc})T^c

For SU(2), d_{abc} = 0 (no symmetric structure constants), so:
T^a T^b = (1/4)δ_{ab}I + (i/2)f_{abc}T^c
"""
@inline function su2_product_result(a::Int, b::Int)
    if a == b
        # T^a T^a = (1/4)I (diagonal case)
        return (0.25, 0, 0.0im)
    else
        c = 6 - a - b
        # f_{abc} = ε_{abc}: cyclic permutations give +1
        s = (a % 3 + 1 == b) ? 1 : -1
        # T^a T^b = (i/2)ε_{abc}T^c
        return (0.0, c, 0.5im * s)
    end
end

# ============================================================================
# SU(3) Fast Path: Precomputed product lookup table
# ============================================================================
#
# For SU(3), products λ^a λ^b produce 1-2 generator terms plus optional identity.
# We precompute all 64 products to eliminate Dict allocations and structure
# constant lookups at runtime.
#
# Product formula: λ^a λ^b = (1/6)δ_{ab}I + (1/2)(d^{abc} + i f^{abc})λ^c

const SU3_ALGEBRA_ID = UInt16(2)

"""
    SU3ProductEntry

Precomputed result for λ^a λ^b product in SU(3).
Stores identity coefficient and up to 2 generator terms.

# Fields
- `id_coeff::Float64`: Coefficient of identity (1/6 for a==b, 0 otherwise)
- `c1::Int8`: First generator index (1-8)
- `coeff1::ComplexF64`: Coefficient of first generator
- `c2::Int8`: Second generator index (0 if single-term result)
- `coeff2::ComplexF64`: Coefficient of second generator (0 if c2==0)
"""
struct SU3ProductEntry
    id_coeff::Float64
    c1::Int8
    coeff1::ComplexF64
    c2::Int8
    coeff2::ComplexF64
end

# Precomputed constants for SU(3) structure constants
# These appear frequently: 1/(2√3) = √3/6, √3/4, 1/(4√3) = √3/12
const _SU3_INV_2SQRT3 = 0.28867513459481287   # 1/(2√3) = √3/6
const _SU3_SQRT3_4 = 0.4330127018922193       # √3/4
const _SU3_INV_4SQRT3 = 0.14433756729740643   # 1/(4√3) = √3/12
const _SU3_ID_COEFF = 0.16666666666666666     # 1/6

# Precomputed 8×8 lookup table for all SU(3) products
# Generated from structure constants: λ^a λ^b = (1/6)δ_{ab}I + (1/2)(d^{abc} + if^{abc})λ^c
# Entry format: SU3ProductEntry(id_coeff, c1, coeff1, c2, coeff2)
const SU3_PRODUCTS = [
    # Row 1: λ¹ × λʲ
    SU3ProductEntry(_SU3_ID_COEFF, 8, _SU3_INV_2SQRT3+0im, 0, 0.0+0im)     # (1,1): 1/6 I + (√3/6)λ⁸
    SU3ProductEntry(0.0, 3, 0.25+0im, 6, 0.25im)                           # (1,2): (1/4)λ³ + (i/4)λ⁶
    SU3ProductEntry(0.0, 2, 0.25+0im, 5, 0.25im)                           # (1,3): (1/4)λ² + (i/4)λ⁵
    SU3ProductEntry(0.0, 7, 0.5im, 0, 0.0+0im)                             # (1,4): (i/2)λ⁷
    SU3ProductEntry(0.0, 3, -0.25im, 6, 0.25+0im)                          # (1,5): (-i/4)λ³ + (1/4)λ⁶
    SU3ProductEntry(0.0, 2, -0.25im, 5, 0.25+0im)                          # (1,6): (-i/4)λ² + (1/4)λ⁵
    SU3ProductEntry(0.0, 4, -0.5im, 0, 0.0+0im)                            # (1,7): (-i/2)λ⁴
    SU3ProductEntry(0.0, 1, _SU3_INV_2SQRT3+0im, 0, 0.0+0im)               # (1,8): (√3/6)λ¹
    ;; # Row 2: λ² × λʲ
    SU3ProductEntry(0.0, 3, 0.25+0im, 6, -0.25im)                          # (2,1): (1/4)λ³ + (-i/4)λ⁶
    SU3ProductEntry(_SU3_ID_COEFF, 7, 0.25+0im, 8, -_SU3_INV_4SQRT3+0im)   # (2,2): 1/6 I + (1/4)λ⁷ + (-√3/12)λ⁸
    SU3ProductEntry(0.0, 1, 0.25+0im, 4, 0.25im)                           # (2,3): (1/4)λ¹ + (i/4)λ⁴
    SU3ProductEntry(0.0, 3, -0.25im, 6, -0.25+0im)                         # (2,4): (-i/4)λ³ + (-1/4)λ⁶
    SU3ProductEntry(0.0, 7, 0.25im, 8, _SU3_SQRT3_4*1im)                   # (2,5): (i/4)λ⁷ + (i√3/4)λ⁸
    SU3ProductEntry(0.0, 1, 0.25im, 4, -0.25+0im)                          # (2,6): (i/4)λ¹ + (-1/4)λ⁴
    SU3ProductEntry(0.0, 2, 0.25+0im, 5, -0.25im)                          # (2,7): (1/4)λ² + (-i/4)λ⁵
    SU3ProductEntry(0.0, 2, -_SU3_INV_4SQRT3+0im, 5, -_SU3_SQRT3_4*1im)    # (2,8): (-√3/12)λ² + (-i√3/4)λ⁵
    ;; # Row 3: λ³ × λʲ
    SU3ProductEntry(0.0, 2, 0.25+0im, 5, -0.25im)                          # (3,1): (1/4)λ² + (-i/4)λ⁵
    SU3ProductEntry(0.0, 1, 0.25+0im, 4, -0.25im)                          # (3,2): (1/4)λ¹ + (-i/4)λ⁴
    SU3ProductEntry(_SU3_ID_COEFF, 7, -0.25+0im, 8, -_SU3_INV_4SQRT3+0im)  # (3,3): 1/6 I + (-1/4)λ⁷ + (-√3/12)λ⁸
    SU3ProductEntry(0.0, 2, 0.25im, 5, 0.25+0im)                           # (3,4): (i/4)λ² + (1/4)λ⁵
    SU3ProductEntry(0.0, 1, 0.25im, 4, 0.25+0im)                           # (3,5): (i/4)λ¹ + (1/4)λ⁴
    SU3ProductEntry(0.0, 7, -0.25im, 8, _SU3_SQRT3_4*1im)                  # (3,6): (-i/4)λ⁷ + (i√3/4)λ⁸
    SU3ProductEntry(0.0, 3, -0.25+0im, 6, 0.25im)                          # (3,7): (-1/4)λ³ + (i/4)λ⁶
    SU3ProductEntry(0.0, 3, -_SU3_INV_4SQRT3+0im, 6, -_SU3_SQRT3_4*1im)    # (3,8): (-√3/12)λ³ + (-i√3/4)λ⁶
    ;; # Row 4: λ⁴ × λʲ
    SU3ProductEntry(0.0, 7, -0.5im, 0, 0.0+0im)                            # (4,1): (-i/2)λ⁷
    SU3ProductEntry(0.0, 3, 0.25im, 6, -0.25+0im)                          # (4,2): (i/4)λ³ + (-1/4)λ⁶
    SU3ProductEntry(0.0, 2, -0.25im, 5, 0.25+0im)                          # (4,3): (-i/4)λ² + (1/4)λ⁵
    SU3ProductEntry(_SU3_ID_COEFF, 8, _SU3_INV_2SQRT3+0im, 0, 0.0+0im)     # (4,4): 1/6 I + (√3/6)λ⁸
    SU3ProductEntry(0.0, 3, 0.25+0im, 6, 0.25im)                           # (4,5): (1/4)λ³ + (i/4)λ⁶
    SU3ProductEntry(0.0, 2, -0.25+0im, 5, -0.25im)                         # (4,6): (-1/4)λ² + (-i/4)λ⁵
    SU3ProductEntry(0.0, 1, 0.5im, 0, 0.0+0im)                             # (4,7): (i/2)λ¹
    SU3ProductEntry(0.0, 4, _SU3_INV_2SQRT3+0im, 0, 0.0+0im)               # (4,8): (√3/6)λ⁴
    ;; # Row 5: λ⁵ × λʲ
    SU3ProductEntry(0.0, 3, 0.25im, 6, 0.25+0im)                           # (5,1): (i/4)λ³ + (1/4)λ⁶
    SU3ProductEntry(0.0, 7, -0.25im, 8, -_SU3_SQRT3_4*1im)                 # (5,2): (-i/4)λ⁷ + (-i√3/4)λ⁸
    SU3ProductEntry(0.0, 1, -0.25im, 4, 0.25+0im)                          # (5,3): (-i/4)λ¹ + (1/4)λ⁴
    SU3ProductEntry(0.0, 3, 0.25+0im, 6, -0.25im)                          # (5,4): (1/4)λ³ + (-i/4)λ⁶
    SU3ProductEntry(_SU3_ID_COEFF, 7, 0.25+0im, 8, -_SU3_INV_4SQRT3+0im)   # (5,5): 1/6 I + (1/4)λ⁷ + (-√3/12)λ⁸
    SU3ProductEntry(0.0, 1, 0.25+0im, 4, 0.25im)                           # (5,6): (1/4)λ¹ + (i/4)λ⁴
    SU3ProductEntry(0.0, 2, 0.25im, 5, 0.25+0im)                           # (5,7): (i/4)λ² + (1/4)λ⁵
    SU3ProductEntry(0.0, 2, _SU3_SQRT3_4*1im, 5, -_SU3_INV_4SQRT3+0im)     # (5,8): (i√3/4)λ² + (-√3/12)λ⁵
    ;; # Row 6: λ⁶ × λʲ
    SU3ProductEntry(0.0, 2, 0.25im, 5, 0.25+0im)                           # (6,1): (i/4)λ² + (1/4)λ⁵
    SU3ProductEntry(0.0, 1, -0.25im, 4, -0.25+0im)                         # (6,2): (-i/4)λ¹ + (-1/4)λ⁴
    SU3ProductEntry(0.0, 7, 0.25im, 8, -_SU3_SQRT3_4*1im)                  # (6,3): (i/4)λ⁷ + (-i√3/4)λ⁸
    SU3ProductEntry(0.0, 2, -0.25+0im, 5, 0.25im)                          # (6,4): (-1/4)λ² + (i/4)λ⁵
    SU3ProductEntry(0.0, 1, 0.25+0im, 4, -0.25im)                          # (6,5): (1/4)λ¹ + (-i/4)λ⁴
    SU3ProductEntry(_SU3_ID_COEFF, 7, -0.25+0im, 8, -_SU3_INV_4SQRT3+0im)  # (6,6): 1/6 I + (-1/4)λ⁷ + (-√3/12)λ⁸
    SU3ProductEntry(0.0, 3, -0.25im, 6, -0.25+0im)                         # (6,7): (-i/4)λ³ + (-1/4)λ⁶
    SU3ProductEntry(0.0, 3, _SU3_SQRT3_4*1im, 6, -_SU3_INV_4SQRT3+0im)     # (6,8): (i√3/4)λ³ + (-√3/12)λ⁶
    ;; # Row 7: λ⁷ × λʲ
    SU3ProductEntry(0.0, 4, 0.5im, 0, 0.0+0im)                             # (7,1): (i/2)λ⁴
    SU3ProductEntry(0.0, 2, 0.25+0im, 5, 0.25im)                           # (7,2): (1/4)λ² + (i/4)λ⁵
    SU3ProductEntry(0.0, 3, -0.25+0im, 6, -0.25im)                         # (7,3): (-1/4)λ³ + (-i/4)λ⁶
    SU3ProductEntry(0.0, 1, -0.5im, 0, 0.0+0im)                            # (7,4): (-i/2)λ¹
    SU3ProductEntry(0.0, 2, -0.25im, 5, 0.25+0im)                          # (7,5): (-i/4)λ² + (1/4)λ⁵
    SU3ProductEntry(0.0, 3, 0.25im, 6, -0.25+0im)                          # (7,6): (i/4)λ³ + (-1/4)λ⁶
    SU3ProductEntry(_SU3_ID_COEFF, 8, _SU3_INV_2SQRT3+0im, 0, 0.0+0im)     # (7,7): 1/6 I + (√3/6)λ⁸
    SU3ProductEntry(0.0, 7, _SU3_INV_2SQRT3+0im, 0, 0.0+0im)               # (7,8): (√3/6)λ⁷
    ;; # Row 8: λ⁸ × λʲ
    SU3ProductEntry(0.0, 1, _SU3_INV_2SQRT3+0im, 0, 0.0+0im)               # (8,1): (√3/6)λ¹
    SU3ProductEntry(0.0, 2, -_SU3_INV_4SQRT3+0im, 5, _SU3_SQRT3_4*1im)     # (8,2): (-√3/12)λ² + (i√3/4)λ⁵
    SU3ProductEntry(0.0, 3, -_SU3_INV_4SQRT3+0im, 6, _SU3_SQRT3_4*1im)     # (8,3): (-√3/12)λ³ + (i√3/4)λ⁶
    SU3ProductEntry(0.0, 4, _SU3_INV_2SQRT3+0im, 0, 0.0+0im)               # (8,4): (√3/6)λ⁴
    SU3ProductEntry(0.0, 2, -_SU3_SQRT3_4*1im, 5, -_SU3_INV_4SQRT3+0im)    # (8,5): (-i√3/4)λ² + (-√3/12)λ⁵
    SU3ProductEntry(0.0, 3, -_SU3_SQRT3_4*1im, 6, -_SU3_INV_4SQRT3+0im)    # (8,6): (-i√3/4)λ³ + (-√3/12)λ⁶
    SU3ProductEntry(0.0, 7, _SU3_INV_2SQRT3+0im, 0, 0.0+0im)               # (8,7): (√3/6)λ⁷
    SU3ProductEntry(_SU3_ID_COEFF, 8, -_SU3_INV_2SQRT3+0im, 0, 0.0+0im)    # (8,8): 1/6 I + (-√3/6)λ⁸
]

"""
    su3_product_lookup(a::Int, b::Int) -> SU3ProductEntry

Fast O(1) lookup for SU(3) product λ^a λ^b.
Returns precomputed entry with identity coefficient and up to 2 generator terms.

Note: The lookup table is stored in column-major order (Julia default), 
so we access [b, a] to get the entry for λ^a × λ^b.
"""
@inline function su3_product_lookup(a::Int, b::Int)
    @inbounds SU3_PRODUCTS[b, a]
end

"""
    pauli_to_gen_index(pauli::Symbol)

Convert Pauli matrix symbol (:x, :y, :z) to SU(2) generator index.
"""
function pauli_to_gen_index(pauli::Symbol)
    pauli == :x && return SU2_SIGMA_X_INDEX
    pauli == :y && return SU2_SIGMA_Y_INDEX
    pauli == :z && return SU2_SIGMA_Z_INDEX
    throw(ArgumentError("Unknown Pauli symbol: $pauli"))
end

"""
    gen_index_to_pauli(idx::Int)

Convert SU(2) generator index to Pauli matrix symbol.
"""
function gen_index_to_pauli(idx::Int)
    idx == SU2_SIGMA_X_INDEX && return :x
    idx == SU2_SIGMA_Y_INDEX && return :y
    idx == SU2_SIGMA_Z_INDEX && return :z
    throw(ArgumentError("Invalid SU(2) generator index: $idx"))
end

# ============================================================================
# Ladder operator support for SU(2)
# ============================================================================

# σ± = (σx ± i σy) / 2, but in our convention T = σ/2, so:
# T± = T_x ± i T_y = (σx ± i σy) / 2
# 
# The raising/lowering operators are not generators themselves but linear combinations.
# We'll handle them separately in the operator definitions.

# ============================================================================
# Convenience functions for creating SU(N) generator operators
# These are defined here but use types from operator_defs.jl
# The actual LieAlgebraGenerator function is defined in operator_defs.jl
# ============================================================================

# Note: su_generator and su_generators are thin wrappers that will be defined
# after operator_defs.jl is loaded. See the end of operator_defs.jl for these.
