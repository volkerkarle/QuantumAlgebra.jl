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
    SOAlgebra{N}

Represents the SO(N) Lie algebra (special orthogonal group, rotations in R^N).
Generators are anti-symmetric matrices: N(N-1)/2 generators in the N-dimensional
vector representation. Normalization Tr(T^a T^b) = δ_{ab}/2.
"""
struct SOAlgebra{N} <: AbstractLieAlgebra
    f::Matrix{Dict{Int, Float64}}
    d::Matrix{Dict{Int, Float64}}
    function SOAlgebra{N}() where N
        N >= 3 || throw(ArgumentError("SO(N) requires N >= 3, got N=$N"))
        ngen = div(N * (N - 1), 2)
        f = [Dict{Int, Float64}() for _ in 1:ngen, _ in 1:ngen]
        d = [Dict{Int, Float64}() for _ in 1:ngen, _ in 1:ngen]
        alg = new{N}(f, d)
        _compute_so_structure_constants!(alg)
        alg
    end
end

algebra_dim(::SOAlgebra{N}) where N = N
num_generators(::SOAlgebra{N}) where N = div(N * (N - 1), 2)

"""
    SpAlgebra{N}

Represents the Sp(2N) Lie algebra (symplectic group).
Generators are 2N×2N matrices satisfying M^T J + J M = 0 with J = [[0,I],[-I,0]].
Total generators: N(2N+1) in the 2N-dimensional fundamental representation.
Normalization Tr(T^a T^b) = δ_{ab}/2.
"""
struct SpAlgebra{N} <: AbstractLieAlgebra
    f::Matrix{Dict{Int, Float64}}
    d::Matrix{Dict{Int, Float64}}
    function SpAlgebra{N}() where N
        N >= 1 || throw(ArgumentError("Sp(2N) requires N >= 1, got N=$N"))
        ngen = N * (2N + 1)
        f = [Dict{Int, Float64}() for _ in 1:ngen, _ in 1:ngen]
        d = [Dict{Int, Float64}() for _ in 1:ngen, _ in 1:ngen]
        alg = new{N}(f, d)
        _compute_sp_structure_constants!(alg)
        alg
    end
end

algebra_dim(::SpAlgebra{N}) where N = 2N
num_generators(::SpAlgebra{N}) where N = N * (2N + 1)

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
    _compute_so_structure_constants!(alg::SOAlgebra{N})

Compute f^{abc} for SO(N) generators using the antisymmetric Hermitian basis.
The basis is T^{(ij)} = -i * (e_i e_j^T - e_j e_i^T) / 2 for i < j.
Normalization: Tr(T^a T^b) = δ_{ab}/2.
"""
function _compute_so_structure_constants!(alg::SOAlgebra{N}) where N
    ngen = div(N * (N - 1), 2)
    generators = _so_basis_matrices(N)
    
    for a in 1:ngen
        Ta = generators[a]
        for b in 1:ngen
            Tb = generators[b]
            comm = Ta * Tb - Tb * Ta
            for c in 1:ngen
                Tc = generators[c]
                f_abc_raw = -2im * tr(comm * Tc)
                f_abc = real(f_abc_raw)
                if abs(f_abc) > STRUCTURE_CONSTANT_TOL
                    alg.f[a, b][c] = f_abc
                end
            end
        end
    end
end

"""
    _so_basis_matrices(N)

Generate the Hermitian basis matrices for SO(N) in the vector representation.
Returns N(N-1)/2 matrices of size N×N.
"""
function _so_basis_matrices(N::Int)
    ngen = div(N * (N - 1), 2)
    basis = Vector{Matrix{ComplexF64}}(undef, ngen)
    k = 0
    for j in 2:N
        for i in 1:j-1
            k += 1
            M = zeros(ComplexF64, N, N)
            M[i, j] = 1.0
            M[j, i] = -1.0
            basis[k] = -im * M / 2
        end
    end
    basis
end

"""
    _compute_sp_structure_constants!(alg::SpAlgebra{N})

Compute f^{abc} for Sp(2N) generators using the fundamental 2N×2N representation.
The basis uses the block decomposition M = [[A,B],[C,-A^T]] with B=B^T, C=C^T.
"""
function _compute_sp_structure_constants!(alg::SpAlgebra{N}) where N
    ngen = N * (2N + 1)
    generators = _sp_basis_matrices(N)
    
    for a in 1:ngen
        Ta = generators[a]
        for b in 1:ngen
            Tb = generators[b]
            comm = Ta * Tb - Tb * Ta
            for c in 1:ngen
                Tc = generators[c]
                f_abc_raw = -2im * tr(comm * Tc)
                f_abc = real(f_abc_raw)
                if abs(f_abc) > STRUCTURE_CONSTANT_TOL
                    alg.f[a, b][c] = f_abc
                end
            end
        end
    end
end

"""
    _sp_basis_matrices(N)

Generate the Hermitian basis matrices for Sp(2N) in the fundamental representation.
Returns N(2N+1) matrices of size 2N×2N.

The basis uses the block decomposition with J = [[0, I_N], [-I_N, 0]].
The sp(2N) algebra consists of matrices M = [[A, B], [C, -A^T]]
where B and C are symmetric.

Groups:
1. A-diagonal: N matrices [[E_{ii}, 0], [0, -E_{ii}]]
2. A-off-diag symmetric: N(N-1)/2 matrices [[E_{ij}+E_{ji}, 0], [0, -(E_{ij}+E_{ji})]]
3. A-off-diag anti: N(N-1)/2 matrices [[-im(E_{ij}-E_{ji}), 0], [0, im(E_{ij}-E_{ji})]]
4. B symmetric: N(N+1)/2 matrices [[0, E_{ij}+E_{ji}], [E_{ij}+E_{ji}, 0]]
5. B anti: N(N+1)/2 matrices [[0, -im(E_{ij}+E_{ji})], [im(E_{ij}+E_{ji}), 0]]

Total: N + N(N-1)/2 + N(N-1)/2 + N(N+1)/2 + N(N+1)/2 = 2N² + N = N(2N+1) ✓
"""
function _sp_basis_matrices(N::Int)
    ngen = N * (2N + 1)
    basis = Vector{Matrix{ComplexF64}}(undef, ngen)
    k = 0
    
    # Group 1: A-diagonal generators H_i = [[E_{ii}, 0], [0, -E_{ii}]]
    for i in 1:N
        k += 1
        M = zeros(ComplexF64, 2N, 2N)
        M[i, i] = 1.0
        M[N+i, N+i] = -1.0
        basis[k] = M
    end
    
    # Group 2: A-off-diagonal symmetric: K^+_{ij} = [[E_{ij}+E_{ji}, 0], [0, -(E_{ij}+E_{ji})]]
    for j in 2:N
        for i in 1:j-1
            k += 1
            M = zeros(ComplexF64, 2N, 2N)
            M[i, j] = 1.0
            M[j, i] = 1.0
            M[N+i, N+j] = -1.0
            M[N+j, N+i] = -1.0
            basis[k] = M
        end
    end
    
    # Group 3: A-off-diagonal anti: K^-_{ij} = [[-im(E_{ij}-E_{ji}), 0], [0, im(E_{ij}-E_{ji})]]
    for j in 2:N
        for i in 1:j-1
            k += 1
            M = zeros(ComplexF64, 2N, 2N)
            M[i, j] = -1.0im
            M[j, i] = 1.0im
            M[N+i, N+j] = 1.0im
            M[N+j, N+i] = -1.0im
            basis[k] = M
        end
    end
    
    # Group 4: B-block symmetric: F^+_{ij} = [[0, E_{ij}+E_{ji}], [E_{ij}+E_{ji}, 0]]
    for j in 1:N
        for i in 1:j
            k += 1
            M = zeros(ComplexF64, 2N, 2N)
            M[i, N+j] = 1.0
            M[j, N+i] = 1.0
            M[N+i, j] = 1.0
            M[N+j, i] = 1.0
            basis[k] = M
        end
    end
    
    # Group 5: B-block anti: F^-_{ij} = [[0, -im(E_{ij}+E_{ji})], [im(E_{ij}+E_{ji}), 0]]
    for j in 1:N
        for i in 1:j
            k += 1
            M = zeros(ComplexF64, 2N, 2N)
            M[i, N+j] = -1.0im
            M[j, N+i] = -1.0im
            M[N+i, j] = 1.0im
            M[N+j, i] = 1.0im
            basis[k] = M
        end
    end
    
    # Normalize each generator to Tr(T^a T^a) = 1/2
    for a in 1:ngen
        norm2 = real(tr(basis[a] * basis[a]))
        if !iszero(norm2)
            basis[a] /= sqrt(2 * norm2)
        end
    end
    
    # Verify orthogonality by removing linear dependencies (Gram-Schmidt)
    for a in 1:ngen
        for b in 1:a-1
            overlap = real(tr(basis[a] * basis[b]))
            if abs(overlap) > 1e-12
                factor = overlap / real(tr(basis[b] * basis[b]))
                basis[a] -= factor * basis[b]
            end
        end
        norm2 = real(tr(basis[a] * basis[a]))
        if !iszero(norm2)
            basis[a] /= sqrt(2 * norm2)
        end
    end
    
    basis
end

"""
    structure_constants(alg::AbstractLieAlgebra, a, b)

Returns the non-zero structure constants for [T^a, T^b] = i f^{abc} T^c.
Returns a Dict mapping c => f^{abc} for all c where f^{abc} ≠ 0.
"""
function structure_constants(alg::AbstractLieAlgebra, a::Int, b::Int)
    alg.f[a, b]
end

"""
    symmetric_structure_constants(alg::AbstractLieAlgebra, a, b)

Returns the non-zero symmetric structure constants for the product rule.
Returns a Dict mapping c => d^{abc} for all c where d^{abc} ≠ 0.
"""
function symmetric_structure_constants(alg::AbstractLieAlgebra, a::Int, b::Int)
    alg.d[a, b]
end

"""
    identity_coefficient(alg::AbstractLieAlgebra, a, b)

Returns the coefficient of the identity in T^a T^b = coeff * I + ...
For an N-dimensional algebra: T^a T^b = (1/(2N)) δ_{ab} I + ...
"""
function identity_coefficient(alg::AbstractLieAlgebra, a::Int, b::Int)
    a == b ? 1.0 / (2 * algebra_dim(alg)) : 0.0
end

"""
    product_coefficients(alg::AbstractLieAlgebra, a, b)

Compute the full expansion of T^a T^b.
Returns (id_coeff, gen_coeffs) where:
- id_coeff is the coefficient of the identity (Float64)
- gen_coeffs is a Dict mapping c => coeff for the generator T^c (ComplexF64)

T^a T^b = id_coeff * I + sum_c gen_coeffs[c] * T^c
"""
function product_coefficients(alg::AbstractLieAlgebra, a::Int, b::Int)
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
    commutator_coefficients(alg::AbstractLieAlgebra, a, b)

Compute the expansion of [T^a, T^b] = i f^{abc} T^c.
Returns a Dict mapping c => i*f^{abc} for all c where f^{abc} ≠ 0.
"""
function commutator_coefficients(alg::AbstractLieAlgebra, a::Int, b::Int)
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
const _ALGEBRA_REGISTRY_LOCK = ReentrantLock()

"""
    register_algebra!(alg::AbstractLieAlgebra, type_sym::Symbol)

Register an algebra in the global registry and return its ID.
If an algebra of the same type and dimension is already registered, return its ID.
"""
function register_algebra!(alg::AbstractLieAlgebra, type_sym::Symbol)
    dim = algebra_dim(alg)
    key = (type_sym, dim)
    
    lock(_ALGEBRA_REGISTRY_LOCK) do
        if haskey(ALGEBRA_REGISTRY.type_dim_to_id, key)
            return ALGEBRA_REGISTRY.type_dim_to_id[key]
        end
        
        push!(ALGEBRA_REGISTRY.algebras, alg)
        id = UInt16(length(ALGEBRA_REGISTRY.algebras))
        ALGEBRA_REGISTRY.type_dim_to_id[key] = id
        id
    end
end

"""
    get_algebra(id::UInt16)

Retrieve an algebra from the registry by its ID.
"""
function get_algebra(id::UInt16)
    id == 0 && throw(ArgumentError("Algebra ID 0 is reserved for non-Lie-algebra operators"))
    lock(_ALGEBRA_REGISTRY_LOCK) do
        1 <= id <= length(ALGEBRA_REGISTRY.algebras) || throw(ArgumentError("Invalid algebra ID: $id"))
        ALGEBRA_REGISTRY.algebras[id]
    end
end

"""
    get_or_create_su(N::Int)

Get or create an SU(N) algebra and return its registry ID.
Uses double-checked locking to avoid holding the lock during expensive algebra construction.
"""
function get_or_create_su(N::Int)
    key = (:SU, N)
    
    # Fast path: check if already exists under lock (required for safe read)
    existing_id = lock(_ALGEBRA_REGISTRY_LOCK) do
        get(ALGEBRA_REGISTRY.type_dim_to_id, key, nothing)
    end
    existing_id !== nothing && return existing_id
    
    # Slow path: create algebra OUTSIDE lock (expensive computation)
    # This allows other threads to access existing algebras while we compute
    alg = SUAlgebra{N}()
    
    # Now acquire lock to insert (with re-check in case another thread beat us)
    lock(_ALGEBRA_REGISTRY_LOCK) do
        # Re-check: another thread may have created it while we were computing
        if haskey(ALGEBRA_REGISTRY.type_dim_to_id, key)
            return ALGEBRA_REGISTRY.type_dim_to_id[key]
        end
        # We won the race, insert our algebra
        push!(ALGEBRA_REGISTRY.algebras, alg)
        id = UInt16(length(ALGEBRA_REGISTRY.algebras))
        ALGEBRA_REGISTRY.type_dim_to_id[key] = id
        id
    end
end

"""
    get_or_create_so(N::Int)

Get or create an SO(N) algebra and return its registry ID.
Uses double-checked locking to avoid holding the lock during expensive algebra construction.
"""
function get_or_create_so(N::Int)
    key = (:SO, N)
    
    existing_id = lock(_ALGEBRA_REGISTRY_LOCK) do
        get(ALGEBRA_REGISTRY.type_dim_to_id, key, nothing)
    end
    existing_id !== nothing && return existing_id
    
    alg = SOAlgebra{N}()
    
    lock(_ALGEBRA_REGISTRY_LOCK) do
        if haskey(ALGEBRA_REGISTRY.type_dim_to_id, key)
            return ALGEBRA_REGISTRY.type_dim_to_id[key]
        end
        push!(ALGEBRA_REGISTRY.algebras, alg)
        id = UInt16(length(ALGEBRA_REGISTRY.algebras))
        ALGEBRA_REGISTRY.type_dim_to_id[key] = id
        id
    end
end

"""
    get_or_create_sp(N::Int)

Get or create an Sp(2N) algebra and return its registry ID.
Uses double-checked locking to avoid holding the lock during expensive algebra construction.
"""
function get_or_create_sp(N::Int)
    key = (:SP, N)
    
    existing_id = lock(_ALGEBRA_REGISTRY_LOCK) do
        get(ALGEBRA_REGISTRY.type_dim_to_id, key, nothing)
    end
    existing_id !== nothing && return existing_id
    
    alg = SpAlgebra{N}()
    
    lock(_ALGEBRA_REGISTRY_LOCK) do
        if haskey(ALGEBRA_REGISTRY.type_dim_to_id, key)
            return ALGEBRA_REGISTRY.type_dim_to_id[key]
        end
        push!(ALGEBRA_REGISTRY.algebras, alg)
        id = UInt16(length(ALGEBRA_REGISTRY.algebras))
        ALGEBRA_REGISTRY.type_dim_to_id[key] = id
        id
    end
end

# Pre-register SU(2) with ID 1 for backwards compatibility with TLS
function __init_algebra_registry__()
    lock(_ALGEBRA_REGISTRY_LOCK) do
        # Clear registry (in case of re-initialization)
        empty!(ALGEBRA_REGISTRY.algebras)
        empty!(ALGEBRA_REGISTRY.type_dim_to_id)
        # Register SU(2) as ID 1 (inline to avoid double-lock)
        alg = SUAlgebra{2}()
        push!(ALGEBRA_REGISTRY.algebras, alg)
        ALGEBRA_REGISTRY.type_dim_to_id[(:SU, 2)] = UInt16(1)
    end
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

# ============================================================================
# Coefficient System for SU(N) Algebras
# ============================================================================
# Default: exact Rationals + symbolic √3. Float mode: use_float_coefficients(true)

const _symbolic_sqrt3_provider = Ref{Union{Nothing,Function}}(nothing)
set_symbolic_sqrt3_provider(f) = (_symbolic_sqrt3_provider[] = f)

_sqrt3() = (_symbolic_sqrt3_provider[] !== nothing && !using_float_coefficients()) ? 
           _symbolic_sqrt3_provider[]() : 1.7320508075688772

sun_id_coeff(N::Int) = using_float_coefficients() ? 1.0/(2N) : 1//(2N)
sun_gen_coeff() = using_float_coefficients() ? 0.5 : 1//2

# ============================================================================
# SU(3) Fast Path: Precomputed Products
# ============================================================================
# λ^a λ^b = (1/2N)δ_{ab}I + (1/2)(d^{abc} + if^{abc})λ^c
# Encoded as (has_id, c1, code1, c2, code2) with coefficient codes 0-15

const SU3_ALGEBRA_ID = UInt16(2)

# Float lookup for codes 0-15
const _SU3_COEFF_FLOAT = (
    0.0+0.0im, 0.5+0.0im, 0.25+0.0im, -0.25+0.0im, 0.0+0.5im, 0.0-0.5im, 0.0+0.25im, 0.0-0.25im,
    0.28867513459481287+0.0im, -0.28867513459481287+0.0im, 0.14433756729740643+0.0im, -0.14433756729740643+0.0im,
    0.4330127018922193+0.0im, -0.4330127018922193+0.0im, 0.0+0.4330127018922193im, 0.0-0.4330127018922193im)

# Symbolic lookup for codes 0-7 (no √3)
const _SU3_COEFF_RATIONAL = (0, 1//2, 1//4, -1//4, (1//2)*im, (-1//2)*im, (1//4)*im, (-1//4)*im)

# Decode: codes 0-7 = rational, codes 8-15 = √3 variants
@inline function _su3_decode_coeff(code::Int)
    using_float_coefficients() && return @inbounds _SU3_COEFF_FLOAT[code+1]
    code < 8 && return @inbounds _SU3_COEFF_RATIONAL[code+1]
    s3 = _sqrt3()
    return (code == 8 ? s3/6 : code == 9 ? -s3/6 : code == 10 ? s3/12 : code == 11 ? -s3/12 :
            code == 12 ? s3/4 : code == 13 ? -s3/4 : code == 14 ? s3*im/4 : -s3*im/4)
end

# Product table: (has_id, c1, code1, c2, code2) for all 64 products
const SU3_PRODUCTS_CODED = (
    (true,8,8,0,0),(false,3,2,6,6),(false,2,2,5,6),(false,7,4,0,0),(false,3,7,6,2),(false,2,7,5,2),(false,4,5,0,0),(false,1,8,0,0),
    (false,3,2,6,7),(true,7,2,8,11),(false,1,2,4,6),(false,3,7,6,3),(false,7,6,8,14),(false,1,6,4,3),(false,2,2,5,7),(false,2,11,5,15),
    (false,2,2,5,7),(false,1,2,4,7),(true,7,3,8,11),(false,2,6,5,2),(false,1,6,4,2),(false,7,7,8,14),(false,3,3,6,6),(false,3,11,6,15),
    (false,7,5,0,0),(false,3,6,6,3),(false,2,7,5,2),(true,8,8,0,0),(false,3,2,6,6),(false,2,3,5,7),(false,1,4,0,0),(false,4,8,0,0),
    (false,3,6,6,2),(false,7,7,8,15),(false,1,7,4,2),(false,3,2,6,7),(true,7,2,8,11),(false,1,2,4,6),(false,2,6,5,2),(false,2,14,5,11),
    (false,2,6,5,2),(false,1,7,4,3),(false,7,6,8,15),(false,2,3,5,6),(false,1,2,4,7),(true,7,3,8,11),(false,3,7,6,3),(false,3,14,6,11),
    (false,4,4,0,0),(false,2,2,5,6),(false,3,3,6,7),(false,1,5,0,0),(false,2,7,5,2),(false,3,6,6,3),(true,8,8,0,0),(false,7,8,0,0),
    (false,1,8,0,0),(false,2,11,5,14),(false,3,11,6,14),(false,4,8,0,0),(false,2,15,5,11),(false,3,15,6,11),(false,7,8,0,0),(true,8,9,0,0))

function su3_product_coeffs(a::Int, b::Int)
    has_id, c1, code1, c2, code2 = @inbounds SU3_PRODUCTS_CODED[(a-1)*8 + b]
    id_coeff = has_id ? sun_id_coeff(3) : (using_float_coefficients() ? 0.0 : 0)
    return (id_coeff, c1, _su3_decode_coeff(code1), c2, _su3_decode_coeff(code2))
end

# Pauli <-> generator index conversion
pauli_to_gen_index(p::Symbol) = p == :x ? SU2_SIGMA_X_INDEX : p == :y ? SU2_SIGMA_Y_INDEX : 
                                 p == :z ? SU2_SIGMA_Z_INDEX : throw(ArgumentError("Unknown Pauli: $p"))
gen_index_to_pauli(i::Int) = i == SU2_SIGMA_X_INDEX ? :x : i == SU2_SIGMA_Y_INDEX ? :y : 
                              i == SU2_SIGMA_Z_INDEX ? :z : throw(ArgumentError("Invalid index: $i"))

# ============================================================================
# Ladder operator support for SU(2)
# ============================================================================
# T± = T_x ± i T_y (raising/lowering operators)

# ============================================================================
# Convenience functions for creating SU(N) generator operators
# These are defined here but use types from operator_defs.jl
# The actual LieAlgebraGenerator function is defined in operator_defs.jl
# ============================================================================

# Note: su_generator and su_generators are thin wrappers that will be defined
# after operator_defs.jl is loaded. See the end of operator_defs.jl for these.
