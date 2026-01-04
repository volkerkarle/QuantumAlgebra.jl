# we will want to overload these operators and functions for our custom types
import Base: ==, ≈, *, +, -

export normal_form, comm

Base.length(A::BaseOpProduct) = length(A.v)
Base.length(A::Union{ExpVal,Corr}) = length(A.ops)
Base.length(A::QuTerm) = length(A.bares) + mapreduce(length,+,A.expvals;init=0) + mapreduce(length,+,A.corrs;init=0)

Base.zero(::Type{QuExpr}) = QuExpr()
Base.zero(::QuExpr) = zero(QuExpr)
Base.one(::Type{QuExpr}) = QuExpr(QuTerm())
Base.one(::Type{QuTerm}) = QuExpr(QuTerm())
Base.one(::T) where T<:Union{QuTerm,QuExpr} = one(T)

function Base.:^(A::Union{QuTerm,QuExpr},n::Int)
    if n > 0
        prod(A for _=1:n)
    elseif n == 0
        one(A)
    else
        throw(ArgumentError("Only positive exponents supported for QuantumAlgebra objects."))
    end
end
Base.literal_pow(::typeof(^),A::Union{QuTerm,QuExpr},::Val{n}) where n = A^n

noaliascopy(A::Param) = Param(A.name,A.state,indcopy(A.inds))
# δ is isbitstype
noaliascopy(A::δ) = A
noaliascopy(A::BaseOperator) = BaseOperator(A.t, A.name, indcopy(A.inds), A.algebra_id, A.gen_idx)
noaliascopy(A::T) where T<:Union{ExpVal,Corr} = T(noaliascopy(A.ops))
noaliascopy(A::BaseOpProduct) = BaseOpProduct(noaliascopy(A.v))
noaliascopy(v::Vector{T}) where T<:Union{δ,Param,BaseOperator,ExpVal,Corr} = noaliascopy.(v)
noaliascopy(A::QuTerm) = QuTerm(A.nsuminds, noaliascopy(A.δs), noaliascopy(A.params),
                                noaliascopy(A.expvals), noaliascopy(A.corrs), noaliascopy(A.bares))
# IMPT: create a dictionary directly here, do not go through _add_sum_term
noaliascopy(A::QuExpr) = QuExpr(Dict{QuTerm,Number}(noaliascopy(t) => s for (t,s) in A.terms))

==(A::BaseOperator,B::BaseOperator) = A.t == B.t && A.name == B.name && A.inds == B.inds && A.algebra_id == B.algebra_id && A.gen_idx == B.gen_idx
==(A::BaseOpProduct,B::BaseOpProduct) = A.v == B.v
==(A::Param,B::Param) = A.name == B.name && A.state == B.state && A.inds == B.inds
==(A::T,B::T) where T<:Union{ExpVal,Corr} = A.ops == B.ops
==(A::QuTerm,B::QuTerm) = (A.nsuminds == B.nsuminds && A.δs == B.δs && A.params == B.params &&
                           A.expvals == B.expvals && A.corrs == B.corrs && A.bares == B.bares)
==(A::QuExpr,B::QuExpr) = isequal(A.terms, B.terms)

function ≈(A::QuExpr,B::QuExpr)
    length(A.terms) != length(B.terms) && return false
    for (tA,sA) in A.terms
        sB = get(B.terms,tA) do
            return false
        end
        sA ≈ sB || return false
    end
    return true
end

function Base.hash(v::Vector{T},h::UInt) where T <: QuantumObject
    h = hash(length(v),h)
    for x in v
        h = hash(x,h)
    end
    h
end
function Base.hash(ind::QuIndex,h::UInt)
    ### CAREFUL: This relies on the fact that QuIndex has two 32-bit components
    ### and is the same size as a UInt64
    GC.@preserve ind begin
        hash(Base.unsafe_load(Ptr{UInt64}(Base.unsafe_convert(Ptr{Cvoid},Ref(ind)))), h)
    end
end
function Base.hash(name::QuOpName,h::UInt)
    h = hash(name.i,h)
    h
end
function Base.hash(A::BaseOperator,h::UInt)
    h = hash(UInt(A.t),h)
    h = hash(A.name,h)
    h = hash(A.inds,h)
    h = hash(A.algebra_id,h)
    h = hash(A.gen_idx,h)
    h
end
function Base.hash(A::Param,h::UInt)
    h = hash(A.name,h)
    h = hash(A.state,h)
    h = hash(A.inds,h)
    h
end
function Base.hash(A::BaseOpProduct,h::UInt)
    h = hash(A.v,h)
    h
end
function Base.hash(A::T,h::UInt) where T<:Union{ExpVal,Corr}
    h = hash(T,h)
    h = hash(A.ops,h)
    h
end
function Base.hash(A::QuTerm,h::UInt)
    h = hash(A.nsuminds,h)
    h = hash(A.δs,h)
    h = hash(A.expvals,h)
    h = hash(A.corrs,h)
    h = hash(A.bares,h)
    h
end
function Base.hash(A::QuExpr,h::UInt)
    h = hash(A.terms,h)
    h
end

# Base.cmp(::AbstractArray,::AbstractArray) uses isequal and isless, so doesn't shortcut if a,b are themselves arrays
# https://github.com/JuliaLang/julia/blob/539f3ce943f59dec8aff3f2238b083f1b27f41e5/base/abstractarray.jl
function recursive_cmp(A, B)
    for (a, b) in zip(A, B)
        cc = recursive_cmp(a,b)
        cc == 0 || return cc
    end
    return cmp(length(A), length(B))
end

@inline recursive_cmp(A::BaseOperator,B::BaseOperator) = cmp(A,B)
@inline recursive_cmp(A::Param,B::Param) = cmp(A,B)
@inline recursive_cmp(A::δ,B::δ) = cmp(A,B)
@inline recursive_cmp(A::BaseOpProduct,B::BaseOpProduct) = (cc = cmp(length(A),length(B)); cc == 0 ? recursive_cmp(A.v,B.v) : cc)
@inline recursive_cmp(A::T,B::T) where T<:Union{ExpVal,Corr}  = recursive_cmp(A.ops,B.ops)

macro _cmpAB(member,rec=true)
    fun = rec ? :recursive_cmp : :cmp
    esc(quote
        cc = $fun(A.$member,B.$member)
        cc == 0 || return cc
    end)
end

@inline function Base.cmp(A::BaseOperator,B::BaseOperator)
    @_cmpAB t false
    @_cmpAB name false
    @_cmpAB inds false
    @_cmpAB algebra_id false
    @_cmpAB gen_idx false
    # they are equal
    return 0
end

@inline function Base.cmp(A::Param,B::Param)
    @_cmpAB name false
    @_cmpAB state false
    @_cmpAB inds false
    # they are equal
    return 0
end

@inline function Base.cmp(A::δ,B::δ)
    @_cmpAB iA false
    @_cmpAB iB false
    # they are equal
    return 0
end

function Base.cmp(A::QuTerm,B::QuTerm)
    # only evaluate each part that we need for each step of the comparison to avoid unnecessary work
    # order first by number of operators (also within expectation values)
    cc = cmp(length(A), length(B))
    cc == 0 || return cc
    # then order by operators, expectation values, correlations, params, deltas, nsuminds
    @_cmpAB bares
    @_cmpAB expvals
    @_cmpAB corrs
    @_cmpAB params
    @_cmpAB δs
    @_cmpAB nsuminds false
    # they are equal
    return 0
end

@inline Base.isless(A::BaseOperator,B::BaseOperator) = cmp(A,B) < 0
@inline Base.isless(A::Param,B::Param) = cmp(A,B) < 0
@inline Base.isless(A::δ,B::δ) = cmp(A,B) < 0
@inline Base.isless(A::BaseOpProduct,B::BaseOpProduct) = recursive_cmp(A,B) < 0
@inline Base.isless(A::T,B::T) where T<:Union{ExpVal,Corr} = recursive_cmp(A,B) < 0
@inline Base.isless(A::QuTerm,B::QuTerm) = cmp(A,B) < 0

comm(A,B) = A*B - B*A

Base.adjoint(A::BaseOperator) = BaseOperator(BaseOpType_adj[Int(A.t)], A.name, A.inds, A.algebra_id, A.gen_idx)
Base.adjoint(A::BaseOpProduct) = BaseOpProduct(adjoint.(view(A.v, lastindex(A.v):-1:1)))
Base.adjoint(A::T) where {T<:Union{ExpVal,Corr}} = T(adjoint(A.ops))
Base.adjoint(A::Param) = A.state=='r' ? A : Param(A.name,A.state=='n' ? 'c' : 'n',A.inds)
function Base.adjoint(A::QuTerm)
    B = QuTerm(A.nsuminds,A.δs,adjoint.(A.params),adjoint.(A.expvals),adjoint.(A.corrs),adjoint(A.bares))
    B.nsuminds>0 ? reorder_suminds()(B) : B
end
Base.adjoint(A::QuExpr) = QuExpr((adjoint(t),adjoint(s)) for (t,s) in A.terms)

is_normal_form(A::Union{ExpVal,Corr}) = is_normal_form(A.ops)
is_normal_form(ops::BaseOpProduct) = is_normal_form(ops.v)
function is_normal_form(v::Vector{BaseOperator})
    issorted(v) || return false
    # now check that no contractions occur
    for k in 1:length(v)-1
        O = v[k]
        contract_result = _contract(O,v[k+1])
        dosimplify = contract_result[1]
        dosimplify && return false
        if O.t in (TLSx_,TLSy_,TLSz_)
            # lookahead while we can commute through
            for kp = k+2:length(v)
                v[kp].t in (TLSx_,TLSy_,TLSz_) || break
                _exchange(v[kp-1],v[kp]) == (1,nothing) || break
                contract_result_inner = _contract(O,v[kp])
                dosimplify_inner = contract_result_inner[1]
                dosimplify_inner && return false
            end
        end
    end
    return true
end
function is_normal_form(t::QuTerm)
    (issorted(t.params) && issorted(t.expvals) && issorted(t.corrs)) || return false
    is_normal_form(t.bares) || return false
    for EV in t.expvals
        is_normal_form(EV) || return false
    end
    for CO in t.corrs
        is_normal_form(CO) || return false
    end

    # ensure that sum indices are ordered
    changeable_indices = indices(t,false)
    last_sumind = sumindex(0).num
    for ind in Base.Iterators.filter(issumindex,changeable_indices)
        # sumindex have to be increasing without jumps
        ind.num ≤ last_sumind + oneunit(last_sumind) || return false
        last_sumind = max(ind.num,last_sumind)
    end

    isempty(t.δs) && return true

    # all indices that are not in δs
    changeable_indices_set = Set{QuIndex}(changeable_indices)
    # accumulate indices that δs want to change (to compare with later δs)
    replace_inds = Set{QuIndex}()
    for (iδ,dd) in enumerate(t.δs)
        iA,iB = dd.iA, dd.iB
        # δ is not ordered (or disappears if iA == iB)
        iA ≥ iB && return false
        # the whole term disappears in cleanup
        isintindex(iA) && isintindex(iB) && return false
        (issumindex(iA) || issumindex(iB)) && return false
        (iA in replace_inds || iB in replace_inds) && return false
        if iδ > 1
            (dd > @inbounds t.δs[iδ-1]) || return false
        end
        iB in changeable_indices_set && return false
        push!(replace_inds, iB)
    end
    return true
end
is_normal_form(A::QuExpr) = all(is_normal_form, keys(A.terms))

function _normalize_without_commutation(A::QuTerm)::Union{QuTerm,Nothing}
    # first, clean up the δs
    if isempty(A.δs)
        # we always want _normalize_without_commutation to return a new object so we can modify it later
        A = noaliascopy(A)
    else
        δs = sort(A.δs)
        #println("in _normalize_without_commutation for A = $A, starting with δs = $δs")
        replacements = Dict{QuIndex,QuIndex}()
        delsuminds = IndexInt[]
        iwrite = 1
        for dd in δs
            iA = get(replacements,dd.iA,dd.iA)
            iB = get(replacements,dd.iB,dd.iB)
            if iA == iB
                # this delta just gives one
                continue
            elseif isintindex(iA) && isintindex(iB)
                # the term is zero
                return nothing
            elseif iA > iB
                iA, iB = iB, iA
            end
            if issumindex(iB) # one sum disappears, and the delta does as well
                replacements[iB] = iA
                # we will later need to shift all larger sumindices down by one
                iB.num ∈ delsuminds || push!(delsuminds,iB.num)
            elseif issumindex(iA) # one sum disappears, and the delta does as well
                replacements[iA] = iB
                # we will later need to shift all larger sumindices down by one
                iA.num ∈ delsuminds || push!(delsuminds,iA.num)
            else
                # otherwise, replace the larger index by the smaller one
                replacements[iB] = iA
                # we keep the delta, but include any possible replacements made
                δs[iwrite] = δ(iA,iB)
                iwrite += 1
            end
        end
        resize!(δs,iwrite-1)
        # after replacements, δs can be out of order
        sort!(δs)
        if !isempty(delsuminds)
            sumrepls = Dict{QuIndex,QuIndex}()
            sort!(delsuminds)
            for (shift,startind) in enumerate(delsuminds)
                endind = shift == length(delsuminds) ? A.nsuminds : delsuminds[shift+1]-1
                for num = startind+1:endind
                    sumrepls[sumindex(num)] = sumindex(num-shift)
                end
            end
            # we want to first apply replacements and then sumrepls
            # this is the same as replacing results from replacements with sumrepls,
            # and then applying sumrepls on indices that have not been replaced yet
            for (iold,inew) in replacements
                x = get(sumrepls,inew,inew)
                x != inew && (replacements[iold] = x)
            end
            # sumrepl only for indices not in replacements
            merge!(sumrepls,replacements)
            replacements = sumrepls
        end
        nsuminds = A.nsuminds-length(delsuminds)
        f = replace_inds(replacements)
        #println("in _normalize_without_commutation, deleting sum indices $delsuminds and doing replacements $replacements.")
        A = QuTerm(nsuminds,δs,f.(A.params),f.(A.expvals),f.(A.corrs),f(A.bares))
    end
    # also sort all the commuting terms
    sort!(A.params)
    sort!(A.expvals)
    sort!(A.corrs)
    A
end

# levicivita_lut[a,b] contains the Levi-Cevita symbol ϵ_abc
# for c=6-a-b, i.e, when a,b,c is a permutation of 1,2,3
const levicivita_lut = ((0,1,-1), (-1,0,1), (1,-1,0))
function ϵ_ab(A::BaseOperator,B::BaseOperator)
    # a+b+c == 6 (since a,b,c is a permutation of 1,2,3)
    a = Int(A.t) - Int(TLSx_) + 1
    b = Int(B.t) - Int(TLSx_) + 1
    c = BaseOpType(Int(TLSx_) - 1 + (6 - a - b))
    s = @inbounds levicivita_lut[a][b]
    c, s
end

# ExchangeResult for commutation relations
# For simple cases: one operator with a complex-rational prefactor
# For SU(N) with N>2: multiple operators (ops is a Vector of (prefactor, operator) pairs)
struct ExchangeResult
    pref::Complex{Rational{Int}}  # prefactor for identity term
    δs::Vector{δ}
    op::Union{Nothing,BaseOperator}  # single operator (legacy, for most cases)
    # For SU(N) multi-term results: Vector of (coefficient, operator)
    # If non-empty, this takes precedence over op
    ops::Vector{Tuple{ComplexF64, BaseOperator}}
end

# Legacy constructor for single-operator results (accepts any Number for pref)
ExchangeResult(pref::Number, δs::Vector{δ}, op::Union{Nothing,BaseOperator}) = 
    ExchangeResult(Complex{Rational{Int}}(pref), δs, op, Tuple{ComplexF64, BaseOperator}[])

# Constructor for multi-operator results (SU(N) with N > 2)
ExchangeResult(pref::Number, δs::Vector{δ}, ops::Vector{Tuple{ComplexF64, BaseOperator}}) = 
    ExchangeResult(Complex{Rational{Int}}(pref), δs, nothing, ops)

# Check if this result has multiple operators
has_multi_ops(r::ExchangeResult) = !isempty(r.ops)

# ContractionResult for operator products (simplification)
# Supports the SU(N) product rule: T^a T^b = (1/2N)δ_{ab}I + (1/2)(d^{abc} + if^{abc})T^c
# This can produce an identity term plus multiple generator terms
struct ContractionResult
    id_coeff::ComplexF64  # coefficient of identity term
    ops::Vector{Tuple{ComplexF64, BaseOperator}}  # generator terms: (coefficient, operator)
end

# Constructor for simple contractions (single operator or none)
ContractionResult(id_coeff::Number, op::Union{Nothing,BaseOperator}) = 
    op === nothing ? ContractionResult(ComplexF64(id_coeff), Tuple{ComplexF64, BaseOperator}[]) :
                     ContractionResult(ComplexF64(0), [(ComplexF64(id_coeff), op)])

# Check if contraction produces only identity (no operators)
is_identity_only(r::ContractionResult) = isempty(r.ops)

# Check if contraction produces a single operator
is_single_op(r::ContractionResult) = length(r.ops) == 1

# rewrite B A as x A B + y, with A < B in our ordering
# returns (x::Int,y::Union{ExchangeResult,Nothing})
function _exchange(A::BaseOperator,B::BaseOperator)::Tuple{Int,Union{ExchangeResult,Nothing}}
    if A.t == B.t
        # these operators always commute with the same type
        A.t in (BosonDestroy_,BosonCreate_,TLSCreate_,TLSDestroy_,TLSx_,TLSy_,TLSz_) && return (1,nothing)
        # these operators anticommute if they refer to the same species (name), and commute otherwise
        A.t in (FermionDestroy_,FermionCreate_) && return (A.name == B.name ? -1 : 1,nothing)
    end

    # different types of operators commute
    if A.t in (BosonDestroy_,BosonCreate_) && B.t in (FermionDestroy_,FermionCreate_,TLSx_,TLSy_,TLSz_,TLSDestroy_,TLSCreate_)
        return (1,nothing)
    elseif A.t in (FermionDestroy_,FermionCreate_) && B.t in (BosonDestroy_,BosonCreate_,TLSx_,TLSy_,TLSz_,TLSDestroy_,TLSCreate_)
        return (1,nothing)
    elseif A.t in (TLSx_,TLSy_,TLSz_,TLSDestroy_,TLSCreate_) && B.t in (BosonDestroy_,BosonCreate_,FermionDestroy_,FermionCreate_)
        return (1,nothing)
    end

    # a(i) a'(j) = a'(j) a(i) + δij
    if A.t == BosonCreate_ && B.t == BosonDestroy_
        if A.name == B.name && (dd = δ(A.inds,B.inds)) !== nothing
            return (1, ExchangeResult(1,dd,nothing))
        else
            return (1, nothing)
        end
    end

    # f(i) f'(j) = -f'(j) f(i) + δij
    if A.t == FermionCreate_ && B.t == FermionDestroy_
        if A.name == B.name
            dd = δ(A.inds,B.inds)
            return (-1, dd === nothing ? nothing : ExchangeResult(1,dd,nothing))
        else
            return (1, nothing)
        end
    end

    if A.t == TLSCreate_ && B.t == TLSDestroy_
        if A.name != B.name || (dd = δ(A.inds,B.inds)) === nothing
            return (1,nothing)
        elseif isempty(dd)
            # indices were all the same, σ- σ+ = 1 - σ+ σ-
            return (-1, ExchangeResult(1, dd, nothing))
        else
            # σ-_i σ+_j = σ+_j σ-_i - δij σz_i = σ+_j σ-_i + δij (1 - 2 σ+_i σ-_i)
            # note that we return a TLSz to "fit" in ExchangeResult, but need to undo that later on
            # also note that we pass "+TLSz" instead of "-TLSz" so we do not have to flip the sign of the "1" term in the sort algorithm
            return (1, ExchangeResult(1, dd, TLSz(A.name,A.inds)))
        end
    end

    # B A as x A B + y, with A < B
    if A.t in (TLSx_,TLSy_,TLSz_) && B.t in (TLSx_,TLSy_,TLSz_)
        if A.name != B.name || (dd = δ(A.inds,B.inds)) === nothing
            return (1,nothing)
        else
            # we need ϵbac
            c, s = ϵ_ab(B,A)
            if isempty(dd)
                # indices were all the same,
                # σb σa = δab + i ϵbac σc
                # since A < B, we know that a != b
                return (0, ExchangeResult(im*s,dd,BaseOperator(c,A.name,A.inds)))
            else
                # [σb_i,σa_j] = 2i εbac δij σc_i =>
                # σb_j σa_i = σa_i σb_j + 2i εbac δij σc_i
                return (1, ExchangeResult(2im*s,dd,BaseOperator(c,A.name,A.inds)))
            end
        end
    end

    if (A.t == TLSCreate_ && B.t in (TLSx_,TLSy_,TLSz_)) || (A.t in (TLSx_,TLSy_,TLSz_) && B.t == TLSDestroy_)
        throw(ArgumentError("QuantumAlgebra currently does not support mixing 'normal' (x,y,z) Pauli matrices with jump operators (+,-)."))
    end

    # Lie algebra generators commute with all other operator types
    if A.t == LieAlgebraGen_ && B.t != LieAlgebraGen_
        return (1, nothing)
    elseif A.t != LieAlgebraGen_ && B.t == LieAlgebraGen_
        return (1, nothing)
    end

    # Lie algebra generator commutation: [T^a, T^b] = i f^{abc} T^c
    if A.t == LieAlgebraGen_ && B.t == LieAlgebraGen_
        return _exchange_lie_algebra_generators(A, B)
    end

    error("_exchange should never reach this! A=$A, B=$B.")
end

"""
    _exchange_lie_algebra_generators(A, B)

Handle exchange of two Lie algebra generators.
B A = A B + [B, A] = A B - [A, B] = A B - i f^{abc} T^c

For generators from the same algebra with the same indices:
[T^a, T^b] = i f^{abc} T^c

Returns (exchange_prefactor, ExchangeResult) following the convention:
B A = exchange_prefactor * A B + terms_from_ExchangeResult

Optimized with fast path for SU(2) using direct Levi-Civita computation.
"""
function _exchange_lie_algebra_generators(A::BaseOperator, B::BaseOperator)
    # Different algebras or different names commute
    if A.algebra_id != B.algebra_id || A.name != B.name
        return (1, nothing)
    end
    
    # Check if indices match
    dd = δ(A.inds, B.inds)
    if dd === nothing
        return (1, nothing)
    end
    
    # Same generator commutes with itself
    if A.gen_idx == B.gen_idx
        return (1, nothing)
    end
    
    a, b = Int(A.gen_idx), Int(B.gen_idx)
    
    # =========================================================================
    # FAST PATH: SU(2) - use direct Levi-Civita computation (like TLS)
    # =========================================================================
    if A.algebra_id == SU2_ALGEBRA_ID
        # [T^a, T^b] = i ε_{abc} T^c
        # We want B A = A B + [B, A] = A B - [A, B] = A B - i ε_{abc} T^c
        c, eps_abc = su2_commutator_result(a, b)
        if c == 0
            return (1, nothing)
        end
        # Coefficient is -i * ε_{abc} (note: structure constant f_{abc} = ε_{abc} for SU(2))
        coeff = Complex{Rational{Int}}(0, -eps_abc)
        new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
        return (1, ExchangeResult(coeff, dd, new_op))
    end
    
    # =========================================================================
    # GENERAL PATH: Use structure constant lookup
    # =========================================================================
    alg = get_algebra(A.algebra_id)
    f_ab = structure_constants(alg, a, b)
    
    if isempty(f_ab)
        return (1, nothing)
    end
    
    # Build result from structure constants
    # [B, A] = -[A, B] = -i f^{abc} T^c
    if length(f_ab) == 1
        # Single term result
        c, f_abc = first(f_ab)
        coeff = -im * f_abc
        new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
        r_coeff = _try_rationalize_complex(coeff)
        if r_coeff !== nothing
            return (1, ExchangeResult(r_coeff, dd, new_op))
        else
            return (1, ExchangeResult(0//1, dd, [(coeff, new_op)]))
        end
    else
        # Multi-term result (e.g., SU(3) and higher)
        ops = Tuple{ComplexF64, BaseOperator}[]
        for (c, f_abc) in f_ab
            coeff = -im * f_abc
            new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
            push!(ops, (coeff, new_op))
        end
        return (1, ExchangeResult(0//1, dd, ops))
    end
end

"""
Try to convert a ComplexF64 to Complex{Rational{Int}}.
Returns nothing if not representable as a simple rational.
"""
function _try_rationalize_complex(z::ComplexF64, tol::Float64=1e-10)
    re_r = _to_rational(real(z), tol)
    im_r = _to_rational(imag(z), tol)
    if re_r !== nothing && im_r !== nothing
        return Complex{Rational{Int}}(re_r, im_r)
    end
    return nothing
end

const ComplexInt = Complex{Int}

# Legacy contraction result type for backwards compatibility
const LegacyContractionResult = Tuple{Bool,ComplexInt,Union{BaseOperator,Nothing}}

function _contract(A::BaseOperator,B::BaseOperator)::Union{LegacyContractionResult, Tuple{Bool, ContractionResult}}
    if A.t in (TLSCreate_,TLSDestroy_,FermionDestroy_,FermionCreate_) && A == B
        return (true,zero(ComplexInt),nothing)
    elseif A.t in (TLSx_,TLSy_,TLSz_) && B.t in (TLSx_,TLSy_,TLSz_) && A.name == B.name && A.inds == B.inds
        # σa σb = δab + i ϵabc σc
        if B.t == A.t
            return (true,one(ComplexInt),nothing)
        else
            c, s = ϵ_ab(A,B)
            return (true, ComplexInt(0,s), BaseOperator(c,A.name,A.inds))
        end
    elseif A.t == LieAlgebraGen_ && B.t == LieAlgebraGen_
        return _contract_lie_algebra_generators(A, B)
    else
        return (false,zero(ComplexInt),nothing)
    end
end

"""
    _contract_lie_algebra_generators(A, B)

Implement the full SU(N) product rule for Lie algebra generators:
T^a T^b = (1/2N) δ_{ab} I + (1/2)(d^{abc} + i f^{abc}) T^c

where:
- d^{abc} are the symmetric structure constants
- f^{abc} are the antisymmetric structure constants

Returns (do_contract::Bool, ContractionResult) or legacy tuple.

Optimized with fast path for SU(2) using direct computation.
For SU(2), returns legacy tuple format when possible for inline processing.
"""
function _contract_lie_algebra_generators(A::BaseOperator, B::BaseOperator)
    # Different algebras or different names don't contract
    if A.algebra_id != B.algebra_id || A.name != B.name
        return (false, zero(ComplexInt), nothing)
    end
    
    # Different indices don't contract (would need δ function)
    if A.inds != B.inds
        return (false, zero(ComplexInt), nothing)
    end
    
    a, b = Int(A.gen_idx), Int(B.gen_idx)
    
    # =========================================================================
    # FAST PATH: SU(2) - use legacy tuple format for inline processing
    # =========================================================================
    # For SU(2): T^a T^b = (1/4)δ_{ab}I + (i/2)ε_{abc}T^c
    # 
    # The TLS convention is: σa σb = δab + i ϵabc σc (with σ^2 = 1)
    # Our convention is: T^a T^b = (1/4)δ_{ab}I + (i/2)ε_{abc}T^c (with T = σ/2)
    #
    # Unfortunately, the legacy format assumes a multiplicative identity (prefactor),
    # but our result has an additive identity term. We can only use legacy format
    # for the off-diagonal case where there's no identity contribution.
    #
    # For the diagonal case (a == b), we must use ContractionResult since we need
    # to return (1/4)I which is not representable as a simple prefactor times the
    # remaining expression.
    
    if A.algebra_id == SU2_ALGEBRA_ID
        if a == b
            # Diagonal case: T^a T^a = (1/4)I
            # Must use ContractionResult for the identity coefficient
            result = ContractionResult(0.25, Tuple{ComplexF64, BaseOperator}[])
            return (true, result)
        else
            # Off-diagonal case: T^a T^b = (i/2)ε_{abc}T^c (no identity term!)
            # Can use legacy format for faster inline processing
            c = 6 - a - b
            s = (a % 3 + 1 == b) ? 1 : -1
            # Coefficient is (i/2)ε_{abc}
            coeff = ComplexInt(0, s)  # Will be divided by 2 below... wait, we need 0.5i
            # Actually ComplexInt is Complex{Int}, so we can't represent 0.5i
            # We have to use ContractionResult after all
            new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
            result = ContractionResult(0.0, [(ComplexF64(0.5im * s), new_op)])
            return (true, result)
        end
    end
    
    # =========================================================================
    # GENERAL PATH: Use structure constant lookup
    # =========================================================================
    alg = get_algebra(A.algebra_id)
    N = algebra_dim(alg)
    
    # Identity coefficient: (1/2N) δ_{ab}
    id_coeff = a == b ? 1.0 / (2N) : 0.0
    
    # Generator terms: (1/2)(d^{abc} + i f^{abc}) T^c
    d_ab = symmetric_structure_constants(alg, a, b)
    f_ab = structure_constants(alg, a, b)
    
    # Combine d and f structure constants
    gen_coeffs = Dict{Int, ComplexF64}()
    
    for (c, d_abc) in d_ab
        gen_coeffs[c] = get(gen_coeffs, c, 0.0im) + d_abc / 2
    end
    
    for (c, f_abc) in f_ab
        gen_coeffs[c] = get(gen_coeffs, c, 0.0im) + im * f_abc / 2
    end
    
    # Build the operator list
    ops = Tuple{ComplexF64, BaseOperator}[]
    for (c, coeff) in gen_coeffs
        if abs(coeff) > 1e-12
            new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
            push!(ops, (coeff, new_op))
        end
    end
    
    # Sort operators by generator index for deterministic ordering
    sort!(ops, by = x -> x[2].gen_idx)
    
    result = ContractionResult(id_coeff, ops)
    return (true, result)
end

function normal_order!(ops::BaseOpProduct,term_collector,shortcut_vacA_zero=false)
    # do an insertion sort to get to normal ordering
    # reference: https://en.wikipedia.org/wiki/Insertion_sort
    A = ops.v
    prefactor::ComplexInt = one(ComplexInt)
    for i = 2:length(A)
        j = i
        while j>1 && A[j]<A[j-1]
            # need to commute A[j-1] and A[j]
            pp, exc_res = _exchange(A[j],A[j-1])
            #println("exchanging A[$j] = $(A[j]) and A[$kk] = $(A[kk]) gave result: $pp, $exc_res.")
            if exc_res !== nothing
                if has_multi_ops(exc_res)
                    # Multi-operator result from SU(N) with N > 2
                    # Add identity term if present
                    if !iszero(exc_res.pref)
                        onew = BaseOpProduct([A[1:j-2]; A[j+1:end]])
                        t = QuTerm(exc_res.δs, onew)
                        _add_sum_term!(term_collector, t, exc_res.pref * prefactor)
                    end
                    # Add each operator term
                    for (coeff, op) in exc_res.ops
                        onew = BaseOpProduct([A[1:j-2]; op; A[j+1:end]])
                        t = QuTerm(exc_res.δs, onew)
                        _add_sum_term!(term_collector, t, coeff * prefactor)
                    end
                elseif exc_res.op === nothing
                    onew = BaseOpProduct([A[1:j-2]; A[j+1:end]])
                    t = QuTerm(exc_res.δs, onew)
                    _add_sum_term!(term_collector,t,exc_res.pref*prefactor)
                elseif A[j].t == TLSCreate_
                    @assert exc_res.op.t == TLSz_
                    # we got σz_i, have to replace it by (1 - 2 σ+_i σ-_i) (which is really -σz, see explanation in _exchange)
                    onew = BaseOpProduct([A[1:j-2]; TLSCreate(exc_res.op.name,exc_res.op.inds); TLSDestroy(exc_res.op.name,exc_res.op.inds); A[j+1:end]])
                    t = QuTerm(exc_res.δs, onew)
                    _add_sum_term!(term_collector,t,-2exc_res.pref*prefactor)
                    # this one gives the "1" term that is used below
                    onew = BaseOpProduct([A[1:j-2]; A[j+1:end]])
                    t = QuTerm(exc_res.δs, onew)
                    _add_sum_term!(term_collector,t,exc_res.pref*prefactor)
                else
                    onew = BaseOpProduct([A[1:j-2]; exc_res.op; A[j+1:end]])
                    t = QuTerm(exc_res.δs, onew)
                    _add_sum_term!(term_collector,t,exc_res.pref*prefactor)
                end
                #println("adding term = $(exc_res.pref*prefactor) * $t, term_collector = $(term_collector)")
            end
            # only modify prefactor after the exchange
            prefactor *= pp
            iszero(prefactor) && return prefactor

            # now finally exchange the two
            A[j-1], A[j] = A[j], A[j-1]
            j -= 1
        end
        if shortcut_vacA_zero && A[1].t in (BosonCreate_,FermionCreate_,TLSCreate_)
            return zero(prefactor)
        end
    end
    # check if we have any products that simplify
    did_contractions = false
    k = 1
    while k < length(A)
        contract_result = _contract(A[k],A[k+1])
        dosimplify = contract_result[1]
        if dosimplify
            did_contractions = true
            if contract_result isa Tuple{Bool, ContractionResult}
                # Lie algebra generator contraction (multi-term result)
                result = contract_result[2]
                
                # For Lie algebra contractions, we add all terms to term_collector
                # and then signal that this product should be zeroed out
                
                # Add identity term if present: id_coeff * (remaining operators)
                if abs(result.id_coeff) > 1e-12
                    remaining_ops = [A[1:k-1]; A[k+2:end]]
                    onew = BaseOpProduct(remaining_ops)
                    t = QuTerm(onew)
                    _add_sum_term!(term_collector, t, result.id_coeff * prefactor)
                end
                
                # Add generator terms: coeff * (A[1:k-1] op A[k+2:end])
                for (coeff, op) in result.ops
                    remaining_ops = [A[1:k-1]; op; A[k+2:end]]
                    onew = BaseOpProduct(remaining_ops)
                    t = QuTerm(onew)
                    _add_sum_term!(term_collector, t, coeff * prefactor)
                end
                
                # Remove the contracted pair from A
                deleteat!(A, (k, k+1))
                
                # Since we've added all contributions to term_collector,
                # the "main" term (A with prefactor) should contribute nothing more
                # We set prefactor to 0 and return - further processing of A is meaningless
                return zero(prefactor)
            else
                # Legacy contraction (TLS, fermions, etc.)
                fac = contract_result[2]
                op = contract_result[3]
                prefactor *= fac
                iszero(prefactor) && return prefactor
                if op === nothing
                    deleteat!(A,(k,k+1))
                else
                    A[k] = op
                    deleteat!(A,k+1)
                end
                k > 1 && (k -= 1)
            end
        else
            if !isempty(A) && k <= length(A) && A[k].t in (TLSx_,TLSy_,TLSz_)
                # lookahead while we can commute through
                # we already checked with k+1, so start at k+2
                for kp = k+2:length(A)
                    A[kp].t in (TLSx_,TLSy_,TLSz_) || break
                    _exchange(A[kp-1],A[kp]) == (1,nothing) || break
                    contract_result_inner = _contract(A[k],A[kp])
                    dosimplify_inner = contract_result_inner[1]
                    if dosimplify_inner
                        did_contractions = true
                        # Legacy contraction only for TLS lookahead
                        fac = contract_result_inner[2]
                        op = contract_result_inner[3]
                        prefactor *= fac
                        iszero(prefactor) && return prefactor
                        if op === nothing
                            deleteat!(A,(k,kp))
                        else
                            A[k] = op
                            deleteat!(A,kp)
                        end
                        k > 1 && (k -= 2)
                        break
                    end
                end
            end
            k += 1
        end
    end
    if did_contractions
        # do another round of normal ordering since this could have changed after contractions
        prefactor *= normal_order!(ops,term_collector)
    end
    prefactor
end

normal_order!(A::Union{Corr,ExpVal},term_collector) = normal_order!(A.ops,term_collector)
function normal_order!(v::Vector{T},commterms,pref=1) where {T<:Union{Corr,ExpVal}}
    for (ii,EC) in enumerate(v)
        newterms = QuExpr()
        pp = normal_order!(EC,newterms)
        for (t,s) in newterms.terms
            #@assert iszero(t.nsuminds) && isempty(t.params) && isempty(t.expvals) && isempty(t.corrs)
            nv = [isempty(t.bares) ? T[] : T(t.bares); v[1:ii-1]; v[ii+1:end]]
            nt = QuTerm(t.δs, noaliascopy(nv))
            # include current prefactor here
            _add_sum_term!(commterms,nt,pref*s)
        end
        # only modify prefactor after the exchange
        pref *= pp
    end
    filter!(EC -> !isempty(EC.ops), v)
    sort!(v)
    pref
end

# iterate over a single term t with prefactor s and a sum as if they were in the same sum
termsumiter(t::QuTerm,s,sum::QuExpr) = Base.Iterators.flatten((((t,s),), sum.terms))
termsumiter(t,s,sum::QuExpr) = termsumiter(QuTerm(t),s,sum)

function _add_with_normal_order!(A::QuExpr,t::QuTerm,s,shortcut_vacA_zero=false)
    # no need to do anything since everything is in normal order
    # in particular, no copy necessary!
    if is_normal_form(t)
       _add_sum_term!(A,t,s)
       return
    end

    t = _normalize_without_commutation(t)
    t === nothing && return
    newbareterms = QuExpr()
    newexpvterms = QuExpr()
    newcorrterms = QuExpr()
    prefbare = normal_order!(t.bares,   newbareterms, shortcut_vacA_zero)
    prefexpv = normal_order!(t.expvals, newexpvterms)
    prefcorr = normal_order!(t.corrs,   newcorrterms)

    # normal_form(t) = t.prefs * (prefexpv*tN.ev + ev_new) * (prefcorr*tN.co + co_new) * (prefbare*tN.bare + bare_new)
    # t has already been transformed to tN in-place
    pref = prefbare*prefexpv*prefcorr
    if !iszero(pref)
        if iszero(t.nsuminds)
            # should be guaranteed to be in normal order
            _add_sum_term!(A,t,s*pref)
        else
            # might need to reorder sum indices, which can "break" our normal order
            tn = reorder_suminds()(t)
            # this will check again if it is in normal form to make sure
            _add_with_normal_order!(A,tn,s*pref,shortcut_vacA_zero)
        end
    end

    isempty(newexpvterms) && isempty(newcorrterms) && isempty(newbareterms) && return

    firstterm = true
    for (tev,sev) in termsumiter(t.expvals, prefexpv, newexpvterms)
        #@assert iszero(tev.nsuminds) && isempty(tev.params) && isempty(tev.corrs) && isempty(tev.bares)
        for (tco,sco) in termsumiter(t.corrs, prefcorr, newcorrterms)
            #@assert iszero(tco.nsuminds) && isempty(tco.params) && isempty(tco.expvals) && isempty(tco.bares)
            for (tba,sba) in termsumiter(t.bares, prefbare, newbareterms)
                #@assert iszero(tba.nsuminds) && isempty(tba.params) && isempty(tba.expvals) && isempty(tba.corrs)

                if firstterm
                    # the first term has been added above in _add_sum_term!
                    firstterm = false
                    continue
                end

                pref = sev*sco*sba
                iszero(pref) && continue
                # IMPTE: nt is cleaned afterwards with _normalize_without_commutation,
                # which creates copies of everything, so we can reuse arrays here
                nt = QuTerm(t.nsuminds,[t.δs;tev.δs;tco.δs;tba.δs],t.params,tev.expvals,tco.corrs,tba.bares)
                _add_with_normal_order!(A,nt,s*pref,shortcut_vacA_zero)
            end
        end
    end
end

function normal_form(A::QuExpr,shortcut_vacA_zero=false)
    An = QuExpr()
    for (t,s) in A.terms
        _add_with_normal_order!(An,t,s,shortcut_vacA_zero)
    end
    An
end
normal_form(A::Number) = QuExpr(A)

function *(A::QuTerm,B::QuTerm)
    nsuminds = A.nsuminds + B.nsuminds
    if A.nsuminds > 0 && B.nsuminds > 0
        if A.nsuminds > B.nsuminds
            A = shift_sumind(B.nsuminds)(A)
        else
            B = shift_sumind(A.nsuminds)(B)
        end
    end
    if nsuminds > 0
        f = reorder_suminds()
        apflat = (args...) -> f.(Iterators.flatten(args))
    else
        apflat = vcat
    end
    δs = apflat(A.δs, B.δs)
    params = apflat(A.params, B.params)
    expvals = apflat(A.expvals, B.expvals)
    corrs = apflat(A.corrs, B.corrs)
    barevec = apflat(A.bares.v, B.bares.v)
    QuTerm(nsuminds,δs,params,expvals,corrs,BaseOpProduct(barevec))
end

*(A::Union{QuExpr,QuTerm}) = A
function *(A::QuExpr,B::QuExpr)
    QuExpr((tA*tB,sA*sB) for ((tA,sA),(tB,sB)) in Iterators.product(A.terms,B.terms))
end
*(A::Number,B::QuExpr) = QuExpr((tB,A*sB) for (tB,sB) in B.terms)
*(B::QuExpr,A::Number) = A*B
Base.:/(A::QuExpr,B::Number) = inv(B)*A

+(A::QuExpr) = A
function +(A::QuExpr,B::QuExpr)
    S = copy(A)
    for (t,s) in B.terms
        _add_with_auto_order!(S,t,s)
    end
    S
end
function +(A::QuExpr,B::Number)
    S = copy(A)
    _add_with_auto_order!(S,QuTerm(),B)
    S
end
+(B::Number,A::QuExpr) = A+B

-(A::QuExpr) = -1 * A
function -(A::QuExpr,B::QuExpr)
    S = copy(A)
    for (t,s) in B.terms
        _add_with_auto_order!(S,t,-s)
    end
    S
end
-(B::Number,A::QuExpr) = B + (-A)
-(A::QuExpr,B::Number) = A + (-B)
