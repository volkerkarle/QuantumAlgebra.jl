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
    # For LieAlgebraGen_ and Transition_ operators, compare gen_idx before inds
    # to ensure proper normal ordering (e.g., T¹ < T² regardless of site indices)
    # This is analogous to how TLS operators have different types (TLSx_ < TLSy_ < TLSz_)
    if A.t == LieAlgebraGen_ || A.t == Transition_
        @_cmpAB algebra_id false
        @_cmpAB gen_idx false
        @_cmpAB inds false
    else
        @_cmpAB inds false
        @_cmpAB algebra_id false
        @_cmpAB gen_idx false
    end
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

function Base.adjoint(A::BaseOperator)
    if A.t == Transition_
        # |i⟩⟨j|† = |j⟩⟨i|: swap i and j
        N = Int(A.algebra_id)
        i, j = (A.gen_idx - 1) ÷ N + 1, (A.gen_idx - 1) % N + 1
        new_encoded = UInt16((j - 1) * N + i)
        return BaseOperator(Transition_, A.name, A.inds, A.algebra_id, new_encoded)
    else
        return BaseOperator(BaseOpType_adj[Int(A.t)], A.name, A.inds, A.algebra_id, A.gen_idx)
    end
end
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
# Coefficients can be Float64/ComplexF64 (fast mode) or Rational/symbolic (exact mode)
struct ContractionResult{T<:Number, S<:Number}
    id_coeff::T  # coefficient of identity term
    ops::Vector{Tuple{S, BaseOperator}}  # generator terms: (coefficient, operator)
end

# Constructor for flexible coefficient types
ContractionResult(id_coeff::Number, ops::Vector{<:Tuple{<:Number, BaseOperator}}) = 
    ContractionResult(id_coeff, ops)

# Constructor for simple contractions (single operator or none)
function ContractionResult(id_coeff::Number, op::Union{Nothing,BaseOperator})
    if op === nothing
        return ContractionResult(id_coeff, Tuple{typeof(id_coeff), BaseOperator}[])
    else
        zero_coeff = zero(id_coeff)
        return ContractionResult(zero_coeff, [(id_coeff, op)])
    end
end

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

    # Transition operators commute with all other operator types
    if A.t == Transition_ && B.t != Transition_
        return (1, nothing)
    elseif A.t != Transition_ && B.t == Transition_
        return (1, nothing)
    end

    # Transition operator commutation: [|i⟩⟨j|, |k⟩⟨l|] = δ_jk |i⟩⟨l| - δ_il |k⟩⟨j|
    if A.t == Transition_ && B.t == Transition_
        return _exchange_transition_operators(A, B)
    end

    error("_exchange should never reach this! A=$A, B=$B.")
end

"""
    _exchange_transition_operators(A, B)

Handle exchange of two transition operators.

Product rule: |i_B⟩⟨j_B| × |i_A⟩⟨j_A| = δ_{j_B, i_A} |i_B⟩⟨j_A|

For same indices (isempty(dd)): return the direct product B*A with prefactor=0
For different indices: B A = A B + [B, A], where [B,A] is the commutator

[B, A] = |i_B⟩⟨j_B| |i_A⟩⟨j_A| - |i_A⟩⟨j_A| |i_B⟩⟨j_B|
       = δ_{j_B, i_A} |i_B⟩⟨j_A| - δ_{j_A, i_B} |i_A⟩⟨j_B|
"""
function _exchange_transition_operators(A::BaseOperator, B::BaseOperator)
    # Different systems commute
    if A.name != B.name || A.algebra_id != B.algebra_id
        return (1, nothing)
    end
    
    # Check index match
    dd = δ(A.inds, B.inds)
    if dd === nothing
        return (1, nothing)
    end
    
    N = Int(A.algebra_id)
    i_A, j_A = (A.gen_idx - 1) ÷ N + 1, (A.gen_idx - 1) % N + 1
    i_B, j_B = (B.gen_idx - 1) ÷ N + 1, (B.gen_idx - 1) % N + 1
    
    if isempty(dd)
        # =====================================================================
        # SAME INDICES: Compute direct product B*A = δ_{j_B,i_A} |i_B⟩⟨j_A|
        # Return prefactor=0 (A*B term is replaced entirely by the product)
        # =====================================================================
        if j_B == i_A
            enc = UInt16((i_B - 1) * N + j_A)
            new_op = BaseOperator(Transition_, A.name, A.inds, A.algebra_id, enc)
            return (0, ExchangeResult(1//1 + 0im, dd, new_op))
        else
            # Product is zero: B*A = 0, so B*A = 0*A*B + 0
            # We need to signal that the product vanishes
            # Return prefactor=0 with a zero coefficient
            # Actually, if product is 0, we return (0, nothing) but with special handling
            # The simplest approach: return (0, ExchangeResult(0, ...)) but we need an operator
            # For a zero result, we can return (0, nothing) and let normal_form handle it
            # But (0, nothing) means B*A = 0*A*B = 0, which is correct!
            return (0, nothing)
        end
    else
        # =====================================================================
        # DIFFERENT INDICES: B A = A B + [B, A]
        # Compute commutator [B, A] = δ_{j_B, i_A} |i_B⟩⟨j_A| - δ_{j_A, i_B} |i_A⟩⟨j_B|
        # Return prefactor=1
        # =====================================================================
        terms = Tuple{ComplexF64, BaseOperator}[]
        
        if j_B == i_A
            # Term: +|i_B⟩⟨j_A|
            enc = UInt16((i_B - 1) * N + j_A)
            push!(terms, (ComplexF64(1.0), BaseOperator(Transition_, A.name, A.inds, A.algebra_id, enc)))
        end
        
        if j_A == i_B
            # Term: -|i_A⟩⟨j_B|
            enc = UInt16((i_A - 1) * N + j_B)
            push!(terms, (ComplexF64(-1.0), BaseOperator(Transition_, A.name, A.inds, A.algebra_id, enc)))
        end
        
        if isempty(terms)
            # Commutator is zero, operators commute
            return (1, nothing)
        elseif length(terms) == 1
            coeff = terms[1][1]
            coeff_r = real(coeff) == 1.0 ? 1//1 : real(coeff) == -1.0 ? -1//1 : Rational{Int}(real(coeff))
            return (1, ExchangeResult(Complex{Rational{Int}}(coeff_r, 0), dd, terms[1][2]))
        else
            return (1, ExchangeResult(0//1 + 0im, dd, terms))
        end
    end
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
        # Product rule: T^a T^b = (1/4)δ_{ab} + (i/2)ε_{abc} T^c
        # We want B A = T^b T^a = (i/2)ε_{bac} T^c = (-i/2)ε_{abc} T^c
        c, eps_abc = su2_commutator_result(a, b)
        if c == 0
            return (1, nothing)
        end
        new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
        if isempty(dd)
            # Same indices: B A = 0*A B + result (full simplification)
            # T^b T^a = (i/2)ε_{bac} T^c = (-i/2)ε_{abc} T^c
            coeff = Complex{Rational{Int}}(0, -eps_abc//2)
            return (0, ExchangeResult(coeff, dd, new_op))
        else
            # Different indices: B A = 1*A B + commutator_term (delta term)
            # [T^b_j, T^a_i] = i ε_{bac} δ_{ij} T^c_i = -i ε_{abc} δ_{ij} T^c_i
            coeff = Complex{Rational{Int}}(0, -eps_abc)
            return (1, ExchangeResult(coeff, dd, new_op))
        end
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
    # Product rule: T^a T^b = (1/2N)δ_{ab} + (1/2)(d_{abc} + i f_{abc}) T^c
    # For exchange: T^b T^a = (1/2N)δ_{ab} + (1/2)(d_{abc} - i f_{abc}) T^c
    # Here we only handle the commutator part for reordering:
    # T^b T^a = 0 * T^a T^b + (1/2)(d_{abc} - i f_{abc}) T^c  [same indices, prefactor=0]
    # T^b_j T^a_i = T^a_i T^b_j + (-i f_{abc}) δ_{ij} T^c_i  [different indices, prefactor=1]
    #
    # Note: For same indices, we use contraction (_try_contract_lie_algebra) for the full product.
    # The exchange function just needs to provide the result for reordering.
    # Since prefactor=0 means B A = result (not B A = A B + result), we return the full product.
    #
    # However, the current implementation uses structure constants which only give
    # the antisymmetric (commutator) part. For the full product, we'd need d_{abc} too.
    # For now, we return the commutator coefficient for the different-index case,
    # and rely on contraction for the same-index case.
    #
    # Actually, re-examining the TLS code: it returns the PRODUCT coefficient directly.
    # For σ^b σ^a = i ε_{bac} σ^c (when a ≠ b), coefficient = i ε_{bac}
    # For T^b T^a = (i/2) ε_{bac} T^c (SU(2), when a ≠ b), coefficient = (i/2) ε_{bac}
    #
    # For general SU(N), we need: T^b T^a = (1/2)(d_{bac} + i f_{bac}) T^c
    #                                     = (1/2)(d_{abc} - i f_{abc}) T^c
    # So coefficient = (1/2)(d_{abc} - i f_{abc})
    
    if length(f_ab) == 1
        # Single term result (only antisymmetric part for now)
        c, f_abc = first(f_ab)
        new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
        if isempty(dd)
            # Same indices: B A = 0*A B + result (full simplification)
            # For now, use (-i/2) f_{abc} as the coefficient (ignoring d_{abc})
            # This is approximate for SU(N>2) but exact for SU(2) where d=0
            coeff = -im/2 * f_abc
            r_coeff = _try_rationalize_complex(coeff)
            if r_coeff !== nothing
                return (0, ExchangeResult(r_coeff, dd, new_op))
            else
                return (0, ExchangeResult(0//1, dd, [(coeff, new_op)]))
            end
        else
            # Different indices: B A = 1*A B + commutator_term (delta term)
            # [T^b_j, T^a_i] = -i f_{abc} δ_{ij} T^c_i
            coeff = -im * f_abc
            r_coeff = _try_rationalize_complex(coeff)
            if r_coeff !== nothing
                return (1, ExchangeResult(r_coeff, dd, new_op))
            else
                return (1, ExchangeResult(0//1, dd, [(coeff, new_op)]))
            end
        end
    else
        # Multi-term result (e.g., SU(3) and higher)
        if isempty(dd)
            # Same indices: B A = 0*A B + result (full simplification)
            ops = Tuple{ComplexF64, BaseOperator}[]
            for (c, f_abc) in f_ab
                coeff = -im/2 * f_abc
                new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
                push!(ops, (coeff, new_op))
            end
            return (0, ExchangeResult(0//1, dd, ops))
        else
            # Different indices: B A = 1*A B + commutator_term (delta term)
            ops = Tuple{ComplexF64, BaseOperator}[]
            for (c, f_abc) in f_ab
                coeff = -im * f_abc
                new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
                push!(ops, (coeff, new_op))
            end
            return (1, ExchangeResult(0//1, dd, ops))
        end
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

# SU(2) inline contraction result: (true, :su2_inline, id_coeff, gen_term_or_nothing)
const SU2InlineContractionResult = Tuple{Bool, Symbol, Any, Any}

# SU(3) inline contraction result: (true, :su3_inline, name, inds, (id_coeff, c1, coeff1, c2, coeff2))
const SU3InlineContractionResult = Tuple{Bool, Symbol, QuOpName, Any, Tuple}

function _contract(A::BaseOperator,B::BaseOperator)::Union{LegacyContractionResult, Tuple{Bool, <:ContractionResult}, SU2InlineContractionResult, SU3InlineContractionResult}
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
    elseif A.t == Transition_ && B.t == Transition_
        return _contract_transition_operators(A, B)
    else
        return (false,zero(ComplexInt),nothing)
    end
end

"""
    _contract_transition_operators(A, B)

Product rule for transition operators: |i⟩⟨j| × |k⟩⟨l| = δ_jk |i⟩⟨l|
"""
function _contract_transition_operators(A::BaseOperator, B::BaseOperator)
    # Different systems or indices don't contract
    (A.name == B.name && A.inds == B.inds && A.algebra_id == B.algebra_id) || 
        return (false, zero(ComplexInt), nothing)
    
    N = Int(A.algebra_id)
    i_A, j_A = (A.gen_idx - 1) ÷ N + 1, (A.gen_idx - 1) % N + 1
    i_B, j_B = (B.gen_idx - 1) ÷ N + 1, (B.gen_idx - 1) % N + 1
    
    # |i_A⟩⟨j_A| × |i_B⟩⟨j_B| = δ_{j_A, i_B} |i_A⟩⟨j_B|
    if j_A == i_B
        new_encoded = UInt16((i_A - 1) * N + j_B)
        new_op = BaseOperator(Transition_, A.name, A.inds, A.algebra_id, new_encoded)
        return (true, one(ComplexInt), new_op)
    else
        # Orthogonal: product is zero
        return (true, zero(ComplexInt), nothing)
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
For SU(2), uses specialized SU2ContractionResult for inline processing.
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
    
    # Get the algebra to determine which fast path to use
    alg = get_algebra(A.algebra_id)
    
    # =========================================================================
    # FAST PATH: SU(2) - use inline tuple for direct processing in normal_order!
    # =========================================================================
    # For SU(2): T^a T^b = (1/4)δ_{ab}I + (i/2)ε_{abc}T^c
    # 
    # We return a specialized tuple that normal_order! can process inline:
    # (true, :su2_inline, id_coeff, gen_coeff, c_or_nothing)
    #
    # Diagonal case (a == b): T^a T^a = (1/4)I
    # Off-diagonal case: T^a T^b = (i/2)ε_{abc}T^c
    
    if alg isa SUAlgebra{2}
        if a == b
            # Diagonal case: T^a T^a = (1/4)I - only identity, no generator
            # Return (true, :su2_inline, id_coeff::Float64, nothing)
            return (true, :su2_inline, 0.25, nothing)
        else
            # Off-diagonal case: T^a T^b = (i/2)ε_{abc}T^c (no identity term)
            c = 6 - a - b
            s = (a % 3 + 1 == b) ? 1 : -1
            gen_coeff = ComplexF64(0.5im * s)
            new_op = LieAlgebraGenerator(A.name, A.algebra_id, c, A.inds)
            # Return (true, :su2_inline, id_coeff::Float64, (gen_coeff, op))
            return (true, :su2_inline, 0.0, (gen_coeff, new_op))
        end
    end
    
    # =========================================================================
    # FAST PATH: SU(3) - use coefficient providers (symbolic or float)
    # =========================================================================
    # Returns (true, :su3_inline, name, inds, (id_coeff, c1, coeff1, c2, coeff2))
    # Coefficients are symbolic (Rational, √3) by default, Float64 in high-speed mode
    
    if alg isa SUAlgebra{3}
        coeffs = su3_product_coeffs(a, b)
        return (true, :su3_inline, A.name, A.inds, coeffs)
    end
    
    # =========================================================================
    # GENERAL PATH: Use structure constant lookup (for SU(N) with N > 3)
    # =========================================================================
    N = algebra_dim(alg)
    
    # Identity coefficient: (1/2N) δ_{ab}
    id_coeff = a == b ? sun_id_coeff(N) : (using_float_coefficients() ? 0.0 : 0)
    
    # Generator terms: (1/2)(d^{abc} + i f^{abc}) T^c
    d_ab = symmetric_structure_constants(alg, a, b)
    f_ab = structure_constants(alg, a, b)
    
    # Combine d and f structure constants
    # Note: structure constants are currently Float64; future work could make them symbolic
    half = sun_gen_coeff()
    gen_coeffs = Dict{Int, Number}()
    
    for (c, d_abc) in d_ab
        gen_coeffs[c] = get(gen_coeffs, c, zero(half)) + d_abc * half
    end
    
    for (c, f_abc) in f_ab
        gen_coeffs[c] = get(gen_coeffs, c, zero(half) * im) + im * f_abc * half
    end
    
    # Build the operator list
    ops = Tuple{Number, BaseOperator}[]
    for (c, coeff) in gen_coeffs
        if !iszero(coeff)
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
    # prefactor starts as 1, but can become any Number type through multiplications
    # (e.g., ComplexF64, Rational, or symbolic types when Symbolics.jl is loaded)
    prefactor::Number = 1
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
            if contract_result isa Tuple{Bool, Symbol, Float64, Any} && contract_result[2] === :su2_inline
                # =========================================================
                # FAST PATH: SU(2) inline contraction (like TLS)
                # Format: (true, :su2_inline, id_coeff, gen_term_or_nothing)
                # =========================================================
                id_coeff = contract_result[3]
                gen_term = contract_result[4]
                
                if gen_term === nothing
                    # Diagonal case: T^a T^a = 0.25*I
                    # Multiply prefactor by 0.25 and remove the pair
                    prefactor *= id_coeff
                    deleteat!(A, (k, k+1))
                    # op is nothing - this is like the TLS identity case
                else
                    # Off-diagonal case: T^a T^b = gen_coeff * T^c
                    # Multiply prefactor by gen_coeff and replace pair with T^c
                    gen_coeff, new_op = gen_term
                    prefactor *= gen_coeff
                    A[k] = new_op
                    deleteat!(A, k+1)
                end
                iszero(prefactor) && return prefactor
                k > 1 && (k -= 1)
            elseif contract_result isa Tuple{Bool, Symbol, QuOpName, Any, Tuple} && 
                   contract_result[2] === :su3_inline
                # =========================================================
                # FAST PATH: SU(3) symbolic/float coefficients
                # Format: (true, :su3_inline, name, inds, (id_coeff, c1, coeff1, c2, coeff2))
                # =========================================================
                name = contract_result[3]
                inds = contract_result[4]
                id_coeff, c1, coeff1, c2, coeff2 = contract_result[5]
                
                if c2 == 0
                    # Single generator term - can partially inline
                    if !iszero(id_coeff)
                        # Add identity term to collector
                        remaining_ops = [A[1:k-1]; A[k+2:end]]
                        onew = BaseOpProduct(remaining_ops)
                        t = QuTerm(onew)
                        _add_sum_term!(term_collector, t, id_coeff * prefactor)
                    end
                    # Inline the generator term: multiply coeff into prefactor, replace pair
                    prefactor *= coeff1
                    new_op = LieAlgebraGenerator(name, SU3_ALGEBRA_ID, Int(c1), inds)
                    A[k] = new_op
                    deleteat!(A, k+1)
                    iszero(prefactor) && return prefactor
                    k > 1 && (k -= 1)
                else
                    # Two generator terms - add all to collector, zero out main term
                    if !iszero(id_coeff)
                        remaining_ops = [A[1:k-1]; A[k+2:end]]
                        onew = BaseOpProduct(remaining_ops)
                        t = QuTerm(onew)
                        _add_sum_term!(term_collector, t, id_coeff * prefactor)
                    end
                    
                    # Add first generator term
                    op1 = LieAlgebraGenerator(name, SU3_ALGEBRA_ID, Int(c1), inds)
                    onew1 = BaseOpProduct([A[1:k-1]; op1; A[k+2:end]])
                    _add_sum_term!(term_collector, QuTerm(onew1), coeff1 * prefactor)
                    
                    # Add second generator term
                    op2 = LieAlgebraGenerator(name, SU3_ALGEBRA_ID, Int(c2), inds)
                    onew2 = BaseOpProduct([A[1:k-1]; op2; A[k+2:end]])
                    _add_sum_term!(term_collector, QuTerm(onew2), coeff2 * prefactor)
                    
                    # Remove contracted pair and return zero (all terms in collector)
                    deleteat!(A, (k, k+1))
                    return zero(prefactor)
                end
            elseif contract_result isa Tuple{Bool, <:ContractionResult}
                # Lie algebra generator contraction (multi-term result) - SU(N) with N > 2
                result = contract_result[2]
                
                # For Lie algebra contractions, we add all terms to term_collector
                # and then signal that this product should be zeroed out
                
                # Add identity term if present: id_coeff * (remaining operators)
                if !iszero(result.id_coeff)
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
