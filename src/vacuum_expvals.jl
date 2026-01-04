export Avac, vacA, vacExpVal

function _get_mode_name(ex::QuExpr)
    opprod = to_opprod(ex)
    op = unalias(only(opprod.v))
    isempty(assignedinds(op.inds)) || throw(ArgumentError("Modes in vacuum should be given without indices"))
    op.name
end

_parse_modes_in_vacuum(modes::Nothing) = nothing
_parse_modes_in_vacuum(modes::Set{QuOpName}) = modes
_parse_modes_in_vacuum(modes::QuExpr) = _parse_modes_in_vacuum((modes,))
_parse_modes_in_vacuum(modes) = Set{QuOpName}(_get_mode_name.(modes))

# ============================================================================
# Vacuum eigenvalues for Lie algebra generators
# ============================================================================
# For SU(N) in the fundamental representation, the vacuum |0⟩ is the state |N⟩
# (the N-th basis state, i.e., the lowest weight state).
#
# For diagonal generators (the last N-1 generators in our ordering), the vacuum
# has a well-defined eigenvalue. For off-diagonal generators, acting on |0⟩
# either gives 0 or a different state.
#
# Specifically for our Gell-Mann matrix ordering:
# - Off-diagonal symmetric (indices 1 to (N-1)N/2): mix states, no eigenvalue on vacuum
# - Off-diagonal antisymmetric (next (N-1)N/2): mix states, no eigenvalue on vacuum  
# - Diagonal (last N-1): have eigenvalues on the vacuum state |N⟩

"""
    _lie_gen_vacuum_eigenvalue(algebra_id, gen_idx)

Return the eigenvalue of T^{gen_idx} on the vacuum state |0⟩ for the given algebra,
or `nothing` if the generator does not have |0⟩ as an eigenstate.

For SU(N), the vacuum is |N⟩ (the N-th basis state).
"""
function _lie_gen_vacuum_eigenvalue(algebra_id::UInt16, gen_idx::UInt16)
    alg = get_algebra(algebra_id)
    N = algebra_dim(alg)
    ngen = N^2 - 1
    n_offdiag = N * (N - 1)  # total off-diagonal generators (symmetric + antisymmetric)
    
    # Off-diagonal generators don't have vacuum as eigenstate
    if gen_idx <= n_offdiag
        return nothing
    end
    
    # Diagonal generator: index l = gen_idx - n_offdiag (1 to N-1)
    l = Int(gen_idx) - n_offdiag
    
    # For the l-th diagonal generator acting on |N⟩:
    # T_l = sqrt(1/(2l(l+1))) * (sum_{j=1}^l |j⟩⟨j| - l|l+1⟩⟨l+1|)
    # For |N⟩: if l+1 < N, coefficient is 0. If l+1 = N, coefficient is -l * norm
    # If l+1 > N (impossible since l ≤ N-1), no contribution
    
    norm = sqrt(1.0 / (2 * l * (l + 1)))
    if l + 1 == N
        # |N⟩ gets coefficient -l * norm
        return -l * norm
    elseif l + 1 < N
        # |N⟩ is not affected by this generator
        return 0.0
    else
        # Should not happen
        return 0.0
    end
end

"""
    _lie_gen_annihilates_vacuum(algebra_id, gen_idx)

Return true if T^{gen_idx} |0⟩ = 0 (lowering-type operator on vacuum).

For SU(N), generators that lower the weight annihilate the vacuum.
These are the antisymmetric off-diagonal generators for pairs (j, N) with j < N,
which correspond to "lowering" from |j⟩ to |N⟩ or vice versa.
"""
function _lie_gen_annihilates_vacuum(algebra_id::UInt16, gen_idx::UInt16)
    alg = get_algebra(algebra_id)
    N = algebra_dim(alg)
    n_sym_offdiag = div(N * (N - 1), 2)
    
    # For antisymmetric off-diagonal: indices n_sym_offdiag+1 to 2*n_sym_offdiag
    # These correspond to pairs (row, col) with row < col
    # The pair (j, N) for j < N corresponds to lowering operators
    
    if gen_idx <= n_sym_offdiag
        # Symmetric off-diagonal: check if it involves |N⟩
        # These are (|j⟩⟨k| + |k⟩⟨j|)/2 for j < k
        # When acting on |N⟩: gives |j⟩/2 if k=N, 0 otherwise
        # So it doesn't annihilate but also doesn't keep |N⟩
        idx = Int(gen_idx)
        count = 0
        for col in 2:N
            for row in 1:col-1
                count += 1
                if count == idx
                    # This is the (row, col) pair
                    if col == N
                        # Acts on |N⟩, produces |row⟩, doesn't annihilate
                        return false
                    else
                        # Doesn't act on |N⟩, gives 0
                        return true
                    end
                end
            end
        end
    elseif gen_idx <= 2 * n_sym_offdiag
        # Antisymmetric off-diagonal: -i(|j⟩⟨k| - |k⟩⟨j|)/2 for j < k
        idx = Int(gen_idx) - n_sym_offdiag
        count = 0
        for col in 2:N
            for row in 1:col-1
                count += 1
                if count == idx
                    if col == N
                        # Acts on |N⟩, produces -i|row⟩/2, doesn't annihilate
                        return false
                    else
                        # Doesn't act on |N⟩
                        return true
                    end
                end
            end
        end
    end
    
    # Diagonal generators don't annihilate
    return false
end

"""
    _lie_gen_vacuum_expval(algebra_id, gen_idx)

Return the vacuum expectation value ⟨0|T^{gen_idx}|0⟩ for the given algebra.

For SU(N), the vacuum is |N⟩. Diagonal generators have expectation value equal
to their eigenvalue. Off-diagonal generators have zero expectation value because
they connect |N⟩ to orthogonal states.
"""
function _lie_gen_vacuum_expval(algebra_id::UInt16, gen_idx::UInt16)
    # Diagonal generators: return eigenvalue
    eigenval = _lie_gen_vacuum_eigenvalue(algebra_id, gen_idx)
    if eigenval !== nothing
        return eigenval
    end
    # Off-diagonal generators: return 0 (they produce states orthogonal to vacuum)
    return 0.0
end

"""
    _lie_gen_vacuum_annihilates_left(algebra_id, gen_idx)

Return true if ⟨0| T^{gen_idx} = 0, i.e., the generator produces a state 
orthogonal to the vacuum when acting from the left (or equivalently, the 
adjoint produces zero when acting on the vacuum from the right).

For Hermitian generators T^a† = T^a, we have ⟨0|T^a = (T^a|0⟩)†.
If T^a|0⟩ produces a state orthogonal to |0⟩, then ⟨0|T^a does NOT give zero
but rather gives a bra orthogonal to ⟨0|.

For SU(N), symmetric off-diagonal generators (j,k) with k=N map |N⟩ → |j⟩,
and ⟨N| → ⟨j|, so they don't annihilate from either side.
The ones that annihilate from the right (col ≠ N) also annihilate from the left
since they don't involve |N⟩ at all.
"""
function _lie_gen_vacuum_annihilates_left(algebra_id::UInt16, gen_idx::UInt16)
    # For Hermitian operators and vacuum expectation, the logic is the same:
    # if the generator doesn't touch |N⟩, it gives zero from both sides
    # This is the same as _lie_gen_annihilates_vacuum for our symmetric generators
    return _lie_gen_annihilates_vacuum(algebra_id, gen_idx)
end

# ============================================================================
# Vacuum expectation values for transition operators |i⟩⟨j|
# ============================================================================
# For N-level systems, vacuum is |N⟩ (ground state = lowest level)
# - |N⟩⟨N| |N⟩ = |N⟩, so it's identity on vacuum: eigenvalue = 1
# - |i⟩⟨N| |N⟩ = |i⟩ for i≠N: raises state, doesn't annihilate
# - |N⟩⟨j| |N⟩ = 0 for j≠N: requires ⟨j| overlap with |N⟩
# - |i⟩⟨j| |N⟩ = 0 for j≠N: same reason

"""Return eigenvalue if |N⟩ is eigenstate, nothing otherwise."""
function _transition_vacuum_eigenvalue(N::Int, i::Int, j::Int)
    (i == N && j == N) ? 1.0 : nothing
end

"""Return true if |i⟩⟨j| annihilates vacuum |N⟩."""
function _transition_annihilates_vacuum(N::Int, i::Int, j::Int)
    j != N  # |i⟩⟨j| |N⟩ = δ_{jN} |i⟩, so zero when j≠N
end

"""Return true if ⟨N| |i⟩⟨j| = 0."""
function _transition_annihilates_left(N::Int, i::Int, j::Int)
    i != N  # ⟨N| |i⟩⟨j| = δ_{Ni} ⟨j|, so zero when i≠N
end

# helper function that is only called on normal-ordered terms that only contain
# operators that are in the vacuum state
function _Avac(BO::BaseOpProduct,fac)
    ii = length(BO.v)
    while ii >= 1
        O = BO.v[ii]
        if O.t in (BosonCreate_,FermionCreate_,TLSCreate_)
            ii==length(BO.v) && return (BO,fac)
            break
        elseif O.t in (BosonDestroy_,FermionDestroy_,TLSDestroy_)
            return (BO,zero(fac))
        elseif O.t == TLSz_
            # vacuum is eigenvalue of σz - since the term is in normal order and
            # we break at TLSx_,TLSy_, we know it acts on the vacuum
            fac = -fac
        elseif O.t in (TLSx_,TLSy_)
            break
        elseif O.t == LieAlgebraGen_
            # Check if this generator annihilates the vacuum
            if _lie_gen_annihilates_vacuum(O.algebra_id, O.gen_idx)
                return (BO, zero(fac))
            end
            # Check if vacuum is an eigenstate of this generator
            eigenval = _lie_gen_vacuum_eigenvalue(O.algebra_id, O.gen_idx)
            if eigenval !== nothing
                # Diagonal generator: multiply by eigenvalue and remove operator
                fac = fac * eigenval
            else
                # Off-diagonal generator that doesn't annihilate: produces different state
                # Can't simplify further, keep operator
                break
            end
        elseif O.t == Transition_
            # Transition operator |i⟩⟨j| acting on vacuum |N⟩
            N = Int(O.algebra_id)
            i, j = (O.gen_idx - 1) ÷ N + 1, (O.gen_idx - 1) % N + 1
            if _transition_annihilates_vacuum(N, i, j)
                return (BO, zero(fac))
            end
            eigenval = _transition_vacuum_eigenvalue(N, i, j)
            if eigenval !== nothing
                fac = fac * eigenval
            else
                # Raises to different state, can't simplify
                break
            end
        else
            error("should not be reached")
        end
        ii -= 1
    end
    return (BaseOpProduct(BO.v[1:ii]), fac)
end

# helper function that is only called on normal-ordered terms that only contain
# operators that are in the vacuum state
function _vacA(BO::BaseOpProduct,fac)
    ii = 1
    while ii <= length(BO.v)
        O = BO.v[ii]
        if O.t in (BosonDestroy_,FermionDestroy_,TLSDestroy_)
            ii==1 && return (BO,fac)
            break
        elseif O.t in (BosonCreate_,FermionCreate_,TLSCreate_)
            return (BO,zero(fac))
        elseif O.t == TLSz_
            # vacuum is eigenvalue of σz - since the term is in normal order and
            # we break at TLSx_,TLSy_, we know it acts on the vacuum
            fac = -fac
        elseif O.t in (TLSx_,TLSy_)
            break
        elseif O.t == LieAlgebraGen_
            # For ⟨0|T^a, since generators are Hermitian (T^a† = T^a):
            # ⟨0|T^a = (T^a|0⟩)†
            # If the generator annihilates from the left, return zero
            if _lie_gen_vacuum_annihilates_left(O.algebra_id, O.gen_idx)
                return (BO, zero(fac))
            end
            # Check if vacuum is an eigenstate
            eigenval = _lie_gen_vacuum_eigenvalue(O.algebra_id, O.gen_idx)
            if eigenval !== nothing
                # Diagonal generator: multiply by eigenvalue (real since Hermitian)
                fac = fac * eigenval
            else
                # Off-diagonal generator: produces different state, can't simplify
                break
            end
        elseif O.t == Transition_
            # ⟨N| |i⟩⟨j| = δ_{Ni} ⟨j|
            N = Int(O.algebra_id)
            i, j = (O.gen_idx - 1) ÷ N + 1, (O.gen_idx - 1) % N + 1
            if _transition_annihilates_left(N, i, j)
                return (BO, zero(fac))
            end
            eigenval = _transition_vacuum_eigenvalue(N, i, j)
            if eigenval !== nothing
                fac = fac * eigenval
            else
                break
            end
        else
            error("should not be reached")
        end
        ii += 1
    end
    return (BaseOpProduct(BO.v[ii:end]), fac)
end

function _apply_on_vacuum_modes(func,A::QuTerm,fac,modes_in_vacuum::Nothing)
    BO, fac = func(A.bares,fac)
    Anew = BO === A.bares ? A : QuTerm(A.nsuminds,A.δs,A.params,A.expvals,A.corrs,BO)
    return (Anew, fac)
end

function _apply_on_vacuum_modes(func,A::QuTerm,fac,modes_in_vacuum::Set{QuOpName})
    BO_vacmodes = BaseOpProduct()
    BO_novacmodes = BaseOpProduct()
    for O in A.bares.v
        Oua = unalias(O)
        BO = Oua.name ∈ modes_in_vacuum ? BO_vacmodes : BO_novacmodes
        push!(BO.v, O)
    end
    BO_vacnew, fac = func(BO_vacmodes,fac)
    BO_vacnew === BO_vacmodes && return (A,fac)
    # we know that operators were normal ordered and our operations did not
    # change this, so just sort without commutations
    newv = sort!([BO_novacmodes.v; BO_vacnew.v])
    Anew = QuTerm(A.nsuminds,A.δs,A.params,A.expvals,A.corrs,BaseOpProduct(newv))
    return (Anew, fac)
end

Avac(A::QuTerm,fac,modes_in_vacuum) = _apply_on_vacuum_modes(_Avac, A, fac, modes_in_vacuum)
vacA(A::QuTerm,fac,modes_in_vacuum) = _apply_on_vacuum_modes(_vacA, A, fac, modes_in_vacuum)

Avac(A::QuExpr,modes_in_vacuum=nothing) = (mv = _parse_modes_in_vacuum(modes_in_vacuum); QuExpr(Avac(t,s,mv) for (t,s) in normal_form(A).terms))
vacA(A::QuExpr,modes_in_vacuum=nothing) = (mv = _parse_modes_in_vacuum(modes_in_vacuum); QuExpr(vacA(t,s,mv) for (t,s) in normal_form(A).terms))
Avac(s::Number,modes_in_vacuum=nothing) = QuExpr(s)
vacA(s::Number,modes_in_vacuum=nothing) = QuExpr(s)


"""
    Avac(A::QuExpr), vacA(A::QuExpr)

Simplify operator by assuming it is applied to the vacuum from the left or
right, respectively. To be precise, `Avac(A)` returns ``A'`` such that ``A'|0⟩ =
A|0⟩``, while `vacA(A)` does the same for ``⟨0|A``."""
Avac, vacA

function _TLS_to_pm_normal(A::QuExpr,shortcut_vacA_zero)
    An = QuExpr()
    terms_to_clean = collect(A.terms)
    while !isempty(terms_to_clean)
        (t,s) = pop!(terms_to_clean)
        v = t.bares.v
        iTLS = findfirst(A->A.t ∈ (TLSx_,TLSy_,TLSz_), v)
        if iTLS === nothing
            _add_with_normal_order!(An,t,s,shortcut_vacA_zero) # last argument is shortcut_vacA_zero
        else
            O = v[iTLS]
            if O.t == TLSx_
                # σˣ = σ⁺ + σ⁻
                for On in (TLSCreate(O.name,O.inds),TLSDestroy(O.name,O.inds))
                    vn = [v[1:iTLS-1]; On; v[iTLS+1:end]]
                    tn = QuTerm(t.nsuminds,t.δs,t.params,t.expvals,t.corrs,BaseOpProduct(vn))
                    push!(terms_to_clean,tn=>s)
                end
            elseif O.t == TLSy_
                # σʸ = -i σ⁺ + i σ⁻
                for (On,sn) in ((TLSCreate(O.name,O.inds),-1im),(TLSDestroy(O.name,O.inds),1im))
                    vn = [v[1:iTLS-1]; On; v[iTLS+1:end]]
                    tn = QuTerm(t.nsuminds,t.δs,t.params,t.expvals,t.corrs,BaseOpProduct(vn))
                    push!(terms_to_clean,tn=>s*sn)
                end
            elseif O.t == TLSz_
                # σᶻ = 2σ⁺σ⁻ - 1
                for (On,sn) in (([TLSCreate(O.name,O.inds),TLSDestroy(O.name,O.inds)],2),(BaseOperator[],-1))
                    vn = [v[1:iTLS-1]; On; v[iTLS+1:end]]
                    tn = QuTerm(t.nsuminds,t.δs,t.params,t.expvals,t.corrs,BaseOpProduct(vn))
                    push!(terms_to_clean,tn=>s*sn)
                end
            end
        end
    end
    An
end

"""
    vacExpVal(A::QuExpr,S::QuExpr=1)

Calculate the vacuum expectation value ``⟨0|S^\\dagger A S|0⟩``, i.e., the
expectation value ``⟨ψ|A|ψ⟩`` for the state defined by ``|ψ⟩= S|0⟩```.
"""
function vacExpVal(A::QuExpr,stateop::QuExpr=QuExpr(QuTerm()),modes_in_vacuum=nothing)
    # simplify down as much as possible by applying vacuum from left and right
    # convert TLSx/y/z operators to TLSCreate/TLSDestroy to ensure that no bare operators survive

    mv = _parse_modes_in_vacuum(modes_in_vacuum)

    # since we will have <vac|stateop' * A * stateop|vac>, we can simplify
    # stateop by applying vacuum from right
    stateop = Avac(stateop,mv)
    # second argument is shortcut_vacA_zero, if mv is nothing, we can set it to true
    x = normal_form(stateop' * A, isnothing(mv))
    # same here, simplify since we will have <vac|stateop' * A
    x = vacA(x,mv)
    x = _TLS_to_pm_normal(x*stateop, isnothing(mv)) # second argument is shortcut_vacA_zero

    vAv = QuExpr()
    for (t,s) in x.terms
        tn,sn = Avac(vacA(t,s,mv)...,mv)
        if isnothing(mv)
            # Check if remaining operators are all Lie algebra generators
            # For such operators, compute their vacuum expectation value directly
            if !isempty(tn.bares) && sn != 0
                # Check if all remaining operators are Lie algebra generators
                all_lie = all(op -> op.t == LieAlgebraGen_, tn.bares.v)
                if all_lie
                    # Compute the vacuum expectation value of the product
                    # For now, only handle single generators
                    if length(tn.bares.v) == 1
                        op = tn.bares.v[1]
                        lie_vev = _lie_gen_vacuum_expval(op.algebra_id, op.gen_idx)
                        sn = sn * lie_vev
                        tn = QuTerm(tn.nsuminds, tn.δs, tn.params, tn.expvals, tn.corrs, BaseOpProduct())
                    else
                        # Multiple Lie generators remaining - this is more complex
                        # For now, we need to compute ⟨0|T^a T^b ... |0⟩
                        # This requires expanding using the product rule
                        # For simplicity, return zero for off-diagonal combinations
                        # and handle diagonal combinations
                        # TODO: Implement proper multi-generator vacuum expectation
                        sn = zero(sn)
                        tn = QuTerm(tn.nsuminds, tn.δs, tn.params, tn.expvals, tn.corrs, BaseOpProduct())
                    end
                else
                    @assert false "Expected all remaining operators to be Lie algebra generators or empty"
                end
            end
        end
        _add_sum_term!(vAv,tn,simplify_number(sn))
    end
    vAv
end

vacExpVal(A::QuExpr,stateop::Number,modes_in_vacuum=nothing) = vacExpVal(A,QuExpr(stateop),modes_in_vacuum)
vacExpVal(A::Number,stateop=1,modes_in_vacuum=nothing) = vacExpVal(QuExpr(A),stateop,modes_in_vacuum)
