using Preferences
using OrderedCollections

export QuExpr
export boson_ops, fermion_ops
export @boson_ops, @fermion_ops, @anticommuting_fermion_group
export tlspm_ops, tlsxyz_ops
export @tlspm_ops, @tlsxyz_ops
export @Pr_str, @Pc_str, ∑
export param, expval, corr
export map_scalar_function
export LieAlgebraGenerator, is_lie_algebra_gen, su_generator, su_generators,
       so_generator, so_generators, sp_generator, sp_generators
export TransitionOperator, is_transition_op, nlevel_ops, σ

# compile-time options
const _DEFINE_DEFAULT_OPS = @load_preference("define_default_ops", true)

"""
    set_define_default_ops(t::Bool)

Set the preference for whether to define default operators (`a`, `adag`, `f`, `fdag`, `σx`, `σy`, `σz`, `σp`, `σm`) upon import.
Default is `true`.

Note that changing this preference requires restarting the Julia session to take effect.
"""
function set_define_default_ops(t::Bool)
    @set_preferences!("define_default_ops" => t)
    if t != _DEFINE_DEFAULT_OPS
        @info("define_default_ops setting changed to $t; restart your Julia session for this change to take effect!")
    end
end

# compile-time options
const _QUINDICES_TYPE = @load_preference("quindices_type", "Vector")

"""
    set_quindices_type(t::String)

Set the preference for the underlying type used to store indices. `t` must be
either `"Vector"` (default) or `"NTuple{N}"` (where `N` is an integer).
- `"Vector"`: Uses `Vector{QuIndex}`. Allows arbitrary number of indices but is slower due to heap allocation.
- `"NTuple{N}"`: Uses `NTuple{N,QuIndex}`. Faster (stack allocation) but limits the number of indices per operator to a maximum of `N`.

Note that changing this preference requires restarting the Julia session to take effect.
"""
function set_quindices_type(t::String)
    parse_quindices_type(t)
    @set_preferences!("quindices_type" => t)
    if t != _QUINDICES_TYPE
        @info("quindices_type setting changed to $t; restart your Julia session for this change to take effect!")
    end
end


# dynamically changeable options
const _using_σpm = Ref(false)
"""
    use_σpm(t::Bool=true; set_preference=false)

Select whether TLS operators are expressed in the `(σ₊, σ₋)` basis (`true`) or
in the `(σx, σy, σz)` basis (`false`). If `set_preference` is `true`, the choice
is persisted via `Preferences.jl` for future sessions.
"""
function use_σpm(t::Bool=true; set_preference=false)
    _using_σpm[] = t
    set_preference && @set_preferences!("use_σpm" => t)
    nothing
end
"""
    use_σxyz(; set_preference=false)

Convenience helper to switch to the `(σx, σy, σz)` basis. Equivalent to
`use_σpm(false; set_preference)`.
"""
use_σxyz(;set_preference=false) = use_σpm(false;set_preference)
"""
    using_σpm()

Return whether TLS operators are currently configured to use the `(σ₊, σ₋)`
basis.
"""
using_σpm() = _using_σpm[]

const _auto_normal_form = Ref(false)
"""
    auto_normal_form(t::Bool=true; set_preference=false)

Enable or disable automatic normal-ordering of expressions on construction.
When `set_preference` is `true`, the setting is stored with `Preferences.jl`.
"""
function auto_normal_form(t::Bool=true; set_preference=false)
    _auto_normal_form[] = t
    set_preference && @set_preferences!("auto_normal_form" => t)
    nothing
end
"""
    using_auto_normal_form()

Return whether automatic normal-ordering is currently enabled.
"""
using_auto_normal_form() = _auto_normal_form[]

# High-speed mode: use Float64 coefficients instead of symbolic (Rational/√3)
# Default is false (symbolic mode), which gives exact results
# Set to true for faster computation with floating-point approximations
const _use_float_coefficients = Ref(false)
function use_float_coefficients(t::Bool=true; set_preference=false)
    _use_float_coefficients[] = t
    set_preference && @set_preferences!("use_float_coefficients" => t)
    nothing
end
using_float_coefficients() = _use_float_coefficients[]

const IndexInt = Int32

@concrete struct QuIndex
    # sym is the index character, apart from two special values:
    # '\0' for integer indices, which ensures they are ordered before any other indices
    # '#' for sum indices, which ensures that they are ordered before letters etc, and
    sym::Char
    # num is integer subindex, special value typemin(IndexInt) means no subindex
    num::IndexInt
    QuIndex(sym::Char,num::Integer=typemin(IndexInt)) = new(sym,num)
end

"""
    QuIndex

Represents an index in a quantum operator expression.

A `QuIndex` can be:
- An integer index (created with `QuIndex(i)` for integer `i`)
- A symbolic index like `QuIndex(:i)` or `QuIndex("i_2")`
- A sum index created with `sumindex(n)`

# Examples
```julia
i = QuIndex(1)              # Integer index 1
i = QuIndex(:i)             # Symbolic index 'i'
i = QuIndex("i_2")          # Symbolic index 'i' with subscript 2
si = sumindex(1)            # Sum index #₁
```

# See also
[`sumindex`](@ref), [`issumindex`](@ref), [`isintindex`](@ref)
"""
QuIndex
QuIndex(ii::QuIndex) = ii
QuIndex(ii::Symbol) = QuIndex(string(ii))
function QuIndex(ii::String)
    s = split(ii,"_")
    length(s) <= 2 || throw(ArgumentError("index can have at most one subindex, got $ii with subindices $(s[2:end])"))
    length(s[1]) == 1 || throw(ArgumentError("index names must be single character, got $ii."))
    sym = s[1][1]
    # we do not allow the ascii control characters as symbolic indices to ensure ordering
    sym > '#' || throw(ArgumentError("Symbolic index name cannot be character less than '#' (codepoint 35), passed Int(sym) = $(Int(sym))."))
    if length(s)==2
        num = try
            parse(IndexInt,s[2])
        catch
            throw(ArgumentError("Only integer literals are supported as index subscripts. Got \"$(s[2])\"."))
        end
        QuIndex(sym,num)
    else
        QuIndex(sym)
    end
end
QuIndex(ii::Integer) = QuIndex('\0',ii)

"""
    sumindex(i)

Create a summation index with index `i`. These indices are used internally for
implicit summation (Einstein summation convention).
"""
sumindex(ii) = QuIndex('#',ii)
# by convention, these should never be present in the input or output of any function,
# but just as temporary variables that cannot conflict with any other within a computation
tmpindex(ii) = QuIndex('!',ii)

"""
    isintindex(i::QuIndex)

Check if the index `i` is an integer index (created with `QuIndex(i::Integer)`).
"""
isintindex(ii::QuIndex) = ii.sym=='\0'

"""
    issumindex(i::QuIndex)

Check if the index `i` is a summation index (created with `sumindex(i)`).
"""
issumindex(ii::QuIndex) = ii.sym=='#'
const NoIndex = QuIndex(typemin(IndexInt))
isnoindex(ii::QuIndex) = ii == NoIndex

@inline Base.isless(i1::QuIndex,i2::QuIndex) = isless((i1.sym,i1.num),(i2.sym,i2.num))

function parse_quindices_type(t::String)
    if t == "Vector"
        return Vector{QuIndex}
    elseif startswith(t,"NTuple")
        m = match(r"NTuple\{(\d+)\}", t)
        m !== nothing || throw(ArgumentError("Invalid quindices_type specification: $t"))
        N = parse(Int, m.captures[1])
        return NTuple{N,QuIndex}
    else
        throw(ArgumentError("quindices_type has to be one of \"Vector\" or \"NTuple{N}\", got \"$t\"."))
    end
end

const QuIndices = parse_quindices_type(_QUINDICES_TYPE)
@static if QuIndices == Vector{QuIndex}
    const _QUINDICES_LENGTH = typemax(Int)
    assignedinds(inds::QuIndices) = inds
    indcopy(inds::QuIndices) = copy(inds)
    Base.tail(inds::QuIndices) = inds[2:end]
    make_indices(inds...)::QuIndices = [QuIndex(i) for i in inds]
elseif QuIndices <: NTuple
    # no need to define Base.tail for NTuple
    assignedinds(inds::QuIndices) = filter(!isnoindex,inds)
    indcopy(inds::QuIndices) = inds
    let
        tuple_len(::Type{NTuple{N,T}}) where {N,T} = N
        global const _QUINDICES_LENGTH = tuple_len(QuIndices)
        syms = Symbol.(:i, 1:_QUINDICES_LENGTH)
        args = Expr.(:kw, syms, :NoIndex)
        body = Expr(:tuple, Expr.(:call, :QuIndex, syms)...)
        @eval make_indices($(args...))::QuIndices = $body
    end
else
    error("Unexpected QuIndices type")
end

make_indices(inds::QuIndices) = inds
make_indices(inds::Union{Vector,Tuple}) = make_indices(inds...)

const _NameTable = Dict{Symbol,IndexInt}()
const _NameTableInv = Symbol[]
const _NAME_TABLE_LOCK = ReentrantLock()

@concrete struct QuOpName
    i::IndexInt
    function QuOpName(name::Symbol)
        i = lock(_NAME_TABLE_LOCK) do
            get!(_NameTable,name) do
                push!(_NameTableInv,name)
                length(_NameTableInv)
            end
        end
        new(i)
    end
    QuOpName(name::QuOpName) = name
end
sym(ind::QuOpName) = lock(_NAME_TABLE_LOCK) do
    _NameTableInv[ind.i]
end
Base.print(io::IO, ind::QuOpName) = print(io, sym(ind))
Base.isless(i1::QuOpName,i2::QuOpName) = isless(sym(i1),sym(i2))
const NoName = QuOpName(Symbol())

# the enum also directly defines a natural ordering,so choose this directly how we later want it
# start the counting at 1 so we can index into the tuples defined below with Int(BaseOpType)
@enum BaseOpType        BosonCreate_=1 FermionCreate_   TLSCreate_   TLSx_  TLSy_  TLSz_  TLSDestroy_ FermionDestroy_ BosonDestroy_ LieAlgebraGen_ Transition_
const BaseOpType_adj = (BosonDestroy_, FermionDestroy_, TLSDestroy_, TLSx_, TLSy_, TLSz_, TLSCreate_, FermionCreate_, BosonCreate_, LieAlgebraGen_, Transition_)
const BaseOpType_sym  = ("†", "†", "⁺", "ˣ", "ʸ", "ᶻ", "⁻", "", "", "", "")
const BaseOpType_expr = ("ᴴ", "ᴴ", "⁺", "ˣ", "ʸ", "ᶻ", "⁻", "", "", "", "")
const BaseOpType_latex = ("^{\\dagger}", "^{\\dagger}", "^{+}", "^{x}", "^{y}", "^{z}", "^{-}", "", "", "", "")

# BaseOperator struct with optional Lie algebra / transition operator fields
# For non-Lie-algebra operators (bosons, fermions, TLS), algebra_id = 0 and gen_idx = 0
# For Lie algebra generators: algebra_id = registry ID, gen_idx = generator index
# For transition operators |i⟩⟨j|: algebra_id = N (dimension), gen_idx = (i-1)*N + j
@concrete struct BaseOperator
    t::BaseOpType
    name::QuOpName
    inds::QuIndices
    algebra_id::UInt16
    gen_idx::UInt16
    # Constructor for basic operators (backwards compatible)
    function BaseOperator(t::BaseOpType, name, inds...)
        t ∉ (LieAlgebraGen_, Transition_) || throw(ArgumentError("Use specialized constructor"))
        new(t, QuOpName(name), make_indices(inds...), UInt16(0), UInt16(0))
    end
    # Full constructor for Lie algebra generators and transition operators
    function BaseOperator(t::BaseOpType, name, inds::QuIndices, algebra_id::UInt16, gen_idx::UInt16)
        if t ∈ (LieAlgebraGen_, Transition_)
            algebra_id != 0 || throw(ArgumentError("$t requires non-zero algebra_id"))
            gen_idx != 0 || throw(ArgumentError("$t requires non-zero gen_idx"))
        else
            algebra_id == 0 || throw(ArgumentError("Basic operators must have algebra_id = 0"))
            gen_idx == 0 || throw(ArgumentError("Basic operators must have gen_idx = 0"))
        end
        new(t, QuOpName(name), inds, algebra_id, gen_idx)
    end
end

is_lie_algebra_gen(op::BaseOperator) = op.t == LieAlgebraGen_
is_transition_op(op::BaseOperator) = op.t == Transition_

"""
    LieAlgebraGenerator(name, algebra_id, gen_idx, inds...)

Create a Lie algebra generator operator.
- `name`: operator name (symbol)
- `algebra_id`: ID of the registered algebra (from get_or_create_su, etc.)
- `gen_idx`: index of the generator (1 to num_generators)
- `inds...`: optional indices
"""
function LieAlgebraGenerator(name, algebra_id::UInt16, gen_idx::Integer, inds...)
    BaseOperator(LieAlgebraGen_, QuOpName(name), make_indices(inds...), algebra_id, UInt16(gen_idx))
end

# Convenience constructor with integer algebra_id
LieAlgebraGenerator(name, algebra_id::Integer, gen_idx::Integer, inds...) = 
    LieAlgebraGenerator(name, UInt16(algebra_id), gen_idx, inds...)

# Note: su_generator and su_generators are defined in lie_algebra.jl after the algebra registry is available

# ============================================================================
# Transition Operators for N-level systems: |i⟩⟨j|
# ============================================================================
# Encodes N in algebra_id, (i,j) in gen_idx as (i-1)*N + j
# Product rule: |i⟩⟨j| × |k⟩⟨l| = δ_jk |i⟩⟨l|
# Adjoint: (|i⟩⟨j|)† = |j⟩⟨i|

"""
    TransitionOperator(name, N, i, j, inds...)

Create a transition operator |i⟩⟨j| for an N-level system.
- `name`: operator name (symbol)  
- `N`: dimension of the system (number of levels)
- `i`, `j`: level indices (1 to N)
- `inds...`: optional site/mode indices
"""
function TransitionOperator(name, N::Integer, i::Integer, j::Integer, inds...)
    (1 ≤ i ≤ N && 1 ≤ j ≤ N) || throw(ArgumentError("Level indices must be in 1:$N, got i=$i, j=$j"))
    encoded = UInt16((i - 1) * N + j)
    BaseOperator(Transition_, QuOpName(name), make_indices(inds...), UInt16(N), encoded)
end

# Decode (i,j) from transition operator
transition_levels(op::BaseOperator) = (op.t == Transition_) ? 
    (Int((op.gen_idx - 1) ÷ op.algebra_id) + 1, Int((op.gen_idx - 1) % op.algebra_id) + 1) : 
    throw(ArgumentError("Not a transition operator"))
transition_dim(op::BaseOperator) = op.t == Transition_ ? Int(op.algebra_id) : throw(ArgumentError("Not a transition operator"))

"""
    σ(name, i, j, inds...)

Shorthand for 2-level transition operator |i⟩⟨j|. Equivalent to `TransitionOperator(name, 2, i, j, inds...)`.
"""
σ(name, i::Integer, j::Integer, inds...) = TransitionOperator(name, 2, i, j, inds...)

"""
    nlevel_ops(N, name, inds...)

Create all N² transition operators |i⟩⟨j| for an N-level system.
Returns an N×N matrix where entry [i,j] is |i⟩⟨j|.
"""
function nlevel_ops(N::Integer, name, inds...)
    [QuExpr(TransitionOperator(name, N, i, j, inds...)) for i in 1:N, j in 1:N]
end

"""
    BaseOperator

Represents a single quantum operator in an expression.

A `BaseOperator` is one of:
- Bosonic operators: `BosonCreate`, `BosonDestroy`
- Fermionic operators: `FermionCreate`, `FermionDestroy`
- Two-level system (TLS) operators: `TLSCreate`, `TLSDestroy`, `TLSx`, `TLSy`, `TLSz`

Each operator is associated with a name and indices.

# Examples
```julia
a = BosonCreate(:a)        # Bosonic creation operator a†
f = FermionDestroy(:f, 1)  # Fermionic annihilation operator f₁
σx = TLSx(:σ, :i)          # TLS x operator σ_i
```
"""
BaseOperator

for (op,desc) in (
    (:BosonDestroy,"bosonic annihilation"),
    (:BosonCreate,"bosonic creation"),
    (:FermionDestroy,"fermionic annihilation"),
    (:FermionCreate,"fermionic creation"),
    (:TLSDestroy,"TLS annihilation"),
    (:TLSCreate,"TLS creation"),
    (:TLSx,"TLS x"),
    (:TLSy,"TLS y"),
    (:TLSz,"TLS z"))
    @eval begin
        "`$($op)(name,inds)`: represent $($desc) operator ``name_{inds}``"
        $op(name,inds...) = BaseOperator($(Symbol(op,:_)),name,make_indices(inds...))
    end
end

@concrete struct BaseOpProduct
    v::Vector{BaseOperator}
end
Base.isempty(A::BaseOpProduct) = isempty(A.v)
BaseOpProduct() = BaseOpProduct(BaseOperator[])

# a struct representing a delta function with unequal indices
@concrete struct δ
    iA::QuIndex
    iB::QuIndex
    δ(iA,iB) = (@assert !isnoindex(iA) && !isnoindex(iB); new(iA,iB))
end

"""
    δ(iA, iB)

Represents a Kronecker delta function δ_{iA,iB} in a quantum expression.

The delta function is zero if indices are different, and one if they are the same.

# Examples
```julia
δ(QuIndex(1), QuIndex(2))    # δ₁₂
δ(QuIndex(:i), QuIndex(:j))  # δᵢⱼ
```
"""
δ
function δ(Ainds::QuIndices,Binds::QuIndices)
    length(Ainds) == length(Binds) || return nothing

    res = Vector{δ}(undef,length(Ainds))
    jj = 0
    for (iA,iB) in zip(Ainds,Binds)
        if iA != iB
            (isnoindex(iA) || isnoindex(iB)) && return nothing
            # if they are integer indices and different, the result is zero
            isintindex(iA) && isintindex(iB) && return nothing
            res[jj+=1] = iB<iA ? δ(iB,iA) : δ(iA,iB)
        end
    end
    resize!(res,jj)
    res
end

@concrete struct ExpVal
    ops::BaseOpProduct
end

"""
    ExpVal(ops::BaseOpProduct)

Represents an expectation value ⟨ops⟩ in a quantum expression.

Expectation values contain products of quantum operators that should be evaluated as a single unit.

# Examples
```julia
ev = ExpVal(BaseOpProduct([a, a_dag]))  # ⟨a† a⟩
```

# See also
[`Corr`](@ref), [`expval`](@ref)
"""
ExpVal

@concrete struct Corr
    ops::BaseOpProduct
end

"""
    Corr(ops::BaseOpProduct)

Represents a correlator ⟨ops⟩c in a quantum expression.

Correlators (or cumulants) represent connected correlations in the expectation value.

# Examples
```julia
corr = Corr(BaseOpProduct([a, a_dag]))  # ⟨a† a⟩c
```

# See also
[`ExpVal`](@ref), [`corr`](@ref)
"""
Corr

@concrete struct Param
    name::QuOpName
    state::Char
    inds::QuIndices
end

"""
    Param(name, state, inds)

Represents a parameter in a quantum expression.

Parameters are scalar values that can appear in quantum expressions, with optional indices
and conjugation state.

# Arguments
- `name`: Symbol identifying the parameter
- `state`: 'r' for real, 'n' for normal, 'c' for conjugate
- `inds`: Quantum indices (optional)

# Examples
```julia
p = Param(:λ, 'r')        # Real parameter λ
p = Param(:ω, 'r', 1)     # Real parameter ω₁
```
"""
Param

@concrete struct QuTerm
    nsuminds::IndexInt # have a sum over n indices, represented by QuIndex with issumindex(ind)==true
    δs::Vector{δ}
    params::Vector{Param}
    expvals::Vector{ExpVal}
    corrs::Vector{Corr}
    bares::BaseOpProduct
end

"""
    QuTerm

Represents a single term in a quantum algebra expression.

A term consists of:
- Sum indices (summed-over indices)
- Delta functions
- Parameters
- Expectation values
- Correlators/cumulants
- Bare operators

# See also
[`QuExpr`](@ref), [`BaseOperator`](@ref)
"""
QuTerm
QuTerm() = QuTerm(BaseOpProduct())
QuTerm(op::BaseOperator) = QuTerm(BaseOpProduct([op]))
QuTerm(ops::BaseOpProduct) = QuTerm(0,δ[],Param[],ExpVal[],Corr[],ops)
QuTerm(δs::Vector{δ},ops::BaseOpProduct) = QuTerm(0,δs,Param[],ExpVal[],Corr[],ops)
QuTerm(p::Param)  = QuTerm(0,δ[],Param[p],ExpVal[],Corr[],BaseOpProduct())
QuTerm(E::ExpVal) = QuTerm(0,δ[],Param[],ExpVal[E],Corr[],BaseOpProduct())
QuTerm(C::Corr)   = QuTerm(0,δ[],Param[],ExpVal[],Corr[C],BaseOpProduct())
QuTerm(Es::Vector{ExpVal}) = QuTerm(0,δ[],Param[],Es,Corr[],BaseOpProduct())
QuTerm(Cs::Vector{Corr}) = QuTerm(0,δ[],Param[],ExpVal[],Cs,BaseOpProduct())
QuTerm(δs::Vector{δ},Es::Vector{ExpVal}) = QuTerm(0,δs,Param[],Es,Corr[],BaseOpProduct())
QuTerm(δs::Vector{δ},Cs::Vector{Corr})   = QuTerm(0,δs,Param[],ExpVal[],Cs,BaseOpProduct())

Base.isempty(A::QuTerm) = A.nsuminds == 0 && isempty(A.δs) && isempty(A.params) && isempty(A.expvals) && isempty(A.corrs) && isempty(A.bares)

"""
    QuExpr

A quantum algebra expression represented as a sum of terms.

A `QuExpr` is a dictionary mapping `QuTerm`s to numeric coefficients stored as `Number`.
Using `Number` allows heterogeneous numeric types (e.g., `Int`, `Float64`, `Rational`,
`Complex`) to coexist without forced type conversions.

# Constructors

- The quantum operator constructors generated with `@boson_ops`, `@fermion_ops`, etc., return `QuExpr` objects.
- `QuExpr()`: Create an empty expression (zero) - `zero(QuExpr)` works as well.
- `QuExpr(x)`: Create a constant expression from a number `x`. `QuExpr(1)` is equivalent to `one(QuExpr)`.

# See also
[`QuTerm`](@ref), [`normal_form`](@ref)
"""
struct QuExpr
    # A QuantumAlgebra Expression is saved as a Dictionary of QuTerms with scalar prefactors
    terms::Dict{QuTerm,Number}
    QuExpr() = new(Dict{QuTerm,Number}())
    QuExpr(terms::Dict{QuTerm,Number}) = new(terms)
end

function QuExpr(itr)
    A = QuExpr()
    for (t,s) in itr
        _add_with_auto_order!(A,t,simplify_number(s))
    end
    A
end
QuExpr(A::Union{BaseOperator,Param,Corr,ExpVal}) = QuExpr(QuTerm(A))
QuExpr(A::QuTerm) = QuExpr(((A,1),))
QuExpr(s::Number) = QuExpr(((QuTerm(),s),))
Base.isempty(A::QuExpr) = isempty(A.terms)
Base.copy(A::QuExpr) = QuExpr(copy(A.terms))

_add_with_auto_order!(A::QuExpr,B::QuTerm,sB) = using_auto_normal_form() ? _add_with_normal_order!(A,B,sB) : _add_sum_term!(A,B,sB)

function _add_sum_term!(A::QuExpr,oB::QuTerm,sB)
    iszero(sB) && return A
    sold = get(A.terms,oB,zero(sB))
    # function barrier to have concrete types
    _add_sum_term!(A,oB,sB,sold)
end
function _add_sum_term!(A::QuExpr,oB::QuTerm,sB,sold)
    snew = sB + sold
    if iszero(snew)
        delete!(A.terms,oB)
    else
        A.terms[oB] = simplify_number(snew)
    end
    A
end
_map_quexpr_ops(f,A::QuExpr) = QuExpr((f(t),s) for (t,s) in A.terms)

"""
    map_scalar_function(f, A::QuExpr)

Apply a function to all scalar coefficients in a quantum expression.

# Arguments
- `f`: Function to apply to each coefficient
- `A`: Quantum expression

# Returns
A new `QuExpr` with function applied to all coefficients

# Examples
```julia
A = QuExpr(a) + 2 * QuExpr(a_dag)
B = map_scalar_function(x -> 2x, A)  # Double all coefficients
```
"""
map_scalar_function(f,A::QuExpr) = QuExpr((t,f(s)) for (t,s) in A.terms)

"""
    QuExprConstructor

Helper struct for creating operator expressions with automatic conjugate support.

Used internally by macros like `@boson_ops` and `@fermion_ops` to provide
both normal and daggered operator constructors.
"""
struct QuExprConstructor{F,Fdag}
    name::String
    f::F
    namedag::String
    fdag::Fdag
    QuExprConstructor(name,f,namedag,fdag) = new{Core.Typeof(f),Core.Typeof(fdag)}(name,f,namedag,fdag)
    QuExprConstructor(name,f) = QuExprConstructor(name,f,name,f)
end
(q::QuExprConstructor)(args...)::QuExpr = q.f(args...)
Base.adjoint(q::QuExprConstructor) = QuExprConstructor(q.namedag,q.fdag,q.name,q.f)
Base.print(io::IO, q::QuExprConstructor) = print(io, q.name, " (QuExpr constructor)")
Base.show(io::IO, q::QuExprConstructor) = print(io, q)
Base.show(io::IO, ::MIME"text/plain", q::QuExprConstructor) = show(io, q)

wrapdagdeprecated(f) = (args...) -> (Base.depwarn("`xdag()` constructors are deprecated, use `x'()` instead", :xdag; force=true); f(args...))

#################################################################
## "external" functions that always construct QuExpr           ##
#################################################################

"""
    boson_ops(name::Symbol)

Create constructors for bosonic operators with the given name.

Returns a tuple of `(annihilation, creation)` operators that can be called
to create quantum expressions containing these operators.

# Arguments
- `name`: Symbol for the operator name (e.g., `:a`, `:b`)

# Returns
- Tuple `(a, a†)` of `QuExprConstructor` objects

# Examples
```julia
a, a_dag = boson_ops(:a)
expr = a(1) + a_dag(2)  # a₁ + a₂†
```

# See also
[`fermion_ops`](@ref), [`@boson_ops`](@ref)
"""
function boson_ops(name::Symbol)
    op_name = QuOpName(name)
    c = QuExprConstructor(string(name),     (args...) -> QuExpr(BosonDestroy(op_name,args...)),
                          string(name,"†"), (args...) -> QuExpr(BosonCreate( op_name,args...)))
    cdag = QuExprConstructor(c.namedag, wrapdagdeprecated(c.fdag), c.name, wrapdagdeprecated(c.f))
    c, cdag
end

"""
    fermion_ops(name::Symbol)

Create constructors for fermionic operators with the given name.

Returns a tuple of `(annihilation, creation)` operators that can be called
to create quantum expressions containing these operators.

# Arguments
- `name`: Symbol for the operator name (e.g., `:f`, `:c`)

# Returns
- Tuple `(f, f†)` of `QuExprConstructor` objects

# Examples
```julia
f, f_dag = fermion_ops(:f)
expr = f(1) + f_dag(2)  # f₁ + f₂†
```

# See also
[`boson_ops`](@ref), [`@fermion_ops`](@ref)
"""
function fermion_ops(name::Symbol)
    op_name = QuOpName(name)
    c = QuExprConstructor(string(name),     (args...) -> QuExpr(FermionDestroy(op_name,args...)),
                          string(name,"†"), (args...) -> QuExpr(FermionCreate( op_name,args...)))
    cdag = QuExprConstructor(c.namedag, wrapdagdeprecated(c.fdag), c.name, wrapdagdeprecated(c.f))
    c, cdag
end

"""
    tlspm_ops(name::Symbol)

Return constructors for TLS ladder operators `(σ₋, σ₊)` for the given `name`.

The concrete output depends on `use_σpm`: when true, returns annihilation/creation
expressions; when false, returns linear combinations of `σx` and `σy`.
"""
function tlspm_ops(name::Symbol)
    op_name = QuOpName(name)
    namem = string(name,"m")
    namep = string(name,"p")
    fm = (args...) -> using_σpm() ? QuExpr(TLSDestroy(op_name,args...)) : QuExpr((QuTerm(TLSx(op_name,args...))=>1//2, QuTerm(TLSy(op_name,args...))=>-1im//2))
    fp = (args...) -> using_σpm() ? QuExpr(TLSCreate( op_name,args...)) : QuExpr((QuTerm(TLSx(op_name,args...))=>1//2, QuTerm(TLSy(op_name,args...))=>1im//2))
    (QuExprConstructor(namem, fm, namep, fp), QuExprConstructor(namep, fp, namem, fm))
end

"""
    tlsxyz_ops(name::Symbol)

Return constructors for Pauli operators `(σₓ, σ_y, σ_z)` for the given `name`.

The concrete output depends on `use_σpm`: when true, returns combinations of ladder
operators; when false, returns the direct `σx/σy/σz` operators.
"""
function tlsxyz_ops(name::Symbol)
    op_name = QuOpName(name)
    namex = string(name,"x")
    namey = string(name,"y")
    namez = string(name,"z")
    tlsx = QuExprConstructor(namex, (args...) -> using_σpm() ? QuExpr((QuTerm(TLSDestroy(op_name,args...))=>1, QuTerm(TLSCreate(op_name,args...))=>1)) : QuExpr(TLSx(op_name,args...)))
    tlsy = QuExprConstructor(namey, (args...) -> using_σpm() ? QuExpr((QuTerm(TLSDestroy(op_name,args...))=>1im, QuTerm(TLSCreate(op_name,args...))=>-1im)) : QuExpr(TLSy(op_name,args...)))
    tlsz = QuExprConstructor(namez, (args...) -> using_σpm() ? QuExpr((QuTerm(BaseOpProduct([TLSCreate(op_name,args...),TLSDestroy(op_name,args...)]))=>2, QuTerm()=>-1)) : QuExpr(TLSz(op_name,args...)))
    tlsx, tlsy, tlsz
end

"`@boson_ops name`: define function `\$name` for creating bosonic annihilation operators with name `name` (also defines deprecated `\$(name)dag`, use `\$(name)'` instead) "
macro boson_ops(name)
    :( ($(esc(name)), $(esc(Symbol(name,:dag)))) = boson_ops($(Meta.quot(name))) )
end

"`@fermion_ops name`: define function `\$name` for creating fermionic annihilation operators with name `name` (also defines deprecated `\$(name)dag`, use `\$(name)'` instead)"
macro fermion_ops(name)
    :( ($(esc(name)), $(esc(Symbol(name,:dag)))) = fermion_ops($(Meta.quot(name))) )
end

"`@anticommuting_fermion_group name1 name2 ...`: define a group of mutually anticommuting fermionic operators"
macro anticommuting_fermion_group(names...)
    # start the groupname (which is the "internal" species name) by concatenating all names,
    # to ensure reasonable sorting relative to other fermionic species
    code = quote
        groupname = Symbol($names...,gensym())
        add_groupaliases(groupname,$names)
        ann, cre = fermion_ops(groupname)
    end
    for (ii,name) in enumerate(names)
        push!(code.args,:( $(esc(name)) = QuExprConstructor($(string(name)), (args...) -> ann($ii,args...),
                                                             $(string(name,"†")), (args...) -> ann'($ii,args... )) ))
        push!(code.args,:( $(esc(Symbol(name,:dag))) = ($(esc(name)))' ))
    end
    push!(code.args, :( nothing ))
    code
end

# functions for constructing Pauli operators depending on whether we use (σ+,σ-) or (σx,σy,σz) as the "basic" operators

"`@tlspm_ops name`: define functions `\$(name)m` and `\$(name)p` creating jump operators for a two-level system with name `name`."
macro tlspm_ops(name)
    :( ($(esc(Symbol(name,:m))), $(esc(Symbol(name,:p)))) = tlspm_ops($(Meta.quot(name))) )
end

"`@tlsxyz_ops name`: define functions `\$(name)x`, `\$(name)y`, and `\$(name)z` creating Pauli operators for a two-level system with name `name`."
macro tlsxyz_ops(name)
    :( ($(esc(Symbol(name,:x))), $(esc(Symbol(name,:y))), $(esc(Symbol(name,:z)))) = tlsxyz_ops($(Meta.quot(name))) )
end

## functions for constructing `param`s with string macros,
# Pc"ω_i,j" = param(:ω,'n',:i,:j) (complex parameter)
# Pr"ω_i,j" = param(:ω,'r',:i,:j) (real parameter)

"""
    param(name, [state], inds...)

Construct a parameter as a `QuExpr`. `state` is one of `n` (normal/complex),
`r` (real), or `c` (conjugate). Indices are optional.
"""
param(name::Symbol,args...) = param(QuOpName(name),args...)
param(name::QuOpName,args...) = param(name,'n',args...)
function param(name::QuOpName,state::Char,inds...)
    state ∈ ('r','n','c') || throw(ArgumentError("state has to be one of n,r,c"))
    QuExpr(Param(name,state,make_indices(inds...)))
end

function parse_paramstr(s)
    s = split(s, "_"; limit=2)
    par = Symbol(s[1])
    if length(s) == 1
        inds = ()
    else
        indstr = s[2]
        # allow {} around the index expression
        if startswith(indstr,"{") && endswith(indstr,"}")
            indstr = chop(indstr,head=1,tail=1)
        end
        inds = Meta.parse.(split(indstr,","))
    end
    par, inds
end

macro Pc_str(s)
    par, inds = parse_paramstr(s)
    param(par,'n',inds)
end
macro Pr_str(s)
    par, inds = parse_paramstr(s)
    param(par,'r',inds)
end

"""
    expval(A)

Wrap bare operators inside `A` into a formal expectation value ⟨A⟩.
Accepts `QuTerm` or `QuExpr`; numeric inputs are returned as constant expressions.
"""
function expval(A::QuTerm)
    if isempty(A.bares)
        A
    else
        QuTerm(A.nsuminds,A.δs,A.params,[A.expvals; ExpVal(A.bares)],A.corrs,BaseOpProduct())
    end
end
"""
    corr(A)

Wrap bare operators inside `A` into a formal connected correlator ⟨A⟩₍c₎.
Accepts `QuTerm` or `QuExpr`; numeric inputs are returned as constant expressions.
"""
function corr(A::QuTerm)
    if isempty(A.bares)
        A
    else
        QuTerm(A.nsuminds,A.δs,A.params,A.expvals,[A.corrs; Corr(A.bares)],BaseOpProduct())
    end
end

expval(A::QuExpr) = _map_quexpr_ops(expval,A)
corr(A::QuExpr) = _map_quexpr_ops(corr,A)

expval(A::Number) = QuExpr(A)
corr(A::Number) = QuExpr(A)

"`∑(ind,A::QuExpr)`: return (formal) sum of expression A over index ind."
function ∑(ind::QuIndex,A::QuTerm)
    (issumindex(ind) || isintindex(ind)) && error("Index $ind to be summed over needs to be symbolic!")
    sumind = sumindex(A.nsuminds+one(A.nsuminds))
    f = replace_inds(ind=>sumind)
    g = reorder_suminds()
    # use the form that also sets nsuminds
    g(f(A,sumind.num))
end
∑(ind::QuIndex,A::QuExpr) = _map_quexpr_ops(t->∑(ind,t),A)
∑(ind::Symbol,A::QuExpr) = ∑(QuIndex(ind),A)
# foldr calculates ∑(i1,∑(i2,∑(i3,A))), i.e., the sum over all indices in inds
∑(inds::T,A::QuExpr) where T<:NTuple{N,Union{Symbol,QuIndex}} where N = foldr(∑,inds,init=A)
∑(ind,A::Number) = ∑(ind,QuExpr(A))

const EqDict{T<:Union{ExpVal,Corr}} = OrderedDict{T,QuExpr}

struct QuEqSys{T<:Union{ExpVal,Corr}}
    eqs::EqDict{T}
end

const QuantumObject = Union{QuIndex,QuOpName,BaseOperator,Param,BaseOpProduct,ExpVal,Corr,QuTerm,QuExpr,QuEqSys}

@static if _DEFINE_DEFAULT_OPS
    # do not use the macros here so that we can define the constructors as const
    const a, adag = boson_ops(:a)
    const f, fdag = fermion_ops(:f)
    const σx, σy, σz = tlsxyz_ops(:σ)
    const σm, σp = tlspm_ops(:σ)

    export σx, σy, σz, σp, σm
    export a, adag, f, fdag
end

# ============================================================================
# SU(N) generator convenience functions
# ============================================================================

"""
    su_generator(N::Int, name::Symbol, gen_idx::Int, inds...)

Create a single SU(N) generator operator as a QuExpr.

# Arguments
- `N`: dimension of SU(N) (e.g., 2 for SU(2), 3 for SU(3))
- `name`: operator name (symbol)
- `gen_idx`: generator index (1 to N²-1)
- `inds...`: optional indices

# Example
```julia
# Create the first SU(3) generator named λ
T1 = su_generator(3, :λ, 1)

# Create with an index
T1_i = su_generator(3, :λ, 1, :i)
```
"""
function su_generator(N::Int, name::Symbol, gen_idx::Int, inds...)
    ngen = N^2 - 1
    1 <= gen_idx <= ngen || throw(ArgumentError("Generator index must be 1 to $ngen for SU($N), got $gen_idx"))
    algebra_id = get_or_create_su(N)
    QuExpr(LieAlgebraGenerator(name, algebra_id, gen_idx, inds...))
end

"""
    su_generators(N::Int, name::Symbol, inds...)

Create all SU(N) generators as a tuple of QuExpr.

# Arguments
- `N`: dimension of SU(N)
- `name`: base name for operators
- `inds...`: optional indices (applied to all generators)

# Returns
A tuple of N²-1 QuExpr, one for each generator.

# Example
```julia
# Create all 8 SU(3) generators
λ = su_generators(3, :λ)
λ[1] * λ[2]  # Product of first two generators

# With indices
λ_i = su_generators(3, :λ, :i)
```
"""
function su_generators(N::Int, name::Symbol, inds...)
    ngen = N^2 - 1
    algebra_id = get_or_create_su(N)
    ntuple(k -> QuExpr(LieAlgebraGenerator(name, algebra_id, k, inds...)), ngen)
end

# ============================================================================
# SO(N) and Sp(2N) generator constructors
# ============================================================================

"""
    so_generator(N::Int, name::Symbol, gen_idx::Int, inds...)

Create a single SO(N) generator operator.

# Arguments
- `N`: dimension of SO(N) (N >= 3)
- `name`: operator name
- `gen_idx`: generator index (1 to N(N-1)/2)
- `inds...`: optional indices
"""
function so_generator(N::Int, name::Symbol, gen_idx::Int, inds...)
    ngen = div(N * (N - 1), 2)
    1 <= gen_idx <= ngen || throw(ArgumentError("Generator index must be 1 to $ngen for SO($N), got $gen_idx"))
    algebra_id = get_or_create_so(N)
    QuExpr(LieAlgebraGenerator(name, algebra_id, gen_idx, inds...))
end

"""
    so_generators(N::Int, name::Symbol, inds...)

Create all SO(N) generators as a tuple of QuExpr.
Returns N(N-1)/2 generators.
"""
function so_generators(N::Int, name::Symbol, inds...)
    ngen = div(N * (N - 1), 2)
    algebra_id = get_or_create_so(N)
    ntuple(k -> QuExpr(LieAlgebraGenerator(name, algebra_id, k, inds...)), ngen)
end

"""
    sp_generator(N::Int, name::Symbol, gen_idx::Int, inds...)

Create a single Sp(2N) generator operator.

# Arguments
- `N`: rank of Sp(2N) (N >= 1)
- `name`: operator name
- `gen_idx`: generator index (1 to N(2N+1))
- `inds...`: optional indices
"""
function sp_generator(N::Int, name::Symbol, gen_idx::Int, inds...)
    ngen = N * (2N + 1)
    1 <= gen_idx <= ngen || throw(ArgumentError("Generator index must be 1 to $ngen for Sp($(2N)), got $gen_idx"))
    algebra_id = get_or_create_sp(N)
    QuExpr(LieAlgebraGenerator(name, algebra_id, gen_idx, inds...))
end

"""
    sp_generators(N::Int, name::Symbol, inds...)

Create all Sp(2N) generators as a tuple of QuExpr.
Returns N(2N+1) generators.
"""
function sp_generators(N::Int, name::Symbol, inds...)
    ngen = N * (2N + 1)
    algebra_id = get_or_create_sp(N)
    ntuple(k -> QuExpr(LieAlgebraGenerator(name, algebra_id, k, inds...)), ngen)
end

# ============================================================================
# SU(2) Ladder Operators
# ============================================================================
#
# For SU(2), the ladder operators are defined as:
#   T⁺ = T¹ + iT²  (raising operator)
#   T⁻ = T¹ - iT²  (lowering operator)
#
# These are analogous to σ⁺ = (σˣ + iσʸ)/2 and σ⁻ = (σˣ - iσʸ)/2 for TLS.
# Note: T⁺ = σ⁺ and T⁻ = σ⁻ (not scaled by 2!)

export su2_raising, su2_lowering

"""
    su2_raising(name::Symbol=:T, inds...)

Create the SU(2) raising operator T⁺ = T¹ + iT².

This operator raises the eigenvalue of T³ by 1 (within the representation).
It is analogous to σ⁺ for TLS operators.

# Example
```julia
Tp = su2_raising(:T)
Tm = su2_lowering(:T)
comm(T³, Tp)  # => Tp (raising operator has eigenvalue +1 under T³)
comm(Tp, Tm)  # => 2T³
```
"""
function su2_raising(name::Symbol=:T, inds...)
    T = su_generators(2, name, inds...)
    T[1] + 1im * T[2]
end

"""
    su2_lowering(name::Symbol=:T, inds...)

Create the SU(2) lowering operator T⁻ = T¹ - iT².

This operator lowers the eigenvalue of T³ by 1 (within the representation).
It is analogous to σ⁻ for TLS operators.

# Example
```julia
Tp = su2_raising(:T)
Tm = su2_lowering(:T)
comm(T³, Tm)  # => -Tm (lowering operator has eigenvalue -1 under T³)
Tm * |0⟩ = 0  # annihilates the lowest weight state
```
"""
function su2_lowering(name::Symbol=:T, inds...)
    T = su_generators(2, name, inds...)
    T[1] - 1im * T[2]
end

"""
    su2_ladder_operators(name::Symbol=:T, inds...)

Return a tuple (T⁺, T⁻, T³) of SU(2) ladder operators and the diagonal generator.

This is convenient for working in the ladder operator basis.

# Example
```julia
Tp, Tm, Tz = su2_ladder_operators(:T)
# Commutation relations:
comm(Tz, Tp)  # => Tp
comm(Tz, Tm)  # => -Tm
comm(Tp, Tm)  # => 2Tz
```
"""
function su2_ladder_operators(name::Symbol=:T, inds...)
    T = su_generators(2, name, inds...)
    Tp = T[1] + 1im * T[2]
    Tm = T[1] - 1im * T[2]
    Tz = T[3]
    (Tp, Tm, Tz)
end

export su2_ladder_operators
