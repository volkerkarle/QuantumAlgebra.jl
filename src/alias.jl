const _GROUPALIASES = Dict{QuOpName,Vector{QuOpName}}()
const _GROUPALIASES_LOCK = ReentrantLock()

add_groupaliases(groupname,names) = lock(_GROUPALIASES_LOCK) do
    _GROUPALIASES[QuOpName(groupname)] = QuOpName.(collect(names))
end

function unalias(A::BaseOperator)::BaseOperator
    lock(_GROUPALIASES_LOCK) do
        if A.name ∈ keys(_GROUPALIASES)
            names = _GROUPALIASES[A.name]
            ind = first(A.inds)
            @assert isintindex(ind)
            BaseOperator(A.t,names[ind.num],Base.tail(A.inds))
        else
            A
        end
    end
end

unalias(A::T) where T<:Union{ExpVal,Corr} = T(BaseOpProduct(unalias.(A.ops.v)))
