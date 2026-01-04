# helper function used in some definitons, copied from ColorTypes.jl
#Base.@pure basetype(T::Type) = Base.typename(T).wrapper
#Base.@pure basetype(A) = basetype(typeof(A))

macro concrete(expr)
    @assert expr.head == :struct
    S = expr.args[2]
    return quote
        $(esc(expr))

        for n in fieldnames($S)
            if !isconcretetype(fieldtype($S, n))
                error("field $n is not concrete")
            end
        end
    end
end

simplify_number(x::Number) = x
simplify_number(x::AbstractFloat) = begin
    # First try to convert to int
    if isinteger(x) && typemin(Int) <= x <= typemax(Int)
        return Int(x)
    end
    # Try to convert to simple rational, but only for "clean" fractions
    # that are commonly encountered (powers of 2 denominators, and small multiples)
    # Avoid converting arbitrary decimals like 9.4 to 47//5
    for denom in (2, 4, 8)
        numer = x * denom
        if isinteger(numer) && abs(numer) <= typemax(Int)
            return Rational{Int}(Int(numer), denom)
        end
    end
    x
end
simplify_number(x::Rational) = isone(denominator(x)) ? numerator(x) : x
simplify_number(x::Complex) = begin
    r, i = reim(x)
    iszero(i) ? simplify_number(r) : complex(simplify_number(r),simplify_number(i))
end
