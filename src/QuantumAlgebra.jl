module QuantumAlgebra

using PackageExtensionCompat

include("tools.jl")
include("lie_algebra.jl")  # Lie algebra infrastructure (before operator_defs)
include("operator_defs.jl")
include("index_handling.jl")
include("operator_baseops.jl")
include("alias.jl")
include("output.jl")
include("correlations.jl")
include("vacuum_expvals.jl")
include("convert_to_expression.jl")
include("eqsofmotion.jl")
include("eqsys.jl")

include("precompile.jl")

function __init__()
    @require_extensions

    # Initialize Lie algebra registry (SU(2) as ID 1)
    __init_algebra_registry__()

    auto_normal = @load_preference("auto_normal_form", false)
    if haskey(ENV,"QUANTUMALGEBRA_AUTO_NORMAL_FORM")
        val = parse(Bool,ENV["QUANTUMALGEBRA_AUTO_NORMAL_FORM"])
        @warn "Environment variable QUANTUMALGEBRA_AUTO_NORMAL_FORM is set to $val. This is deprecated, please instead set it through Preferences.jl by calling `QuantumAlgebra.auto_normal_form($val; set_preference=true) or `Preferences.set_preferences!(QuantumAlgebra,\"auto_normal_form\"=>$val)."
        if !@has_preference("auto_normal_form")
            auto_normal = val
        end
    end
    auto_normal_form(auto_normal)

    use_σpm(@load_preference("use_σpm", false))
    
    # Float coefficients default to false (symbolic/exact mode)
    # Set to true for high-speed Float64 computations
    use_float_coefficients(@load_preference("use_float_coefficients", false))
end

end # module
