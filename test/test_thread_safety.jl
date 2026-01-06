# Thread Safety Analysis for QuantumAlgebra.jl
# 
# This file tests for potential thread safety issues in the package.
# Run with: julia --threads=auto test/test_thread_safety.jl

using QuantumAlgebra
using Test
using Base.Threads

# Get thread count
println("Number of threads: $(nthreads())")

if nthreads() < 2
    @warn "Running with only 1 thread. For proper thread safety testing, run with: julia --threads=auto"
end

# =============================================================================
# Test 1: Operator Name Table Race Condition
# =============================================================================
# The _NameTable and _NameTableInv are global mutable structures accessed
# during operator creation via QuOpName constructor.

@testset "Thread Safety Analysis" begin
    @testset "Operator Name Table Race" begin
        # This tests whether creating operators with new names from multiple threads
        # can cause a race condition in the name table
        
        errors = Atomic{Int}(0)
        n_iterations = 1000
        n_names_per_thread = 50
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            # Each thread creates operators with unique names
                            name = Symbol("op_thread$(t)_iter$(i)")
                            b, _ = boson_ops(name)
                            # Do some work with the operator
                            expr = b() * b'()
                            nf = normal_form(expr)
                            # Verify result
                            if length(nf.terms) != 2
                                atomic_add!(errors, 1)
                            end
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  Operator name table test: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
    
    # =============================================================================
    # Test 2: SU(N) Algebra Registry Race Condition
    # =============================================================================
    # The ALGEBRA_REGISTRY is a global mutable structure accessed via get_or_create_su
    
    @testset "Algebra Registry Race" begin
        errors = Atomic{Int}(0)
        n_iterations = 500
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            # Multiple threads trying to access/create SU(N) algebras
                            N = 2 + (i % 5)  # N = 2, 3, 4, 5, 6
                            T = su_generators(N, Symbol("T_t$(t)_$(i)"))
                            
                            # Do some computation
                            if N == 2
                                result = normal_form(T[1] * T[2])
                                # Should be (i/2) * T[3] for SU(2)
                            end
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in algebra registry test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  Algebra registry test: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
    
    # =============================================================================
    # Test 3: Correlation Cache Race Condition
    # =============================================================================
    # The _EXPVAL2CORRS_CACHE and _CORR2EXPVALS_CACHE use get! which is not thread-safe
    
    @testset "Correlation Cache Race" begin
        errors = Atomic{Int}(0)
        n_iterations = 200
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            # Create expressions of varying lengths to trigger cache population
                            n_ops = 1 + (i % 5)  # 1 to 5 operators
                            ops = [a(k) for k in 1:n_ops]
                            expr = prod(ops)
                            
                            # This calls expval2corrs_inds which uses get! on a shared Dict
                            result = expval_as_corrs(expr)
                            
                            # Verify the result has expected structure
                            if isempty(result.terms)
                                atomic_add!(errors, 1)
                            end
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in cache test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  Correlation cache test: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
    
    # =============================================================================
    # Test 4: CORRHEISDICT Cache Race Condition
    # =============================================================================
    # The _CORRHEISDICT uses get! which is not thread-safe
    
    @testset "CorrHeis Cache Race" begin
        errors = Atomic{Int}(0)
        n_iterations = 100
        
        H = Pr"ω" * a'() * a()
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            # Different threads computing same/similar equations
                            op = a(t)  # Different indices per thread
                            # This triggers _corrheis_cached which uses get!
                            result = QuantumAlgebra.corrheis(op, H, ())
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in corrheis cache test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  CorrHeis cache test: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
    
    # =============================================================================
    # Test 5: Group Aliases Race Condition
    # =============================================================================
    # The _GROUPALIASES Dict is accessed from @anticommuting_fermion_group
    
    @testset "Group Aliases Race" begin
        errors = Atomic{Int}(0)
        n_iterations = 200
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            # Create anticommuting fermion groups
                            groupname = Symbol("group_t$(t)_$(i)")
                            names = (Symbol("c_$(t)_$(i)"), Symbol("d_$(t)_$(i)"))
                            QuantumAlgebra.add_groupaliases(groupname, names)
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in group aliases test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  Group aliases test: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
    
    # =============================================================================
    # Test 6: Concurrent normal_form on independent expressions (should be safe)
    # =============================================================================
    
    @testset "Concurrent normal_form (independent expressions)" begin
        errors = Atomic{Int}(0)
        n_iterations = 500
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            # Each thread works with its own expression
                            expr = a(t, i) * a'(t, i) * σx(t) * a(t, i)
                            result = normal_form(expr)
                            
                            # Basic sanity check
                            if isempty(result.terms)
                                atomic_add!(errors, 1)
                            end
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in normal_form test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  Concurrent normal_form test: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
    
    # =============================================================================
    # Test 7: Shared expression mutation (known unsafe pattern)
    # =============================================================================
    
    @testset "Shared expression modification (unsafe pattern)" begin
        # This test demonstrates that modifying shared expressions is NOT safe
        # It's here to document the known limitation
        
        shared_expr = a() * a'()
        errors = Atomic{Int}(0)
        results = Vector{QuExpr}(undef, nthreads())
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        # All threads work on the same expression
                        # This should still be safe for READ operations
                        results[t] = normal_form(shared_expr)
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        # All results should be the same
        expected = normal_form(shared_expr)
        all_same = all(r -> r == expected, results)
        
        @test errors[] == 0
        @test all_same
        println("  Shared expression read test: $(errors[] == 0 && all_same ? "PASSED" : "FAILED")")
    end
end

println("\n" * "="^60)
println("Thread Safety Analysis Complete")
println("="^60)

# =============================================================================
# Additional Stress Tests: Concurrent Read + Write
# =============================================================================
# These tests specifically target the vulnerabilities identified in code review:
# - Issue #1: sym() reading while QuOpName writing
# - Issue #2: get_algebra() reading while register_algebra! writing
# - Issue #3: unalias() reading while add_groupaliases writing
# - Issue #5: Lock contention in get_or_create_su

@testset "Concurrent Read+Write Stress Tests" begin
    
    # =========================================================================
    # Test: Concurrent operator creation AND sym() access (Issue #1)
    # =========================================================================
    @testset "Concurrent create + sym() read" begin
        errors = Atomic{Int}(0)
        n_iterations = 2000
        
        # Pre-create some operators to read from
        precreated_ops = [boson_ops(Symbol("precreated_$i"))[1] for i in 1:10]
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            if rand() < 0.3
                                # 30% of the time: CREATE new operator (triggers push! to _NameTableInv)
                                name = Symbol("stress_t$(t)_i$(i)")
                                b, _ = boson_ops(name)
                                # Force use of the operator (which calls sym() internally for printing)
                                str = sprint(show, b())
                            else
                                # 70% of the time: READ existing operator (triggers _NameTableInv[i] via sym())
                                op = precreated_ops[rand(1:10)]
                                str = sprint(show, op())
                                # Also test through normal_form which uses sym() in various places
                                nf = normal_form(op() * op'())
                            end
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in create+sym test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  Concurrent create + sym() read: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
    
    # =========================================================================
    # Test: Concurrent algebra creation AND get_algebra access (Issue #2)
    # =========================================================================
    @testset "Concurrent algebra create + get_algebra read" begin
        errors = Atomic{Int}(0)
        n_iterations = 500
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            if rand() < 0.2
                                # 20%: Create new SU(N) algebra (expensive, triggers push!)
                                N = 2 + (i % 8)  # N = 2 to 9
                                T = su_generators(N, Symbol("Tnew_t$(t)_i$(i)"))
                            else
                                # 80%: Use existing algebra (triggers get_algebra reads)
                                T = su_generators(2, Symbol("Texist_t$(t)_i$(i)"))
                                # Force computation using the algebra
                                result = normal_form(T[1] * T[2] - T[2] * T[1])
                            end
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in algebra create+read test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  Concurrent algebra create + get_algebra: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
    
    # =========================================================================
    # Test: Concurrent alias creation AND unalias access (Issue #3)
    # =========================================================================
    @testset "Concurrent alias create + unalias read" begin
        errors = Atomic{Int}(0)
        n_iterations = 1000
        
        # Pre-create some fermion groups to read via unalias
        for i in 1:5
            groupname = Symbol("pregroup_$i")
            names = (Symbol("pref1_$i"), Symbol("pref2_$i"))
            QuantumAlgebra.add_groupaliases(groupname, names)
        end
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            if rand() < 0.3
                                # 30%: Create new group alias (triggers Dict mutation)
                                groupname = Symbol("dyngroup_t$(t)_i$(i)")
                                names = (Symbol("dynf1_t$(t)_i$(i)"), Symbol("dynf2_t$(t)_i$(i)"))
                                QuantumAlgebra.add_groupaliases(groupname, names)
                            else
                                # 70%: Use fermion operators which may trigger unalias reads
                                # Even if the operator doesn't have an alias, it still checks the Dict
                                c, _ = fermion_ops(Symbol("ferm_t$(t)_i$(i)"))
                                expr = c() * c'()
                                nf = normal_form(expr)
                            end
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in alias create+unalias test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  Concurrent alias create + unalias: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
    
    # =========================================================================
    # Test: Heavy contention on SU(N) creation (Issue #5 - lock contention)
    # =========================================================================
    @testset "SU(N) creation contention (double-checked locking)" begin
        errors = Atomic{Int}(0)
        timings = Vector{Float64}(undef, nthreads())
        n_iterations = 200
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        t_start = time()
                        for i in 1:n_iterations
                            # All threads request same large SU(N) simultaneously
                            # With double-checked locking, only one thread should compute
                            # Others should get cached result quickly
                            N = 7  # SU(7) has 48 generators - expensive to compute
                            T = su_generators(N, Symbol("contention_t$(t)_i$(i)"))
                            
                            # Verify we got valid generators
                            if length(T) != N^2 - 1
                                atomic_add!(errors, 1)
                            end
                        end
                        timings[t] = time() - t_start
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in contention test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        # With proper double-checked locking, threads shouldn't be blocked too long
        max_time = maximum(timings)
        min_time = minimum(timings)
        println("  SU(N) contention test: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
        println("    Thread timings: min=$(round(min_time, digits=3))s, max=$(round(max_time, digits=3))s")
    end
    
    # =========================================================================
    # Test: Mixed workload stress test (all operations interleaved)
    # =========================================================================
    @testset "Mixed workload stress test" begin
        errors = Atomic{Int}(0)
        n_iterations = 1000
        
        @sync begin
            for t in 1:nthreads()
                Threads.@spawn begin
                    try
                        for i in 1:n_iterations
                            op_type = rand(1:5)
                            
                            if op_type == 1
                                # Create and use boson operator
                                b, _ = boson_ops(Symbol("mixed_b_t$(t)_i$(i)"))
                                nf = normal_form(b() * b'())
                            elseif op_type == 2
                                # Create and use fermion operator
                                c, _ = fermion_ops(Symbol("mixed_c_t$(t)_i$(i)"))
                                nf = normal_form(c() * c'())
                            elseif op_type == 3
                                # Use SU(N) generators
                                N = 2 + (i % 4)
                                T = su_generators(N, Symbol("mixed_T_t$(t)_i$(i)"))
                                nf = normal_form(T[1] * T[1])
                            elseif op_type == 4
                                # Use correlation functions
                                expr = a(t) * a'(t)
                                corrs = expval_as_corrs(expr)
                            else
                                # Print operators (triggers sym())
                                str = sprint(show, a(t, i) * a'(t, i))
                            end
                        end
                    catch e
                        atomic_add!(errors, 1)
                        @error "Thread $t error in mixed workload test" exception=(e, catch_backtrace())
                    end
                end
            end
        end
        
        @test errors[] == 0
        println("  Mixed workload stress test: $(errors[] == 0 ? "PASSED" : "FAILED ($(errors[]) errors)")")
    end
end

println("\n" * "="^60)
println("All Stress Tests Complete")
println("="^60)
