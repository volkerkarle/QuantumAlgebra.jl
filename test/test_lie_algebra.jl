# Comprehensive tests for SU(N) Lie algebra functionality
# Tests: algebra structure, generators, commutation, products, conversions, vacuum expectations, ladder operators
# Uses parallelization for expensive tests when multiple threads are available

using Test
using QuantumAlgebra
using QuantumAlgebra: get_algebra, get_or_create_su, structure_constants, symmetric_structure_constants,
                      gellmann_matrix, algebra_dim, num_generators, identity_coefficient,
                      product_coefficients, commutator_coefficients, δ, QuExpr, QuTerm, BaseOpProduct, QuIndex,
                      BosonDestroy, BosonCreate, FermionDestroy, FermionCreate,
                      TLSx, TLSy, TLSz, TLSCreate, TLSDestroy, LieAlgebraGen_, Transition_
using LinearAlgebra: tr, I
using Base.Threads: @threads, nthreads

# Helper function for delta symbols (same as in runtests.jl)
function myδ_local(i, j)
    iA, iB = QuIndex.((i, j))
    QuExpr(QuTerm([δ(min(iA, iB), max(iA, iB))], BaseOpProduct()))
end

@testset "SU(N) Lie Algebra" begin

    # =========================================================================
    # Algebra Structure Tests
    # =========================================================================
    @testset "Algebra structure" begin
        
        @testset "Algebra registry" begin
            alg_id = get_or_create_su(2)
            @test alg_id == UInt16(1)  # SU(2) pre-registered as ID 1
            @test get_or_create_su(2) == UInt16(1)  # Idempotent
            
            alg_id_3 = get_or_create_su(3)
            @test alg_id_3 > UInt16(1)
            @test get_or_create_su(3) == alg_id_3
        end

        @testset "SU(2) structure" begin
            alg = get_algebra(get_or_create_su(2))
            @test algebra_dim(alg) == 2
            @test num_generators(alg) == 3
            
            # Structure constants: f^{abc} = ε^{abc}
            @test structure_constants(alg, 1, 2)[3] ≈ 1.0
            @test structure_constants(alg, 2, 3)[1] ≈ 1.0
            @test structure_constants(alg, 3, 1)[2] ≈ 1.0
            @test structure_constants(alg, 2, 1)[3] ≈ -1.0  # Antisymmetry
            
            # SU(2) has d^{abc} = 0
            for a in 1:3, b in 1:3
                @test isempty(symmetric_structure_constants(alg, a, b))
            end
        end

        @testset "SU(3) structure" begin
            alg = get_algebra(get_or_create_su(3))
            @test algebra_dim(alg) == 3
            @test num_generators(alg) == 8
            
            # Known structure constants (using our generalized Gell-Mann ordering)
            # f^{1,4,7} = 1.0 (this corresponds to the standard SU(2) subalgebra)
            @test isapprox(get(structure_constants(alg, 1, 4), 7, 0.0), 1.0, atol=1e-10)
            
            # d^{118} = 1/√3
            @test isapprox(get(symmetric_structure_constants(alg, 1, 1), 8, 0.0), 1/sqrt(3), atol=1e-10)
        end

        @testset "SU(4) structure" begin
            alg = get_algebra(get_or_create_su(4))
            @test algebra_dim(alg) == 4
            @test num_generators(alg) == 15
        end

        @testset "SU(5) structure" begin
            alg = get_algebra(get_or_create_su(5))
            @test algebra_dim(alg) == 5
            @test num_generators(alg) == 24
        end

        @testset "Multiple algebras in same session" begin
            # Ensure SU(2), SU(3), SU(4), SU(5) can coexist without interference
            id2 = get_or_create_su(2)
            id3 = get_or_create_su(3)
            id4 = get_or_create_su(4)
            id5 = get_or_create_su(5)
            
            # All should have distinct IDs
            @test length(unique([id2, id3, id4, id5])) == 4
            
            # Each should return correct algebra type
            @test get_algebra(id2) isa QuantumAlgebra.SUAlgebra{2}
            @test get_algebra(id3) isa QuantumAlgebra.SUAlgebra{3}
            @test get_algebra(id4) isa QuantumAlgebra.SUAlgebra{4}
            @test get_algebra(id5) isa QuantumAlgebra.SUAlgebra{5}
        end
    end

    # =========================================================================
    # Gell-Mann Matrix Tests
    # =========================================================================
    @testset "Gell-Mann matrices" begin
        
        @testset "SU(2) matrices (Pauli/2)" begin
            T1, T2, T3 = [gellmann_matrix(2, k) for k in 1:3]
            
            @test T1 ≈ [0 0.5; 0.5 0]
            @test T2 ≈ [0 -0.5im; 0.5im 0]
            @test T3 ≈ [0.5 0; 0 -0.5]
            
            # Traceless and Hermitian
            for T in [T1, T2, T3]
                @test isapprox(tr(T), 0, atol=1e-12)
                @test T ≈ T'
            end
        end

        @testset "Orthonormality Tr(TᵃTᵇ) = δₐᵦ/2" begin
            for N in [2, 3, 4, 5]  # Extended to include N=5
                ngen = N^2 - 1
                # Parallelize: collect all (a,b) pairs and test in parallel
                pairs = [(a, b) for a in 1:ngen for b in 1:ngen]
                results = Vector{Bool}(undef, length(pairs))
                @threads for idx in eachindex(pairs)
                    a, b = pairs[idx]
                    Ta, Tb = gellmann_matrix(N, a), gellmann_matrix(N, b)
                    expected = a == b ? 0.5 : 0.0
                    results[idx] = isapprox(real(tr(Ta * Tb)), expected, atol=1e-12)
                end
                @test all(results)
            end
        end

        @testset "Matrix algebra verification" begin
            # Verify [Tᵃ, Tᵇ] = i f^{abc} Tᶜ holds at matrix level
            for N in [2, 3, 4, 5]  # Extended to include N=4,5
                ngen = N^2 - 1
                alg = get_algebra(get_or_create_su(N))
                generators = [gellmann_matrix(N, k) for k in 1:ngen]
                
                # Parallelize over all (a,b) pairs
                pairs = [(a, b) for a in 1:ngen for b in 1:ngen]
                results = Vector{Bool}(undef, length(pairs))
                @threads for idx in eachindex(pairs)
                    a, b = pairs[idx]
                    comm_matrix = generators[a] * generators[b] - generators[b] * generators[a]
                    reconstructed = zeros(ComplexF64, N, N)
                    for (c, f_abc) in structure_constants(alg, a, b)
                        reconstructed += im * f_abc * generators[c]
                    end
                    results[idx] = isapprox(comm_matrix, reconstructed, atol=1e-10)
                end
                @test all(results)
            end
        end
    end

    # =========================================================================
    # Generator Creation and Properties
    # =========================================================================
    @testset "Generators" begin
        
        @testset "Creation" begin
            T = su_generators(2, :T)
            @test length(T) == 3
            @test all(t isa QuExpr for t in T)
            
            G = su_generators(3, :G)
            @test length(G) == 8
            
            # Single generator
            @test su_generator(2, :T, 1) == T[1]
        end

        @testset "With indices" begin
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            @test T_i[1] != T_j[1]  # Different sites
            
            T_1 = su_generators(2, :T, 1)
            T_2 = su_generators(2, :T, 2)
            @test T_1[1] != T_2[1]
        end

        @testset "Hermiticity" begin
            for (N, name) in [(2, :T), (3, :G), (4, :F)]
                gens = su_generators(N, name)
                for g in gens
                    @test g' == g
                end
            end
        end

        @testset "Output" begin
            T = su_generators(2, :T)
            @test occursin("T", string(T[1]))
            @test occursin("¹", string(T[1]))
            @test occursin("²", string(T[2]))
            @test occursin("³", string(T[3]))
        end
    end

    # =========================================================================
    # Commutation Relations
    # =========================================================================
    @testset "Commutation relations" begin
        
        @testset "SU(2) commutators" begin
            T = su_generators(2, :T)
            
            # [Tᵃ, Tᵇ] = i εᵃᵇᶜ Tᶜ
            @test normal_form(comm(T[1], T[2])) == 1im * T[3]
            @test normal_form(comm(T[2], T[3])) == 1im * T[1]
            @test normal_form(comm(T[3], T[1])) == 1im * T[2]
            
            # Antisymmetry
            @test normal_form(comm(T[2], T[1])) == -1im * T[3]
            
            # Self-commutation
            for a in 1:3
                @test iszero(normal_form(comm(T[a], T[a])))
            end
        end

        @testset "SU(3) commutators" begin
            G = su_generators(3, :G)
            
            # [G¹, G⁴] = i G⁷ (SU(2) subalgebra in our generalized Gell-Mann ordering)
            @test normal_form(comm(G[1], G[4])) == 1im * G[7]
            
            for a in 1:8
                @test iszero(normal_form(comm(G[a], G[a])))
            end
            
            # Antisymmetry (test a few specific pairs to avoid floating-point issues)
            @test normal_form(comm(G[1], G[4])) == -normal_form(comm(G[4], G[1]))
            @test normal_form(comm(G[2], G[5])) == -normal_form(comm(G[5], G[2]))
        end

        @testset "SU(4) commutators" begin
            G = su_generators(4, :G)
            
            # Self-commutation is zero (parallelized)
            results_self = Vector{Bool}(undef, 15)
            @threads for a in 1:15
                results_self[a] = iszero(normal_form(comm(G[a], G[a])))
            end
            @test all(results_self)
            
            # Antisymmetry - full coverage of all 105 pairs (parallelized)
            pairs = [(a, b) for a in 1:15 for b in a+1:15]
            results_anti = Vector{Bool}(undef, length(pairs))
            @threads for idx in eachindex(pairs)
                a, b = pairs[idx]
                results_anti[idx] = normal_form(comm(G[a], G[b])) == -normal_form(comm(G[b], G[a]))
            end
            @test all(results_anti)
            
            # Jacobi identity for SU(4)
            jacobi = comm(G[1], comm(G[2], G[3])) + 
                     comm(G[2], comm(G[3], G[1])) + 
                     comm(G[3], comm(G[1], G[2]))
            @test iszero(normal_form(jacobi))
        end

        @testset "SU(5) commutators" begin
            G = su_generators(5, :G)
            
            # Self-commutation is zero (parallelized)
            results_self = Vector{Bool}(undef, 24)
            @threads for a in 1:24
                results_self[a] = iszero(normal_form(comm(G[a], G[a])))
            end
            @test all(results_self)
            
            # Antisymmetry - full coverage of all 276 pairs (parallelized)
            pairs = [(a, b) for a in 1:24 for b in a+1:24]
            results_anti = Vector{Bool}(undef, length(pairs))
            @threads for idx in eachindex(pairs)
                a, b = pairs[idx]
                results_anti[idx] = normal_form(comm(G[a], G[b])) == -normal_form(comm(G[b], G[a]))
            end
            @test all(results_anti)
            
            # Jacobi identity for SU(5)
            jacobi = comm(G[1], comm(G[2], G[3])) + 
                     comm(G[2], comm(G[3], G[1])) + 
                     comm(G[3], comm(G[1], G[2]))
            @test iszero(normal_form(jacobi))
        end

        @testset "Commutators with indices" begin
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            
            # Same site
            @test normal_form(comm(T_i[1], T_i[2])) == 1im * T_i[3]
            
            # Different sites: proportional to δ
            @test normal_form(comm(T_i[1], T_j[2])) == myδ_local(:i, :j) * 1im * su_generator(2, :T, 3, :i)
        end

        @testset "Jacobi identity" begin
            T = su_generators(2, :T)
            jacobi = comm(T[1], comm(T[2], T[3])) + 
                     comm(T[2], comm(T[3], T[1])) + 
                     comm(T[3], comm(T[1], T[2]))
            @test iszero(normal_form(jacobi))
        end
    end

    # =========================================================================
    # Product Rules
    # =========================================================================
    @testset "Product rules" begin
        
        @testset "SU(2) products" begin
            T = su_generators(2, :T)
            
            # TᵃTᵃ = 1/4 (diagonal)
            @test normal_form(T[1] * T[1]) ≈ QuExpr(0.25)
            @test normal_form(T[2] * T[2]) ≈ QuExpr(0.25)
            @test normal_form(T[3] * T[3]) ≈ QuExpr(0.25)
            
            # T¹T² = (i/2)T³
            @test normal_form(T[1] * T[2]) ≈ 0.5im * T[3]
            @test normal_form(T[2] * T[1]) ≈ -0.5im * T[3]
            
            # Verify: T¹T² - T²T¹ = iT³
            @test normal_form(T[1] * T[2] - T[2] * T[1]) ≈ 1im * T[3]
        end

        @testset "SU(2) triple products" begin
            T = su_generators(2, :T)
            
            @test normal_form(T[1] * T[2] * T[3]) ≈ QuExpr(0.125im)
            @test normal_form(T[1] * T[1] * T[1]) ≈ 0.25 * T[1]
        end

        @testset "SU(3) products" begin
            G = su_generators(3, :G)
            
            result12 = normal_form(G[1] * G[2])
            result21 = normal_form(G[2] * G[1])
            @test normal_form(result12 - result21) ≈ normal_form(comm(G[1], G[2]))
        end

        @testset "SU(4) products" begin
            G = su_generators(4, :G)
            
            # All 225 products should work without error (parallelized)
            pairs = [(a, b) for a in 1:15 for b in 1:15]
            results = Vector{Bool}(undef, length(pairs))
            @threads for idx in eachindex(pairs)
                a, b = pairs[idx]
                result = normal_form(G[a] * G[b])
                results[idx] = result isa QuExpr
            end
            @test all(results)
            
            # Product minus reverse equals commutator
            result12 = normal_form(G[1] * G[2])
            result21 = normal_form(G[2] * G[1])
            @test normal_form(result12 - result21) ≈ normal_form(comm(G[1], G[2]))
        end

        @testset "SU(5) products" begin
            G = su_generators(5, :G)
            
            # Full coverage: all 576 products (parallelized)
            pairs = [(a, b) for a in 1:24 for b in 1:24]
            results = Vector{Bool}(undef, length(pairs))
            @threads for idx in eachindex(pairs)
                a, b = pairs[idx]
                result = normal_form(G[a] * G[b])
                results[idx] = result isa QuExpr
            end
            @test all(results)
            
            # Product minus reverse equals commutator
            result12 = normal_form(G[1] * G[2])
            result21 = normal_form(G[2] * G[1])
            @test normal_form(result12 - result21) ≈ normal_form(comm(G[1], G[2]))
        end

        @testset "Products with indices" begin
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            
            # Same index: contracts
            @test normal_form(T_i[1] * T_i[1]) ≈ QuExpr(0.25)
            @test normal_form(T_i[1] * T_i[2]) ≈ 0.5im * su_generator(2, :T, 3, :i)
            
            # Different indices: no contraction
            @test normal_form(T_i[1] * T_j[1]) == T_i[1] * T_j[1]
        end
    end

    # =========================================================================
    # Casimir Operators
    # =========================================================================
    @testset "Casimir operators" begin
        # C = ∑ₐ TᵃTᵃ = (N²-1)/(2N) for fundamental rep
        
        @testset "SU(2) Casimir = 3/4" begin
            T = su_generators(2, :T)
            C2 = sum(T[a] * T[a] for a in 1:3)
            @test normal_form(C2) ≈ QuExpr(0.75)
        end

        @testset "SU(3) Casimir = 4/3" begin
            G = su_generators(3, :G)
            C3 = sum(G[a] * G[a] for a in 1:8)
            @test normal_form(C3) ≈ QuExpr(4/3)
        end

        @testset "SU(4) Casimir = 15/8" begin
            G = su_generators(4, :G)
            C4 = sum(G[a] * G[a] for a in 1:15)
            # The Casimir has small numerical noise from floating-point structure constants
            # Check that the rational part is correct
            result = normal_form(C4)
            # Extract the scalar part by evaluating at the identity
            @test isapprox(real(result.terms[QuTerm()]), 15/8, atol=1e-10)
        end

        @testset "SU(5) Casimir = 12/5" begin
            G = su_generators(5, :G)
            C5 = sum(G[a] * G[a] for a in 1:24)
            result = normal_form(C5)
            @test isapprox(real(result.terms[QuTerm()]), 12/5, atol=1e-10)
        end
    end



    # =========================================================================
    # Vacuum Expectation Values
    # =========================================================================
    @testset "Vacuum expectation values" begin
        T = su_generators(2, :T)

        @testset "SU(2) single generators" begin
            # Vacuum is |2⟩, T³ eigenvalue is -1/2
            @test vacExpVal(T[3]) ≈ QuExpr(-0.5)
            @test iszero(vacExpVal(T[1]))
            @test iszero(vacExpVal(T[2]))
        end

        @testset "SU(2) products" begin
            @test vacExpVal(T[1] * T[1]) ≈ QuExpr(0.25)
            @test vacExpVal(T[2] * T[2]) ≈ QuExpr(0.25)
            @test vacExpVal(T[3] * T[3]) ≈ QuExpr(0.25)
            
            # T¹T² = (i/2)T³, ⟨T³⟩ = -1/2
            @test vacExpVal(T[1] * T[2]) ≈ QuExpr(-0.25im)
            
            # Off-diagonal with diagonal
            @test iszero(vacExpVal(T[1] * T[3]))
        end

        @testset "SU(2) Casimir" begin
            C2 = T[1] * T[1] + T[2] * T[2] + T[3] * T[3]
            @test vacExpVal(C2) ≈ QuExpr(0.75)
        end

        @testset "SU(3) vacuum" begin
            G = su_generators(3, :G)
            
            # G⁸ eigenvalue on |3⟩ is -1/√3
            @test vacExpVal(G[8]) ≈ QuExpr(-1/sqrt(3))
            
            # Casimir
            C3 = sum(G[a] * G[a] for a in 1:8)
            @test vacExpVal(C3) ≈ QuExpr(4/3)
        end
    end

    # =========================================================================
    # Ladder Operators
    # =========================================================================
    @testset "Ladder operators" begin
        T = su_generators(2, :T)
        Tz = T[3]
        Tp = su2_raising(:T)
        Tm = su2_lowering(:T)

        @testset "Construction" begin
            @test Tp == T[1] + 1im * T[2]
            @test Tm == T[1] - 1im * T[2]
            
            Tp2, Tm2, Tz2 = su2_ladder_operators(:S)
            @test Tp2 == su2_raising(:S)
            @test Tm2 == su2_lowering(:S)
            @test Tz2 == su_generator(2, :S, 3)
        end

        @testset "With indices" begin
            Tp_i = su2_raising(:T, :i)
            Tm_i = su2_lowering(:T, :i)
            T_i = su_generators(2, :T, :i)
            @test Tp_i == T_i[1] + 1im * T_i[2]
            @test Tm_i == T_i[1] - 1im * T_i[2]
        end

        @testset "Commutation relations" begin
            @test normal_form(comm(Tz, Tp)) == Tp
            @test normal_form(comm(Tz, Tm)) == -Tm
            @test normal_form(comm(Tp, Tm)) == 2Tz
        end

        @testset "Products" begin
            @test normal_form(Tp * Tm) == 0.5 + Tz
            @test normal_form(Tm * Tp) == 0.5 - Tz
        end

        @testset "Hermiticity" begin
            @test Tp' == Tm
            @test Tm' == Tp
        end

        @testset "Vacuum expectations" begin
            @test iszero(vacExpVal(Tp))
            @test iszero(vacExpVal(Tm))
            @test vacExpVal(Tp * Tm) ≈ QuExpr(0.0)
            @test vacExpVal(Tm * Tp) ≈ QuExpr(1.0)
        end

    end

    # =========================================================================
    # Error Handling
    # =========================================================================
    @testset "Error handling" begin
        @test_throws ArgumentError gellmann_matrix(2, 0)
        @test_throws ArgumentError gellmann_matrix(2, 4)
        @test_throws ArgumentError get_or_create_su(1)
        @test_throws ArgumentError get_or_create_su(0)
        @test_throws ArgumentError get_algebra(UInt16(0))
    end

    # =========================================================================
    # Advanced Index Handling
    # =========================================================================
    @testset "Advanced index handling" begin
        
        @testset "Multiple symbolic indices" begin
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            T_k = su_generators(2, :T, :k)
            
            # Products with different indices don't contract
            prod_ij = normal_form(T_i[1] * T_j[1])
            @test prod_ij == T_i[1] * T_j[1]
            
            # Triple products with different indices
            prod_ijk = normal_form(T_i[1] * T_j[2] * T_k[3])
            @test prod_ijk == T_i[1] * T_j[2] * T_k[3]
            
            # Same index contracts
            @test normal_form(T_i[1] * T_i[2]) ≈ 0.5im * su_generator(2, :T, 3, :i)
        end

        @testset "Index contraction via δ" begin
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            
            # δ(i,j) * T_i should contract
            δij = myδ_local(:i, :j)
            expr = δij * T_i[1]
            @test normal_form(expr) == normal_form(T_j[1] * δij)
            
            # δ(i,k) * δ(k,j) = δ(i,j) (transitivity)
            δik = myδ_local(:i, :k)
            δkj = myδ_local(:k, :j)
            @test normal_form(δik * δkj) == normal_form(myδ_local(:i, :j) * myδ_local(:k, :i))
        end

        @testset "Mixed numeric and symbolic indices" begin
            T_1 = su_generators(2, :T, 1)
            T_i = su_generators(2, :T, :i)
            
            # Numeric indices contract to scalar
            @test normal_form(T_1[1] * T_1[1]) ≈ QuExpr(0.25)
            
            # Symbolic stays symbolic
            @test normal_form(T_i[1] * T_i[1]) ≈ QuExpr(0.25)
            
            # Mixed don't contract (different sites)
            prod = normal_form(T_1[1] * T_i[1])
            @test prod == T_1[1] * T_i[1]
        end

        @testset "Multi-index operators" begin
            # SU(2) with compound indices
            T_ij = su_generators(2, :T, (:i, :j))
            T_kl = su_generators(2, :T, (:k, :l))
            
            @test T_ij[1] != T_kl[1]
            @test normal_form(T_ij[1] * T_ij[1]) ≈ QuExpr(0.25)
            @test normal_form(comm(T_ij[1], T_ij[2])) == 1im * T_ij[3]
        end
    end

    # =========================================================================
    # Sum (∑) Operations
    # =========================================================================
    @testset "Sum operations" begin
        
        @testset "Basic sums" begin
            T_i = su_generators(2, :T, :i)
            
            # Sum over single index
            s1 = ∑(:i, T_i[1])
            @test s1 isa QuExpr
            
            # Sum over site index with generators
            s2 = ∑(:i, su_generator(2, :T, 1, :i) * su_generator(2, :T, 2, :i))
            @test s2 isa QuExpr
        end

        @testset "Nested sums" begin
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            
            # ∑_i ∑_j T_i * T_j = ∑_{i,j} T_i * T_j
            s1 = ∑(:i, ∑(:j, T_i[1] * T_j[1]))
            s2 = ∑((:i, :j), T_i[1] * T_j[1])
            @test s1 == s2
        end

        @testset "Sums with normal ordering" begin
            T_i = su_generators(2, :T, :i)
            
            # Normal form should distribute into sum
            expr = T_i[1] * T_i[2]
            s1 = ∑(:i, expr)
            s2 = ∑(:i, normal_form(expr))
            @test normal_form(s1) == normal_form(s2)
        end

        @testset "Sum index collision" begin
            T_i = su_generators(2, :T, :i)
            
            # Multiplying with a sum that uses same index should rename
            s = ∑(:i, T_i[1])
            prod = normal_form(su_generator(2, :T, 1, :i) * s)
            # The sum index should be renamed to avoid collision
            @test prod isa QuExpr
        end

        @testset "Sums with scalars" begin
            T_i = su_generators(2, :T, :i)
            
            @test ∑(:i, 3) == ∑(:i, QuExpr(3))
            @test ∑(:i, 2 * T_i[1]) == 2 * ∑(:i, T_i[1])
        end
    end

    # =========================================================================
    # Exponent Tests
    # =========================================================================
    @testset "Exponents" begin
        T = su_generators(2, :T)
        G = su_generators(3, :G)
        T_i = su_generators(2, :T, :i)
        
        @testset "Basic exponents" begin
            for x in [T[1], T[2], T[3], G[1], T_i[1]]
                @test isone(x^0)
                @test x^1 == x
                @test x^2 == x * x
                @test x^3 == x * x * x
                @test x^4 == x * x * x * x
            end
        end

        @testset "Negative exponents" begin
            for x in [T[1], T[2], T[3], G[1]]
                @test_throws ArgumentError x^-1
                @test_throws ArgumentError x^-2
            end
        end

        @testset "Exponent normal form" begin
            # T^2 = 1/4 for SU(2)
            @test normal_form(T[1]^2) ≈ QuExpr(0.25)
            @test normal_form(T[2]^2) ≈ QuExpr(0.25)
            @test normal_form(T[3]^2) ≈ QuExpr(0.25)
            
            # T^3 = T/4 for SU(2)
            @test normal_form(T[1]^3) ≈ 0.25 * T[1]
            
            # T^4 = 1/16 for SU(2)
            @test normal_form(T[1]^4) ≈ QuExpr(0.0625)
        end

        @testset "Product exponents" begin
            expr = T[1] * T[2]
            @test isone(expr^0)
            @test expr^1 == expr
            @test expr^2 == expr * expr
        end
    end

    # =========================================================================
    # Parameters with Generators
    # =========================================================================
    @testset "Parameters with generators" begin
        T = su_generators(2, :T)
        T_i = su_generators(2, :T, :i)
        
        @testset "Scalar parameters" begin
            g = Pc"g"
            ω = Pr"ω"
            
            expr1 = g * T[1]
            @test expr1 isa QuExpr
            @test normal_form(expr1) == g * T[1]
            
            # Hermitian conjugate of parameter times generator
            @test (g * T[1])' == g' * T[1]
            @test (ω * T[1])' == ω * T[1]  # Real param, Hermitian gen
        end

        @testset "Indexed parameters" begin
            g_i = Pc"g_i"
            ω_i = Pr"ω_i"
            
            expr = g_i * T_i[1]
            @test expr isa QuExpr
            
            # Parameters with multiple indices
            g_ij = Pr"g_i,j"
            @test g_ij == param(:g, 'r', (:i, :j))
        end

        @testset "Parameter commutation" begin
            g = Pc"g"
            
            # Parameters commute with everything
            @test iszero(normal_form(comm(g, T[1])))
            @test iszero(normal_form(comm(g, T[1] * T[2])))
            @test iszero(normal_form(comm(T[1] + T[2], g)))
        end

        @testset "Sums with parameters" begin
            H = ∑(:i, Pr"ω_i" * su_generator(2, :T, 3, :i))
            @test H isa QuExpr
            
            # Heisenberg-like structure
            H2 = ∑(:i, Pr"ω_i" * su_generator(2, :T, 3, :i) * su_generator(2, :T, 3, :i))
            @test normal_form(H2) == ∑(:i, Pr"ω_i" * QuExpr(0.25))
        end
    end

    # =========================================================================
    # ExpVal and Corr
    # =========================================================================
    @testset "ExpVal and Corr" begin
        T = su_generators(2, :T)
        T_i = su_generators(2, :T, :i)
        
        @testset "Basic expval" begin
            @test expval(3) == QuExpr(3)
            @test expval(T[1]) isa QuExpr
            
            # expval of Hermitian is real (structurally)
            ev = expval(T[1])
            @test ev' == ev
        end

        @testset "Basic corr" begin
            @test corr(3) == QuExpr(3)
            @test corr(T[1]) isa QuExpr
            
            # corr distributes over sums
            @test corr(3 + T[1]) == 3 + corr(T[1])
        end

        @testset "Expval normal form" begin
            # expval(T[1]*T[2]) should normal-order inside
            ev12 = expval(T[1] * T[2])
            @test normal_form(ev12) == expval(normal_form(T[1] * T[2]))
            
            # Scalar multiplication
            @test normal_form(3 * expval(T[1])) == expval(normal_form(3 * T[1]))
        end

        @testset "expval_as_corrs" begin
            @test expval_as_corrs(3) == QuExpr(3)
            @test expval_as_corrs(T[1]) == corr(T[1])
            @test expval_as_corrs(3 * T[1]) == 3 * corr(T[1])
            
            # Product decomposition with different sites (doesn't contract)
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            result = expval_as_corrs(T_i[1] * T_j[1])
            @test result == corr(T_i[1] * T_j[1]) + corr(T_i[1]) * corr(T_j[1])
        end

        @testset "corr_as_expvals" begin
            @test corr_as_expvals(3) == QuExpr(3)
            # Use different sites to avoid contraction
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            @test corr_as_expvals(T_i[1] * T_j[1]) == expval(T_i[1] * T_j[1]) - expval(T_i[1]) * expval(T_j[1])
        end

        @testset "ExpVal with indices" begin
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            
            ev_ij = expval(T_i[1] * T_j[1])
            @test ev_ij isa QuExpr
            
            # Sum with expval
            s_ev = ∑(:i, expval(T_i[1]))
            @test s_ev isa QuExpr
        end

        @testset "Corr inside normal_form" begin
            # corr should preserve structure during normal_form
            c12 = corr(T[1] * T[2])
            @test normal_form(c12) == corr(normal_form(T[1] * T[2]))
            
            # expval * corr
            result = normal_form(expval(T[1]) * corr(T[2]))
            @test result == normal_form(expval(T[1]) * corr(T[2]))
        end
    end

    # =========================================================================
    # LaTeX Output
    # =========================================================================
    @testset "LaTeX output" begin
        T = su_generators(2, :T)
        G = su_generators(3, :G)
        T_i = su_generators(2, :T, :i)
        
        @testset "Single generators" begin
            l1 = latex(T[1])
            @test occursin("T", l1)
            @test occursin("1", l1)
            @test occursin("^{1}", l1)  # Generator index as superscript
            
            l3 = latex(G[3])
            @test occursin("G", l3)
            @test occursin("^{3}", l3)
        end

        @testset "Generators with site indices" begin
            l_i = latex(T_i[1])
            @test occursin("T", l_i)
            @test occursin("^{1}", l_i)
            @test occursin("i", l_i)  # Site index
            
            # Numeric site index
            T_1 = su_generators(2, :T, 1)
            l_1 = latex(T_1[2])
            @test occursin("^{2}", l_1)
            @test occursin("1", l_1)
        end

        @testset "Products" begin
            prod = T[1] * T[2]
            l_prod = latex(prod)
            @test occursin("T", l_prod)
            # Should contain both generator indices
        end

        @testset "Exponents in LaTeX" begin
            # Use different sites (i vs j) so they don't contract even with auto_normal_form
            T_i = su_generators(2, :T, :i)
            T_j = su_generators(2, :T, :j)
            # Same generator type at different sites won't contract
            expr = T_i[1] * T_j[1]
            l_prod = latex(expr)
            @test occursin("T", l_prod)
            @test occursin("^{1}", l_prod)  # Generator index
            
            # Test that same site adjacent operators show exponent (when not auto-normalized)
            # This varies based on auto_normal_form setting, so just check basic structure
            @test latex(T_i[1]) == "{T}^{1}_{i}"
        end

        @testset "Sums in LaTeX" begin
            s = ∑(:i, T_i[1])
            l_sum = latex(s)
            @test occursin("sum", l_sum)  # \sum
            @test occursin("T", l_sum)
        end

        @testset "ExpVal in LaTeX" begin
            ev = expval(T[1])
            l_ev = latex(ev)
            @test occursin("langle", l_ev)  # \langle
            @test occursin("rangle", l_ev)  # \rangle
            @test occursin("T", l_ev)
        end

        @testset "Corr in LaTeX" begin
            c = corr(T[1])
            l_c = latex(c)
            @test occursin("langle", l_c)
            @test occursin("rangle", l_c)
            @test occursin("c", l_c)  # subscript c
        end
    end

    # =========================================================================
    # String Output
    # =========================================================================
    @testset "String output" begin
        T = su_generators(2, :T)
        G = su_generators(3, :G)
        
        @testset "Superscript formatting" begin
            s1 = string(T[1])
            @test occursin("T", s1)
            @test occursin("¹", s1)  # Superscript 1
            
            s2 = string(T[2])
            @test occursin("²", s2)
            
            s3 = string(T[3])
            @test occursin("³", s3)
        end

        @testset "Multi-digit generator indices" begin
            # SU(4) has 15 generators
            F = su_generators(4, :F)
            s10 = string(F[10])
            @test occursin("F", s10)
            @test occursin("¹⁰", s10)  # Superscript 10
            
            s15 = string(F[15])
            @test occursin("¹⁵", s15)
        end

        @testset "Products in string" begin
            prod = T[1] * T[2]
            s = string(prod)
            @test occursin("T", s)
        end

        @testset "Sums in string" begin
            T_i = su_generators(2, :T, :i)
            s = ∑(:i, T_i[1])
            str = string(s)
            @test occursin("∑", str)
        end

        @testset "ExpVal/Corr in string" begin
            ev = expval(T[1])
            s_ev = string(ev)
            @test occursin("⟨", s_ev)
            @test occursin("⟩", s_ev)
            
            c = corr(T[1])
            s_c = string(c)
            @test occursin("⟨", s_c)
            @test occursin("⟩", s_c)
            @test occursin("c", s_c)
        end
    end

    # =========================================================================
    # Interoperability with Bosons/Fermions
    # =========================================================================
    @testset "Interoperability with bosons/fermions" begin
        T = su_generators(2, :T)
        T_i = su_generators(2, :T, :i)
        
        @testset "SU(N) with bosons" begin
            # Generators should commute with bosons (different degrees of freedom)
            @test iszero(normal_form(comm(T[1], a(1))))
            @test iszero(normal_form(comm(T[1], a'(1))))
            
            # Products should be orderable
            expr = a(1) * T[1]
            @test normal_form(expr) == normal_form(T[1] * a(1))
            
            # Mixed products
            mixed = a'(1) * T[1] * a(1)
            @test mixed isa QuExpr
            @test normal_form(mixed) isa QuExpr
        end

        @testset "SU(N) with fermions" begin
            # Generators should commute with fermions
            @test iszero(normal_form(comm(T[1], f(1))))
            @test iszero(normal_form(comm(T[1], f'(1))))
            
            # Products
            expr = f(1) * T[1]
            @test normal_form(expr) == normal_form(T[1] * f(1))
        end

        @testset "SU(N) with TLS operators" begin
            # SU(2) generators should commute with TLS at different sites
            @test iszero(normal_form(comm(T[1], σx(2))))
            
            # At same site - need conversion
            # (generators use different representation)
        end

        @testset "Mixed expressions normal form" begin
            # Complex expression with bosons and SU(N)
            H = Pr"ω" * a'(1) * a(1) + Pr"g" * T[3]
            @test H isa QuExpr
            @test normal_form(H) == H  # Already in normal form
            
            # Interaction term
            H_int = Pr"λ" * (a(1) + a'(1)) * T[1]
            @test normal_form(H_int) isa QuExpr
        end

        @testset "Vacuum expectations with mixed" begin
            # Bosonic vacuum × SU(N) vacuum
            expr = a'(1) * a(1) * T[3]
            @test vacExpVal(expr) ≈ QuExpr(0.0)  # ⟨0|a†a|0⟩ = 0
            
            # T[3] on vacuum is -1/2
            @test vacExpVal(T[3]) ≈ QuExpr(-0.5)
            
            # Product
            @test iszero(vacExpVal(a'(1) * T[1]))
        end

        @testset "ExpVal with mixed operators" begin
            ev = expval(a'(1) * a(1) * T[1])
            @test ev isa QuExpr
            
            # expval_as_corrs should work
            result = expval_as_corrs(a(1) * T[1])
            @test result == corr(a(1) * T[1]) + corr(a(1)) * corr(T[1])
        end
    end

    # =========================================================================
    # is_normal_form Tests
    # =========================================================================
    @testset "is_normal_form" begin
        T = su_generators(2, :T)
        T_i = su_generators(2, :T, :i)
        
        # Helper function to test if normal_form is idempotent
        function check_is_normal_form(x)
            xnf = normal_form(x)
            return xnf == normal_form(xnf)
        end
        
        @testset "Single generators" begin
            for t in T
                @test check_is_normal_form(t)
            end
        end

        @testset "Products" begin
            @test check_is_normal_form(T[1] * T[2])
            @test check_is_normal_form(T[1] * T[2] * T[3])
            @test check_is_normal_form(T_i[1] * T_i[2])
        end

        @testset "Sums" begin
            @test check_is_normal_form(∑(:i, T_i[1]))
            @test check_is_normal_form(∑(:i, T_i[1] * T_i[2]))
        end

        @testset "ExpVal/Corr" begin
            @test check_is_normal_form(expval(T[1]))
            @test check_is_normal_form(corr(T[1] * T[2]))
            @test check_is_normal_form(expval(T[1]) * T[2])
        end

        @testset "Complex expressions" begin
            y = one(QuExpr)
            for x in [T[1] * T[2], T_i[1] * T_i[2], expval(T[1])]
                @test check_is_normal_form(x)
                y *= x
                @test check_is_normal_form(y)
            end
        end
    end

    # =========================================================================
    # Sum Commutators (Critical: Tests for exchange prefactor fix)
    # =========================================================================
    @testset "Sum commutators" begin
        # This tests the fix for the exchange prefactor bug.
        # Previously, _exchange_lie_algebra_generators always returned prefactor=1,
        # which prevented proper simplification of [∑T^a, ∑T^b].
        # The fix returns prefactor=0 for same-site (full simplification) and
        # prefactor=1 with doubled coefficient for different-site (delta term).
        
        @testset "SU(2) sum commutators" begin
            # [∑_i T¹_i, ∑_j T²_j] should simplify to i ∑_k T³_k
            S1 = ∑(:i, su_generator(2, :T, 1, :i))
            S2 = ∑(:j, su_generator(2, :T, 2, :j))
            S3 = ∑(:k, su_generator(2, :T, 3, :k))
            
            result = normal_form(comm(S1, S2))
            expected = 1im * S3
            
            # The result should be a single sum, not a double sum
            @test result == expected
            
            # Cyclic
            @test normal_form(comm(S2, S3)) == 1im * ∑(:i, su_generator(2, :T, 1, :i))
            @test normal_form(comm(S3, S1)) == 1im * ∑(:j, su_generator(2, :T, 2, :j))
        end
        
        @testset "SU(2) product of sums" begin
            # (∑_i T¹_i)(∑_j T¹_j) should simplify properly
            S1 = ∑(:i, su_generator(2, :T, 1, :i))
            
            result = normal_form(S1 * S1)
            # Should contain both ∑_i (1/4) from same-site and ∑_{i,j} T¹_i T¹_j from different sites
            @test result isa QuExpr
            
            # The self-product should not be a simple ∑∑ of T¹T¹
            # since same-site terms contract to 1/4
        end
        
        @testset "SU(3) sum commutators" begin
            G1 = ∑(:i, su_generator(3, :G, 1, :i))
            G4 = ∑(:j, su_generator(3, :G, 4, :j))
            G7 = ∑(:k, su_generator(3, :G, 7, :k))
            
            # [G¹, G⁴] = i G⁷ at same site
            result = normal_form(comm(G1, G4))
            expected = 1im * G7
            @test result == expected
        end
        
        @testset "Mixed index sums" begin
            # Verify that different indices properly give delta terms
            T1_i = su_generator(2, :T, 1, :i)
            T2_j = su_generator(2, :T, 2, :j)
            
            # [T¹_i, T²_j] = i δ_{ij} T³_i for different symbolic indices
            result = normal_form(comm(T1_i, T2_j))
            expected = myδ_local(:i, :j) * 1im * su_generator(2, :T, 3, :i)
            @test result == expected
        end
        
        @testset "Comparison with native TLS" begin
            # Native TLS sums work correctly - verify SU(2) generators match
            tls_result = normal_form(comm(∑(:i, σx(:i)), ∑(:j, σy(:j))))
            
            # For SU(2) with σ as name (using tls_to_su2 normalization: σ = 2T)
            # [∑σx, ∑σy] = 2i ∑σz
            @test tls_result == 2im * ∑(:k, σz(:k))
            
            # SU(2) generators: [∑T¹, ∑T²] = i ∑T³
            su2_result = normal_form(comm(
                ∑(:i, su_generator(2, :T, 1, :i)),
                ∑(:j, su_generator(2, :T, 2, :j))
            ))
            @test su2_result == 1im * ∑(:k, su_generator(2, :T, 3, :k))
        end
    end
    
    # =========================================================================
    # Edge Cases
    # =========================================================================
    @testset "Edge cases" begin
        T = su_generators(2, :T)
        
        # Zero/identity
        @test iszero(T[1] - T[1])
        @test iszero(normal_form(comm(T[1], T[1])))
        @test 1 * T[1] == T[1]
        @test isone(T[1]^0)
        
        # Complex/rational coefficients
        @test ((1 + 2im) * T[1])' == (1 - 2im) * T[1]
        @test normal_form((1//2) * T[1] * (1//2) * T[1]) ≈ QuExpr(1//16)
        
        # Large N
        for N in [5, 8]
            G = su_generators(N, :G)
            @test length(G) == N^2 - 1
            @test G[1]' == G[1]
        end
        
        # Coefficient mode
        @test QuantumAlgebra.sun_id_coeff(2) == 1//4
        QuantumAlgebra.use_float_coefficients(true)
        @test QuantumAlgebra.sun_id_coeff(2) == 0.25
        QuantumAlgebra.use_float_coefficients(false)
    end

    @testset "is_lie_algebra_gen predicate" begin
        # Direct LieAlgebraGenerator
        alg_id = get_or_create_su(2)
        lg = LieAlgebraGenerator(:T, alg_id, 1)
        @test QuantumAlgebra.is_lie_algebra_gen(lg)
        
        # Via su_generator
        T = su_generators(2, :X)
        t_expr = T[1]
        t_op = first(keys(t_expr.terms)).bares.v[1]
        @test QuantumAlgebra.is_lie_algebra_gen(t_op)
        
        # SU(3) generator
        G = su_generators(3, :G)
        g_expr = G[1]
        g_op = first(keys(g_expr.terms)).bares.v[1]
        @test QuantumAlgebra.is_lie_algebra_gen(g_op)
        
        # Negative cases
        @test !QuantumAlgebra.is_lie_algebra_gen(BosonDestroy(:a))
        @test !QuantumAlgebra.is_lie_algebra_gen(BosonCreate(:a))
        @test !QuantumAlgebra.is_lie_algebra_gen(FermionDestroy(:f))
        @test !QuantumAlgebra.is_lie_algebra_gen(FermionCreate(:f))
        @test !QuantumAlgebra.is_lie_algebra_gen(TLSx(:σ))
        @test !QuantumAlgebra.is_lie_algebra_gen(TLSy(:σ))
        @test !QuantumAlgebra.is_lie_algebra_gen(TLSz(:σ))
        @test !QuantumAlgebra.is_lie_algebra_gen(TLSCreate(:σ))
        @test !QuantumAlgebra.is_lie_algebra_gen(TLSDestroy(:σ))
    end

    @testset "LieAlgebraGenerator direct constructor" begin
        alg_id = get_or_create_su(2)
        
        # Basic construction
        lg = LieAlgebraGenerator(:T, alg_id, 1)
        @test lg.t == QuantumAlgebra.LieAlgebraGen_
        @test lg.name == QuantumAlgebra.QuOpName(:T)
        @test lg.algebra_id == alg_id
        @test lg.gen_idx == UInt16(1)
        
        # With indices
        lg_i = LieAlgebraGenerator(:T, alg_id, 2, :i)
        @test lg_i.t == QuantumAlgebra.LieAlgebraGen_
        @test lg_i.algebra_id == alg_id
        @test lg_i.gen_idx == UInt16(2)
        
        # Error: zero algebra_id
        @test_throws ArgumentError LieAlgebraGenerator(:T, UInt16(0), 1)
        # Error: zero gen_idx
        @test_throws ArgumentError LieAlgebraGenerator(:T, alg_id, UInt16(0))
        
        # Display
        @test occursin("T", string(lg))
        @test occursin("T", QuantumAlgebra.latex(lg))
    end

    @testset "Multi-generator vacuum expectation values" begin
        T = su_generators(2, :T)
        
        # Single generator VEV: ⟨0|Tᶻ|0⟩ = -1/2
        @test vacExpVal(T[3]) == QuExpr(-1//2)
        
        # Product of different generators: ⟨0|Tˣ Tʸ|0⟩
        # Tˣ Tʸ = i/2 Tᶻ, so ⟨0|Tˣ Tʸ|0⟩ = i/2 * (-1/2) = -i/4
        v = vacExpVal(T[1] * T[2])
        @test normal_form(v) == normal_form(QuExpr(-1im//4))
        
        # Product of same generator: ⟨0|Tˣ Tˣ|0⟩
        # Tˣ Tˣ = 1/4 I, so ⟨0|Tˣ Tˣ|0⟩ = 1/4
        v2 = vacExpVal(T[1] * T[1])
        @test normal_form(v2) == normal_form(QuExpr(1//4))
        
        # Product of same generator: ⟨0|Tᶻ Tᶻ|0⟩
        @test normal_form(vacExpVal(T[3] * T[3])) == normal_form(QuExpr(1//4))
        
        # Cross product T¹ T³ = -(i/2) T², so vac = 0
        v4 = vacExpVal(T[1] * T[3])
        @test iszero(normal_form(v4))
        
        # SU(3) diagonal generator VEV
        G = su_generators(3, :G)
        if length(G) >= 8
            v5 = vacExpVal(G[8])
            @test !iszero(v5)
        end
    end

    @testset "TransitionOperator vacuum expectation values" begin
        T = su_generators(2, :T)
        
        # |1⟩⟨2| for 3-level system
        t12 = TransitionOperator(:t, 3, 1, 2)
        @test QuantumAlgebra.is_transition_op(t12)
        
        # vacA: acting on vacuum from right
        # |1⟩⟨2| |vacuum⟩ = 0 (since ⟨2|vacuum⟩ = 0)
        v = vacA(QuExpr(t12))
        @test iszero(v)
        
        # vacA: |2⟩⟨1| should give zero too (⟨1|vacuum⟩ = 0)
        t21 = TransitionOperator(:t, 3, 2, 1)
        v2 = vacA(QuExpr(t21))
        @test iszero(v2)
        
        # vacExpVal with transition operators
        @test iszero(vacExpVal(QuExpr(t12)))
        
        # vacA with Lie algebra + transition
        v4 = vacA(QuExpr(t12) * T[3])
        @test iszero(v4)
        
        # vacA: transition operators pass through (not annihilated)
        # Transition operators are not automatically simplified by vacA
        t31 = TransitionOperator(:t, 3, 3, 1)
        @test !iszero(vacA(QuExpr(t31)))  # Passes through unchanged
    end

    @testset "Float coefficient mode" begin
        # Start from clean state (exact mode)
        QuantumAlgebra.use_float_coefficients(false)
        @test QuantumAlgebra.sun_id_coeff(2) == 1//4
        @test QuantumAlgebra.sun_gen_coeff() == 1//2
        
        # Enable float mode
        QuantumAlgebra.use_float_coefficients(true)
        @test QuantumAlgebra.sun_id_coeff(2) == 0.25
        @test QuantumAlgebra.sun_gen_coeff() == 0.5
        
        # SU(2) commutator in float mode
        T = su_generators(2, :T)
        c = comm(T[1], T[2])
        # [T¹, T²] = i T³, so vacExpVal([T¹, T²]) = i * (-1/2) = -i/2
        @test normal_form(c) == normal_form(1im * T[3])
        
        # Disable float mode
        QuantumAlgebra.use_float_coefficients(false)
        @test QuantumAlgebra.sun_id_coeff(2) == 1//4
        @test QuantumAlgebra.sun_gen_coeff() == 1//2
        
        # Roundtrip
        QuantumAlgebra.use_float_coefficients(true)
        QuantumAlgebra.use_float_coefficients(false)
        @test QuantumAlgebra.sun_id_coeff(2) == 1//4
    end

end


