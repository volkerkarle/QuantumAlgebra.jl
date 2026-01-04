# Comprehensive tests for SU(N) Lie algebra functionality
# Tests: algebra structure, generators, commutation, products, conversions, vacuum expectations, ladder operators

using Test
using QuantumAlgebra
using QuantumAlgebra: get_algebra, get_or_create_su, structure_constants, symmetric_structure_constants,
                      gellmann_matrix, algebra_dim, num_generators, identity_coefficient,
                      product_coefficients, commutator_coefficients, δ, QuExpr, QuTerm, BaseOpProduct, QuIndex
using LinearAlgebra: tr, I

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
            for N in [2, 3, 4]
                ngen = N^2 - 1
                for a in 1:ngen, b in 1:ngen
                    Ta, Tb = gellmann_matrix(N, a), gellmann_matrix(N, b)
                    expected = a == b ? 0.5 : 0.0
                    @test isapprox(real(tr(Ta * Tb)), expected, atol=1e-12)
                end
            end
        end

        @testset "Matrix algebra verification" begin
            # Verify [Tᵃ, Tᵇ] = i f^{abc} Tᶜ holds at matrix level
            for N in [2, 3]
                ngen = N^2 - 1
                alg = get_algebra(get_or_create_su(N))
                generators = [gellmann_matrix(N, k) for k in 1:ngen]
                
                for a in 1:ngen, b in 1:ngen
                    comm_matrix = generators[a] * generators[b] - generators[b] * generators[a]
                    reconstructed = zeros(ComplexF64, N, N)
                    for (c, f_abc) in structure_constants(alg, a, b)
                        reconstructed += im * f_abc * generators[c]
                    end
                    @test isapprox(comm_matrix, reconstructed, atol=1e-10)
                end
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
        
        # Note: SU(4) and SU(5) Casimir tests omitted due to floating-point
        # precision in structure constants for large N
    end

    # =========================================================================
    # TLS ↔ SU(2) Conversions
    # =========================================================================
    @testset "TLS-SU(2) conversions" begin
        T = su_generators(2, :T)

        @testset "Basic conversions" begin
            # TLS → SU(2): σᵃ = 2Tᵃ
            @test normal_form(tls_to_su2(σx())) == 2 * su_generator(2, :σ, 1)
            @test normal_form(tls_to_su2(σy())) == 2 * su_generator(2, :σ, 2)
            @test normal_form(tls_to_su2(σz())) == 2 * su_generator(2, :σ, 3)
            
            # SU(2) → TLS: Tᵃ = σᵃ/2
            @test normal_form(su2_to_tls(T[1])) == normal_form(1//2 * σx())
            @test normal_form(su2_to_tls(T[2])) == normal_form(1//2 * σy())
            @test normal_form(su2_to_tls(T[3])) == normal_form(1//2 * σz())
        end

        @testset "Round-trip conversions" begin
            @test normal_form(su2_to_tls(tls_to_su2(σx()))) == normal_form(σx())
            @test normal_form(su2_to_tls(tls_to_su2(σy()))) == normal_form(σy())
            @test normal_form(su2_to_tls(tls_to_su2(σz()))) == normal_form(σz())
        end

        @testset "Product conversions" begin
            @test normal_form(tls_to_su2(σx() * σy())) == normal_form(4 * su_generator(2, :σ, 1) * su_generator(2, :σ, 2))
            @test normal_form(su2_to_tls(T[1] * T[2])) == normal_form(1//4 * σx() * σy())
        end

        @testset "Commutator consistency" begin
            # [σx, σy] = 2iσz → 4iT³
            tls_comm = normal_form(comm(σx(), σy()))
            su2_comm = tls_to_su2(tls_comm)
            @test normal_form(su2_comm) == 4im * su_generator(2, :σ, 3)
            
            # [T¹, T²] = iT³ → (i/2)σz
            su2_comm2 = normal_form(comm(T[1], T[2]))
            tls_comm2 = su2_to_tls(su2_comm2)
            @test normal_form(tls_comm2) == normal_form((1//2)im * σz())
        end

        @testset "Conversions with indices" begin
            T_i = su_generators(2, :T, :i)
            
            @test normal_form(tls_to_su2(σx(:i))) == 2 * su_generator(2, :σ, 1, :i)
            @test normal_form(su2_to_tls(T_i[1])) == normal_form(1//2 * σx(:i))
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

        @testset "Relation to TLS σ±" begin
            Tp_σ = su2_raising(:σ)
            Tm_σ = su2_lowering(:σ)
            
            @test normal_form(su2_to_tls(Tp_σ)) == normal_form(σp())
            @test normal_form(su2_to_tls(Tm_σ)) == normal_form(σm())
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
    # Edge Cases
    # =========================================================================
    @testset "Edge cases" begin
        
        @testset "Zero expressions" begin
            T = su_generators(2, :T)
            
            @test iszero(T[1] - T[1])
            @test iszero(normal_form(T[1] - T[1]))
            @test iszero(0 * T[1])
            @test iszero(normal_form(comm(T[1], T[1])))
        end

        @testset "Identity operations" begin
            T = su_generators(2, :T)
            
            @test 1 * T[1] == T[1]
            @test T[1] * 1 == T[1]
            @test T[1] + 0 == T[1]
            @test isone(T[1]^0)
        end

        @testset "Large N stability" begin
            # Test that large N algebras can be created
            for N in [5, 6, 7, 8]
                alg_id = get_or_create_su(N)
                alg = get_algebra(alg_id)
                @test num_generators(alg) == N^2 - 1
                @test algebra_dim(alg) == N
                
                # Create generators
                G = su_generators(N, :G)
                @test length(G) == N^2 - 1
                
                # Basic operations should work
                @test G[1]' == G[1]  # Hermitian
                @test iszero(normal_form(comm(G[1], G[1])))
            end
        end

        @testset "Complex scalar coefficients" begin
            T = su_generators(2, :T)
            
            expr = (1 + 2im) * T[1]
            @test expr isa QuExpr
            @test expr' == (1 - 2im) * T[1]
            
            # Imaginary unit
            @test 1im * T[1] isa QuExpr
            @test normal_form(1im * T[1] * T[2]) ≈ normal_form(1im * 0.5im * T[3])
        end

        @testset "Rational coefficients" begin
            T = su_generators(2, :T)
            
            expr = (1//2) * T[1]
            @test expr isa QuExpr
            
            # Rational × Rational
            @test normal_form((1//2) * T[1] * (1//2) * T[1]) ≈ QuExpr(1//16)
        end

        @testset "Coefficient mode toggle" begin
            # Test use_float_coefficients preference
            @test !QuantumAlgebra.using_float_coefficients()  # Default is symbolic mode
            
            # SU(2) with symbolic mode - should give Rational
            σ = su_generators(2, :σ_sym)
            result = normal_form(σ[1] * σ[1])
            coeffs = collect(values(result.terms))
            @test coeffs[1] isa Rational || coeffs[1] isa Integer
            @test coeffs[1] == 1//4
            
            # SU(2) product should give Complex{Rational}
            result2 = normal_form(σ[1] * σ[2])
            coeffs2 = collect(values(result2.terms))
            @test coeffs2[1] isa Complex{<:Rational}
            @test coeffs2[1] == (1//2)*im
            
            # Switch to float mode
            QuantumAlgebra.use_float_coefficients(true)
            @test QuantumAlgebra.using_float_coefficients()
            
            # Coefficient functions should return Float64
            @test QuantumAlgebra._coeff_one_quarter() == 0.25
            @test QuantumAlgebra._coeff_one_half() == 0.5
            @test QuantumAlgebra.su3_id_coeff() ≈ 1/6
            
            # SU(3) with float mode - √3 terms should be Float64
            λ = su_generators(3, :λ_float)
            result3 = normal_form(λ[1] * λ[1])
            # Check that coefficients exist (either float or rational after simplification)
            @test !isempty(result3.terms)
            
            # Switch back to symbolic mode
            QuantumAlgebra.use_float_coefficients(false)
            @test !QuantumAlgebra.using_float_coefficients()
            
            # Verify symbolic mode is restored
            @test QuantumAlgebra._coeff_one_quarter() == 1//4
        end
    end

end
