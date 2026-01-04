# Comprehensive tests for N-level transition operators |i⟩⟨j|
# Tests: creation, products, commutators, adjoint, vacuum expectations, output

using Test
using QuantumAlgebra
using QuantumAlgebra: δ, QuExpr, QuTerm, BaseOpProduct, QuIndex, BaseOperator,
                      transition_levels, transition_dim, is_transition_op

@testset "Transition Operators" begin

    # =========================================================================
    # Creation and Basic Properties
    # =========================================================================
    @testset "Creation" begin
        
        @testset "2-level system" begin
            # TransitionOperator returns BaseOperator, wrap in QuExpr for algebra
            σ12_base = TransitionOperator(:σ, 2, 1, 2)
            σ21_base = TransitionOperator(:σ, 2, 2, 1)
            σ11_base = TransitionOperator(:σ, 2, 1, 1)
            σ22_base = TransitionOperator(:σ, 2, 2, 2)
            
            @test σ12_base isa BaseOperator
            @test σ21_base isa BaseOperator
            @test σ11_base isa BaseOperator
            @test σ22_base isa BaseOperator
            
            # Wrap in QuExpr for comparisons
            σ12 = QuExpr(σ12_base)
            σ21 = QuExpr(σ21_base)
            σ11 = QuExpr(σ11_base)
            σ22 = QuExpr(σ22_base)
            
            # Different operators are different
            @test σ12 != σ21
            @test σ11 != σ22
            @test σ12 != σ11
        end

        @testset "3-level system" begin
            Σ = nlevel_ops(3, :Σ)
            @test size(Σ) == (3, 3)
            @test all(op isa QuExpr for op in Σ)
            
            # All 9 operators are distinct
            ops = vec(Σ)
            for i in 1:9, j in i+1:9
                @test ops[i] != ops[j]
            end
        end

        @testset "4-level system" begin
            Σ = nlevel_ops(4, :Λ)
            @test size(Σ) == (4, 4)
            @test length(unique(vec(Σ))) == 16
        end

        @testset "With indices" begin
            σ12_i = QuExpr(TransitionOperator(:σ, 2, 1, 2, :i))
            σ12_j = QuExpr(TransitionOperator(:σ, 2, 1, 2, :j))
            σ12_1 = QuExpr(TransitionOperator(:σ, 2, 1, 2, 1))
            σ12_2 = QuExpr(TransitionOperator(:σ, 2, 1, 2, 2))
            
            # Different sites are different
            @test σ12_i != σ12_j
            @test σ12_1 != σ12_2
            @test σ12_i != σ12_1
        end

        @testset "σ shorthand for 2-level" begin
            s12 = σ(:σ, 1, 2)
            t12 = TransitionOperator(:σ, 2, 1, 2)
            @test s12 == t12
            
            s12_i = σ(:σ, 1, 2, :i)
            t12_i = TransitionOperator(:σ, 2, 1, 2, :i)
            @test s12_i == t12_i
        end

        @testset "nlevel_ops with indices" begin
            Σ = nlevel_ops(3, :Σ, :i)
            @test size(Σ) == (3, 3)
            
            Σ_j = nlevel_ops(3, :Σ, :j)
            @test Σ[1, 2] != Σ_j[1, 2]  # Different sites
        end

        @testset "Error handling" begin
            # Out of bounds
            @test_throws ArgumentError TransitionOperator(:σ, 2, 0, 1)  # i < 1
            @test_throws ArgumentError TransitionOperator(:σ, 2, 3, 1)  # i > N
            @test_throws ArgumentError TransitionOperator(:σ, 2, 1, 0)  # j < 1
            @test_throws ArgumentError TransitionOperator(:σ, 2, 1, 3)  # j > N
        end
    end

    # =========================================================================
    # Helper Functions
    # =========================================================================
    @testset "Helper functions" begin
        σ12_base = TransitionOperator(:σ, 2, 1, 2)
        σ33_base = TransitionOperator(:σ, 3, 3, 3)
        
        @testset "transition_dim" begin
            @test transition_dim(σ12_base) == 2
            @test transition_dim(σ33_base) == 3
        end

        @testset "transition_levels" begin
            @test transition_levels(σ12_base) == (1, 2)
            @test transition_levels(σ33_base) == (3, 3)
            
            # Create more operators directly and test
            op23 = TransitionOperator(:Σ, 4, 2, 3)
            op41 = TransitionOperator(:Σ, 4, 4, 1)
            @test transition_levels(op23) == (2, 3)
            @test transition_levels(op41) == (4, 1)
        end

        @testset "is_transition_op" begin
            @test is_transition_op(σ12_base)
            @test is_transition_op(σ33_base)
            
            # Boson operator
            a_base = QuantumAlgebra.BosonDestroy(:a)
            @test !is_transition_op(a_base)
        end
    end

    # =========================================================================
    # Adjoint (Hermitian Conjugate)
    # =========================================================================
    @testset "Adjoint" begin
        
        @testset "Basic adjoint" begin
            Σ = nlevel_ops(2, :σ)
            
            # |i⟩⟨j|† = |j⟩⟨i|
            @test Σ[1, 2]' == Σ[2, 1]
            @test Σ[2, 1]' == Σ[1, 2]
            
            # Diagonal operators are Hermitian
            @test Σ[1, 1]' == Σ[1, 1]
            @test Σ[2, 2]' == Σ[2, 2]
        end

        @testset "3-level adjoint" begin
            Σ = nlevel_ops(3, :Σ)
            
            for i in 1:3, j in 1:3
                @test Σ[i, j]' == Σ[j, i]
            end
        end

        @testset "Adjoint with indices" begin
            σ12_i = QuExpr(TransitionOperator(:σ, 2, 1, 2, :i))
            σ21_i = QuExpr(TransitionOperator(:σ, 2, 2, 1, :i))
            
            @test σ12_i' == σ21_i
        end

        @testset "Double adjoint" begin
            Σ = nlevel_ops(2, :σ)
            @test Σ[1, 2]'' == Σ[1, 2]
        end
    end

    # =========================================================================
    # Product Rule (Contraction)
    # =========================================================================
    @testset "Product rule" begin
        
        @testset "2-level products" begin
            Σ = nlevel_ops(2, :σ)
            
            # |i⟩⟨j| × |k⟩⟨l| = δ_jk |i⟩⟨l|
            
            # |1⟩⟨2| × |2⟩⟨1| = δ_22 |1⟩⟨1| = |1⟩⟨1| = σ11
            @test normal_form(Σ[1, 2] * Σ[2, 1]) == Σ[1, 1]
            
            # |2⟩⟨1| × |1⟩⟨2| = δ_11 |2⟩⟨2| = |2⟩⟨2| = σ22
            @test normal_form(Σ[2, 1] * Σ[1, 2]) == Σ[2, 2]
            
            # |1⟩⟨2| × |1⟩⟨2| = δ_21 |1⟩⟨2| = 0
            @test iszero(normal_form(Σ[1, 2] * Σ[1, 2]))
            
            # |1⟩⟨1| × |1⟩⟨1| = δ_11 |1⟩⟨1| = |1⟩⟨1|
            @test normal_form(Σ[1, 1] * Σ[1, 1]) == Σ[1, 1]
            
            # |2⟩⟨2| × |2⟩⟨2| = |2⟩⟨2|
            @test normal_form(Σ[2, 2] * Σ[2, 2]) == Σ[2, 2]
            
            # Mixed: |1⟩⟨1| × |2⟩⟨2| = δ_12 ... = 0
            @test iszero(normal_form(Σ[1, 1] * Σ[2, 2]))
        end

        @testset "3-level products" begin
            Σ = nlevel_ops(3, :Σ)
            
            # |1⟩⟨2| × |2⟩⟨3| = |1⟩⟨3|
            @test normal_form(Σ[1, 2] * Σ[2, 3]) == Σ[1, 3]
            
            # |1⟩⟨2| × |3⟩⟨1| = 0 (j=2, k=3)
            @test iszero(normal_form(Σ[1, 2] * Σ[3, 1]))
            
            # Chain: |1⟩⟨2| × |2⟩⟨3| × |3⟩⟨1| = |1⟩⟨3| × |3⟩⟨1| = |1⟩⟨1|
            @test normal_form(Σ[1, 2] * Σ[2, 3] * Σ[3, 1]) == Σ[1, 1]
        end

        @testset "Products with different sites" begin
            Σ_i = nlevel_ops(2, :σ, :i)
            Σ_j = nlevel_ops(2, :σ, :j)
            
            # Different sites don't contract - they just form a product
            prod = Σ_i[1, 2] * Σ_j[2, 1]
            @test normal_form(prod) == prod  # No simplification
        end

        @testset "Completeness relation" begin
            Σ = nlevel_ops(2, :σ)
            sum_diag = Σ[1, 1] + Σ[2, 2]
            
            # σ11 × σ12 = |1⟩⟨1| × |1⟩⟨2| = δ_11 |1⟩⟨2| = σ12
            # σ22 × σ12 = |2⟩⟨2| × |1⟩⟨2| = δ_21 |2⟩⟨2| = 0
            @test normal_form(sum_diag * Σ[1, 2]) == Σ[1, 2]
            @test normal_form(Σ[1, 2] * sum_diag) == Σ[1, 2]
        end

        @testset "Triple products" begin
            Σ = nlevel_ops(3, :Σ)
            
            # |1⟩⟨2| × |2⟩⟨2| × |2⟩⟨3| = |1⟩⟨2| × |2⟩⟨3| = |1⟩⟨3|
            @test normal_form(Σ[1, 2] * Σ[2, 2] * Σ[2, 3]) == Σ[1, 3]
            
            # |1⟩⟨2| × |2⟩⟨3| × |3⟩⟨3| = |1⟩⟨3| × |3⟩⟨3| = |1⟩⟨3|
            @test normal_form(Σ[1, 2] * Σ[2, 3] * Σ[3, 3]) == Σ[1, 3]
        end
    end

    # =========================================================================
    # Commutation Relations
    # =========================================================================
    @testset "Commutation relations" begin
        
        @testset "2-level commutators" begin
            Σ = nlevel_ops(2, :σ)
            
            # [|i⟩⟨j|, |k⟩⟨l|] = δ_jk |i⟩⟨l| - δ_il |k⟩⟨j|
            
            # [σ12, σ21] = δ_22 σ11 - δ_11 σ22 = σ11 - σ22
            @test normal_form(comm(Σ[1, 2], Σ[2, 1])) == Σ[1, 1] - Σ[2, 2]
            
            # [σ21, σ12] = -(σ11 - σ22) = σ22 - σ11
            @test normal_form(comm(Σ[2, 1], Σ[1, 2])) == Σ[2, 2] - Σ[1, 1]
            
            # [σ12, σ12] = 0
            @test iszero(normal_form(comm(Σ[1, 2], Σ[1, 2])))
            
            # [|1⟩⟨1|, |1⟩⟨2|] = δ_11 |1⟩⟨2| - δ_12 |1⟩⟨1| = σ12 - 0 = σ12
            @test normal_form(comm(Σ[1, 1], Σ[1, 2])) == Σ[1, 2]
            
            # [|2⟩⟨2|, |1⟩⟨2|] = δ_21 |2⟩⟨2| - δ_22 |1⟩⟨2| = 0 - σ12 = -σ12
            @test normal_form(comm(Σ[2, 2], Σ[1, 2])) == -Σ[1, 2]
        end

        @testset "Self-commutation is zero" begin
            Σ = nlevel_ops(3, :Σ)
            
            for i in 1:3, j in 1:3
                @test iszero(normal_form(comm(Σ[i, j], Σ[i, j])))
            end
        end

        @testset "Antisymmetry" begin
            Σ = nlevel_ops(3, :Σ)
            
            @test normal_form(comm(Σ[1, 2], Σ[2, 1])) == -normal_form(comm(Σ[2, 1], Σ[1, 2]))
            @test normal_form(comm(Σ[1, 3], Σ[3, 2])) == -normal_form(comm(Σ[3, 2], Σ[1, 3]))
        end

        @testset "3-level commutators" begin
            Σ = nlevel_ops(3, :Σ)
            
            # [|1⟩⟨2|, |2⟩⟨3|] = δ_22 |1⟩⟨3| - δ_13 |2⟩⟨2| = σ13 - 0 = σ13
            @test normal_form(comm(Σ[1, 2], Σ[2, 3])) == Σ[1, 3]
            
            # [|1⟩⟨2|, |3⟩⟨1|] = δ_23 |1⟩⟨1| - δ_11 |3⟩⟨2| = 0 - σ32 = -σ32
            @test normal_form(comm(Σ[1, 2], Σ[3, 1])) == -Σ[3, 2]
        end

        @testset "Commutation with bosons" begin
            Σ = nlevel_ops(2, :σ)
            
            # Transition operators commute with bosons
            @test iszero(normal_form(comm(Σ[1, 2], a())))
            @test iszero(normal_form(comm(Σ[1, 2], a'())))
            @test iszero(normal_form(comm(Σ[2, 1], a(1))))
        end

        @testset "Commutation with fermions" begin
            Σ = nlevel_ops(2, :σ)
            
            # Transition operators commute with fermions
            @test iszero(normal_form(comm(Σ[1, 2], f())))
            @test iszero(normal_form(comm(Σ[1, 2], f'())))
        end
    end

    # =========================================================================
    # Vacuum Expectation Values
    # =========================================================================
    @testset "Vacuum expectation values" begin
        # Convention: vacuum is |N⟩ (lowest energy state / ground state)
        
        @testset "2-level vacuum" begin
            Σ = nlevel_ops(2, :σ)
            
            # σ22 = |2⟩⟨2| is the ground state projector → eigenvalue 1
            @test vacExpVal(Σ[2, 2]) == QuExpr(1)
            
            # σ11 = |1⟩⟨1| → eigenvalue 0 (not in ground state)
            @test iszero(vacExpVal(Σ[1, 1]))
            
            # Off-diagonal: ⟨2|σ12|2⟩ = ⟨2|1⟩⟨2|2⟩ = 0 × 1 = 0
            @test iszero(vacExpVal(Σ[1, 2]))
            @test iszero(vacExpVal(Σ[2, 1]))
        end

        @testset "3-level vacuum" begin
            Σ = nlevel_ops(3, :Σ)
            
            # σ33 = |3⟩⟨3| is the ground state projector
            @test vacExpVal(Σ[3, 3]) == QuExpr(1)
            
            # All other diagonal projectors give 0
            @test iszero(vacExpVal(Σ[1, 1]))
            @test iszero(vacExpVal(Σ[2, 2]))
            
            # All off-diagonal give 0
            for i in 1:3, j in 1:3
                if i != j
                    @test iszero(vacExpVal(Σ[i, j]))
                end
            end
        end

        @testset "Products of transition operators" begin
            Σ = nlevel_ops(2, :σ)
            
            # σ12 × σ21 = σ11, ⟨σ11⟩ = 0
            @test iszero(vacExpVal(Σ[1, 2] * Σ[2, 1]))
            
            # σ21 × σ12 = σ22, ⟨σ22⟩ = 1
            @test vacExpVal(Σ[2, 1] * Σ[1, 2]) == QuExpr(1)
            
            # σ22 × σ22 = σ22, ⟨σ22⟩ = 1
            @test vacExpVal(Σ[2, 2] * Σ[2, 2]) == QuExpr(1)
        end

        @testset "Avac and vacA" begin
            Σ = nlevel_ops(2, :σ)
            
            # Avac: operator acting on vacuum from left
            # |i⟩⟨j| |vac⟩ = |i⟩⟨j|2⟩ = δ_j2 |i⟩
            
            # σ12 |2⟩ = δ_22 |1⟩ = |1⟩ → represented as σ12 (keeps ket |1⟩)
            @test Avac(Σ[1, 2]) == Σ[1, 2]
            
            # σ11 |2⟩ = δ_12 |1⟩ = 0
            @test iszero(Avac(Σ[1, 1]))
            
            # σ22 |2⟩ = δ_22 |2⟩ = |2⟩ → but ⟨2|2⟩ = 1, so Avac(σ22) = 1
            @test Avac(Σ[2, 2]) == QuExpr(1)
            
            # σ21 |2⟩ = δ_12 |2⟩ = 0
            @test iszero(Avac(Σ[2, 1]))
            
            # vacA: vacuum bra acting on operator from right
            # ⟨vac| |i⟩⟨j| = ⟨2|i⟩ ⟨j| = δ_2i ⟨j|
            
            # ⟨2| σ21 = ⟨2|2⟩⟨1| = ⟨1| → represented as σ21
            @test vacA(Σ[2, 1]) == Σ[2, 1]
            
            # ⟨2| σ11 = ⟨2|1⟩⟨1| = 0
            @test iszero(vacA(Σ[1, 1]))
            
            # ⟨2| σ22 = ⟨2|2⟩⟨2| = ⟨2| → but ⟨2|2⟩ = 1, so vacA(σ22) = 1
            @test vacA(Σ[2, 2]) == QuExpr(1)
            
            # ⟨2| σ12 = ⟨2|1⟩⟨2| = 0
            @test iszero(vacA(Σ[1, 2]))
        end

        @testset "3-level Avac and vacA" begin
            Σ = nlevel_ops(3, :Σ)
            
            # Only operators with j=3 survive Avac (acting on |3⟩)
            @test Avac(Σ[1, 3]) == Σ[1, 3]
            @test Avac(Σ[2, 3]) == Σ[2, 3]
            # σ33 gives eigenvalue 1
            @test Avac(Σ[3, 3]) == QuExpr(1)
            
            # All others annihilate vacuum
            for j in 1:2
                for i in 1:3
                    @test iszero(Avac(Σ[i, j]))
                end
            end
            
            # Only operators with i=3 survive vacA (acting on ⟨3|)
            @test vacA(Σ[3, 1]) == Σ[3, 1]
            @test vacA(Σ[3, 2]) == Σ[3, 2]
            # σ33 gives eigenvalue 1
            @test vacA(Σ[3, 3]) == QuExpr(1)
            
            # All others annihilate vacuum bra
            for i in 1:2
                for j in 1:3
                    @test iszero(vacA(Σ[i, j]))
                end
            end
        end

        @testset "Mixed with bosons" begin
            Σ = nlevel_ops(2, :σ)
            
            # Bosons annihilate vacuum, so anything with a()|0⟩ = 0
            @test iszero(vacExpVal(a() * Σ[2, 2]))
            @test iszero(vacExpVal(Σ[2, 2] * a()))
            
            # a†|0⟩ = |1⟩, so ⟨0|a†|0⟩ = 0
            @test iszero(vacExpVal(a'() * Σ[2, 2]))
        end
    end

    # =========================================================================
    # String Output
    # =========================================================================
    @testset "String output" begin
        
        @testset "Superscript notation" begin
            Σ = nlevel_ops(2, :σ)
            s = string(Σ[1, 2])
            @test occursin("σ", s)
            @test occursin("¹", s)
            @test occursin("²", s)
        end

        @testset "Multi-digit indices" begin
            Σ = nlevel_ops(12, :Σ)
            s = string(Σ[10, 11])
            @test occursin("Σ", s)
            @test occursin("¹⁰", s)
            @test occursin("¹¹", s)
        end

        @testset "With site indices" begin
            σ12_i = QuExpr(TransitionOperator(:σ, 2, 1, 2, :i))
            s = string(σ12_i)
            @test occursin("σ", s)
            @test occursin("i", s)
        end

        @testset "Products" begin
            Σ = nlevel_ops(3, :Σ)
            s = string(Σ[1, 2] * Σ[2, 3])
            @test occursin("Σ", s)
        end
    end

    # =========================================================================
    # LaTeX Output
    # =========================================================================
    @testset "LaTeX output" begin
        
        @testset "Basic LaTeX" begin
            Σ = nlevel_ops(2, :σ)
            l = latex(Σ[1, 2])
            @test occursin("\\sigma", l) || occursin("σ", l)
            @test occursin("^{12}", l)
        end

        @testset "With site indices" begin
            σ12_i = QuExpr(TransitionOperator(:σ, 2, 1, 2, :i))
            l = latex(σ12_i)
            @test occursin("i", l)
        end
    end

    # =========================================================================
    # Mixed Operations with Other Operator Types
    # =========================================================================
    @testset "Mixed operations" begin
        Σ = nlevel_ops(2, :σtrans)  # Use different name to avoid conflict with TLS σ
        
        @testset "With bosons" begin
            # Transition operators commute with bosons
            expr = a() * Σ[1, 2]
            @test normal_form(expr) == normal_form(Σ[1, 2] * a())
            
            # Products
            mixed = a'() * Σ[1, 2] * a()
            @test mixed isa QuExpr
        end

        @testset "With fermions" begin
            expr = f() * Σ[1, 2]
            @test normal_form(expr) == normal_form(Σ[1, 2] * f())
        end

        @testset "With parameters" begin
            g = Pc"g"
            ω = Pr"ω"
            
            expr = g * Σ[1, 2]
            @test expr isa QuExpr
            
            # Parameters factor through
            @test normal_form(g * Σ[1, 2] * Σ[2, 1]) == g * Σ[1, 1]
        end

        @testset "With sums" begin
            Σ_i = nlevel_ops(2, :σ, :i)
            
            s = ∑(:i, Σ_i[1, 2])
            @test s isa QuExpr
            
            # Sum of operators
            H = ∑(:i, Pr"ω_i" * Σ_i[2, 2])
            @test H isa QuExpr
        end
    end

    # =========================================================================
    # ExpVal and Corr
    # =========================================================================
    @testset "ExpVal and Corr" begin
        Σ = nlevel_ops(2, :σ)
        
        @testset "Basic expval" begin
            @test expval(Σ[1, 2]) isa QuExpr
            ev = expval(Σ[1, 2])
            @test normal_form(ev) == ev
        end

        @testset "Basic corr" begin
            @test corr(Σ[1, 2]) isa QuExpr
            @test corr(3 + Σ[1, 2]) == 3 + corr(Σ[1, 2])
        end

        @testset "expval_as_corrs" begin
            Σ_i = nlevel_ops(2, :σ, :i)
            Σ_j = nlevel_ops(2, :σ, :j)
            
            # Product with different sites
            result = expval_as_corrs(Σ_i[1, 2] * Σ_j[2, 1])
            @test result == corr(Σ_i[1, 2] * Σ_j[2, 1]) + corr(Σ_i[1, 2]) * corr(Σ_j[2, 1])
        end
    end

    # =========================================================================
    # Exponents
    # =========================================================================
    @testset "Exponents" begin
        Σ = nlevel_ops(2, :σ)
        
        @testset "Basic exponents" begin
            for op in [Σ[1, 1], Σ[2, 2], Σ[1, 2]]
                @test isone(op^0)
                @test op^1 == op
                @test op^2 == op * op
            end
        end

        @testset "Projector idempotence" begin
            # Diagonal operators are projectors: P² = P
            @test normal_form(Σ[1, 1]^2) == Σ[1, 1]
            @test normal_form(Σ[2, 2]^2) == Σ[2, 2]
        end

        @testset "Off-diagonal powers" begin
            # σ12² = |1⟩⟨2| × |1⟩⟨2| = δ_21 |1⟩⟨2| = 0
            @test iszero(normal_form(Σ[1, 2]^2))
            @test iszero(normal_form(Σ[2, 1]^2))
        end

        @testset "Negative exponents" begin
            @test_throws ArgumentError Σ[1, 2]^-1
        end
    end

    # =========================================================================
    # is_normal_form Tests
    # =========================================================================
    @testset "is_normal_form" begin
        Σ = nlevel_ops(2, :σ)
        
        # Normal form should be idempotent
        function check_idempotent(x)
            xnf = normal_form(x)
            return xnf == normal_form(xnf)
        end
        
        @testset "Single operators" begin
            for i in 1:2, j in 1:2
                @test check_idempotent(Σ[i, j])
            end
        end

        @testset "Products" begin
            @test check_idempotent(Σ[1, 2] * Σ[2, 1])
        end

        @testset "Sums" begin
            Σ_i = nlevel_ops(2, :σ, :i)
            @test check_idempotent(∑(:i, Σ_i[1, 2]))
        end
    end

    # =========================================================================
    # Edge Cases
    # =========================================================================
    @testset "Edge cases" begin
        Σ = nlevel_ops(2, :σ)
        
        # Zero/identity
        @test iszero(Σ[1, 2] - Σ[1, 2])
        @test 1 * Σ[1, 2] == Σ[1, 2]
        @test isone(Σ[1, 2]^0)
        
        # Complex coefficients
        @test ((1 + 2im) * Σ[1, 2])' == (1 - 2im) * Σ[2, 1]
        
        # Rational coefficients
        @test normal_form((1//2) * Σ[1, 2] * (2) * Σ[2, 1]) == Σ[1, 1]
        
        # Large N
        for N in [5, 8, 10]
            Σ_N = nlevel_ops(N, :Σ)
            @test size(Σ_N) == (N, N)
            @test normal_form(Σ_N[1, 2] * Σ_N[2, 1]) == Σ_N[1, 1]
            @test normal_form(Σ_N[N-1, N] * Σ_N[N, N-1]) == Σ_N[N-1, N-1]
        end
    end

end
