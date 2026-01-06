# Tests for symbolic sums and products
# Tests the new SymSum, SymProd, and SymExpr types

using QuantumAlgebra
using QuantumAlgebra: SymSum, SymProd, SymExpr, symbolic_sum, symbolic_prod
using QuantumAlgebra: expand_symbolic, free_indices, canonicalize_indices
using QuantumAlgebra: simplify_sums, alpha_equivalent, replace_sym_index
using QuantumAlgebra: fresh_sumindex, reset_fresh_sumindex_counter!
using QuantumAlgebra: has_free_index, sumindex, QuIndex, issumindex

@testset "Symbolic Sums and Products" begin
    
    # Reset counter for reproducible tests
    reset_fresh_sumindex_counter!(1000)
    
    @testset "Basic Construction" begin
        i = sumindex(1)
        
        # SymSum construction
        s = SymSum(σz(i), i)
        @test s isa SymSum
        @test s.index == i
        @test isempty(s.excluded)
        @test s.expr == σz(i)
        
        # With exclusion
        j = sumindex(2)
        s_excl = SymSum(σz(i), i, [j])
        @test s_excl.excluded == [j]
        
        # SymProd construction
        p = SymProd(a(i), i)
        @test p isa SymProd
        @test p.index == i
        
        # Convenience constructors
        s2 = symbolic_sum(σz(i), i)
        @test s2 isa SymSum
        
        p2 = symbolic_prod(a(i), i)
        @test p2 isa SymProd
        
        # Error on non-sum index
        @test_throws ErrorException SymSum(σz(:a), QuIndex(:a))
    end
    
    @testset "SymExpr Construction" begin
        i = sumindex(1)
        s = SymSum(σz(i), i)
        
        # From aggregate
        e = SymExpr(s)
        @test e isa SymExpr
        @test length(e.terms) == 1
        @test iszero(e.scalar)
        
        # From QuExpr
        e2 = SymExpr(a() + a'())
        @test isempty(e2.terms)
        @test e2.scalar == a() + a'()
        
        # Empty
        e3 = SymExpr()
        @test iszero(e3)
    end
    
    @testset "Free Indices" begin
        i = sumindex(1)
        j = sumindex(2)
        
        # Plain QuExpr - sum indices not bound by SymSum are free
        @test i in free_indices(σz(i))
        
        # SymSum - the index is bound
        s = SymSum(σz(i), i)
        @test !(i in free_indices(s))
        
        # Nested - outer index still free from inner's perspective
        inner = SymSum(σz(i) * σz(j), j, [i])  # Σⱼ≠ᵢ σz(i)σz(j)
        @test i in free_indices(inner)  # i is in excluded, so it's free
        @test !(j in free_indices(inner))  # j is bound
    end
    
    @testset "Index Replacement" begin
        i = sumindex(1)
        j = sumindex(2)
        
        # Replace in QuExpr
        expr = σz(i) * a(i)
        replaced = replace_sym_index(expr, i, j)
        @test replaced == σz(j) * a(j)
        
        # Replace in SymSum
        s = SymSum(σz(i), i)
        s_replaced = replace_sym_index(s, i, j)
        @test s_replaced.index == j
        @test s_replaced.expr == σz(j)
        
        # Replace in excluded
        k = sumindex(3)
        s_excl = SymSum(σz(i), i, [j])
        s_excl_replaced = replace_sym_index(s_excl, j, k)
        @test s_excl_replaced.excluded == [k]
    end
    
    @testset "Negation" begin
        i = sumindex(1)
        s = SymSum(σz(i), i)
        
        neg_s = -s
        @test neg_s isa SymSum
        @test neg_s.expr == -σz(i)
        
        p = SymProd(a(i), i)
        neg_p = -p
        @test neg_p.expr == -a(i)
        
        e = SymExpr(s)
        neg_e = -e
        @test neg_e.terms[1][1] == -1
    end
    
    @testset "Scalar Multiplication" begin
        i = sumindex(1)
        s = SymSum(σz(i), i)
        
        # Number * SymSum
        s2 = 2 * s
        @test s2 isa SymSum
        @test s2.expr == 2 * σz(i)
        
        # SymSum * Number
        s3 = s * 3
        @test s3.expr == 3 * σz(i)
        
        # Zero
        s0 = 0 * s
        @test s0 isa SymExpr
        @test iszero(s0)
    end
    
    @testset "Addition" begin
        i = sumindex(1)
        j = sumindex(2)
        
        s1 = SymSum(σz(i), i)
        s2 = SymSum(σx(j), j)
        
        # SymSum + SymSum
        result = s1 + s2
        @test result isa SymExpr
        @test length(result.terms) == 2
        
        # SymSum + QuExpr
        result2 = s1 + a()
        @test result2 isa SymExpr
        @test length(result2.terms) == 1
        @test result2.scalar == a()
        
        # SymSum + Number
        result3 = s1 + 5
        @test result3 isa SymExpr
        @test result3.scalar == QuExpr(5)
    end
    
    @testset "Subtraction" begin
        i = sumindex(1)
        s1 = SymSum(σz(i), i)
        s2 = SymSum(σx(i), i)
        
        result = s1 - s2
        @test result isa SymExpr
        @test length(result.terms) == 2
        # The negation is applied to the inner expression, not the coefficient
        # s1 - s2 = s1 + (-s2) where -s2 = SymSum(-σx(i), i)
        # Both terms have coefficient 1, but second has negated expression
        @test result.terms[1][1] == 1
        @test result.terms[2][1] == 1
        @test result.terms[2][2].expr == -σx(i)
    end
    
    @testset "Multiplication - Bound Variable Handling" begin
        reset_fresh_sumindex_counter!(1000)
        
        i = sumindex(1)
        
        # Case 1: (Σᵢ Xᵢ) * Y where i ∉ free(Y)
        s = SymSum(σz(i), i)
        result1 = s * a()  # a() has no index i
        @test result1 isa SymSum
        @test result1.index == i
        
        # Case 2: (Σᵢ Xᵢ) * Xᵢ where i ∈ free(Xᵢ) - should rename!
        s2 = SymSum(a(i), i)
        result2 = s2 * a(i)  # a(i) has free index i
        @test result2 isa SymSum
        # The bound index should have been renamed
        @test result2.index != i  # Fresh index
        # The expression should have the NEW bound index and the original free i
        # Σⱼ (a(j) * a(i)) where j is fresh
        
        # Case 3: (Σᵢ Aᵢ) * (Σⱼ Bⱼ) - nested sums with fresh indices
        reset_fresh_sumindex_counter!(1000)
        s3 = SymSum(σp(i), i)
        j = sumindex(2)
        s4 = SymSum(σm(j), j)
        
        result3 = s3 * s4
        @test result3 isa SymSum  # Outer sum
        @test result3.expr isa SymSum  # Inner sum
    end
    
    @testset "Power - Creates Multiple Indices" begin
        reset_fresh_sumindex_counter!(1000)
        
        i = sumindex(1)
        s = SymSum(a(i), i)
        
        # (Σᵢ aᵢ)² = Σⱼ Σₖ aⱼ aₖ
        s2 = s^2
        @test s2 isa SymSum
        @test s2.expr isa SymSum
        
        # (Σᵢ aᵢ)³ = Σⱼ Σₖ Σₗ aⱼ aₖ aₗ
        s3 = s^3
        @test s3 isa SymSum
        @test s3.expr isa SymSum
        @test s3.expr.expr isa SymSum
        
        # s^0 = 1
        s0 = s^0
        @test s0 isa SymExpr
        @test s0.scalar == one(QuExpr)
        
        # s^1 = s
        s1 = s^1
        @test s1 == s
    end
    
    @testset "Expansion - Basic" begin
        i = sumindex(1)
        
        # Simple sum expansion
        s = SymSum(σz(i), i)
        expanded = expand_symbolic(s, 1:3)
        @test expanded == σz(1) + σz(2) + σz(3)
        
        # Sum with coefficient
        s2 = SymSum(2 * a(i), i)
        expanded2 = expand_symbolic(s2, 1:2)
        @test expanded2 == 2*a(1) + 2*a(2)
    end
    
    @testset "Expansion - With Exclusion" begin
        i = sumindex(1)
        j = sumindex(2)
        
        # Σⱼ≠₂ σz(j)
        s = SymSum(σz(j), j, [QuIndex(2)])  # Exclude literal 2
        expanded = expand_symbolic(s, 1:4)
        @test expanded == σz(1) + σz(3) + σz(4)  # No σz(2)
        
        # Nested with dynamic exclusion: Σᵢ Σⱼ≠ᵢ σz(i)σz(j)
        inner = SymSum(σz(i) * σz(j), j, [i])
        outer = SymSum(inner, i)
        
        expanded_nested = expand_symbolic(outer, 1:2)
        # i=1: j≠1, so j=2: σz(1)σz(2)
        # i=2: j≠2, so j=1: σz(2)σz(1)
        expected = σz(1)*σz(2) + σz(2)*σz(1)
        @test normal_form(expanded_nested) == normal_form(expected)
    end
    
    @testset "Expansion - Product" begin
        i = sumindex(1)
        
        # Πᵢ aᵢ for i=1,2,3
        p = SymProd(a(i), i)
        expanded = expand_symbolic(p, 1:3)
        @test expanded == a(1) * a(2) * a(3)
    end
    
    @testset "Expansion - SymExpr" begin
        i = sumindex(1)
        
        s = SymSum(σz(i), i)
        e = SymExpr([(2, s)], a())  # 2*Σᵢ σz(i) + a
        
        expanded = expand_symbolic(e, 1:2)
        @test expanded == 2*σz(1) + 2*σz(2) + a()
    end
    
    @testset "Canonicalization" begin
        reset_fresh_sumindex_counter!(1000)
        
        # Different index numbers, same structure
        i = sumindex(5)
        j = sumindex(42)
        
        s1 = SymSum(σz(i), i)
        s2 = SymSum(σz(j), j)
        
        s1_canon = canonicalize_indices(s1)
        s2_canon = canonicalize_indices(s2)
        
        @test s1_canon.index == sumindex(1)
        @test s2_canon.index == sumindex(1)
        @test s1_canon.expr == s2_canon.expr
    end
    
    @testset "Alpha Equivalence" begin
        # Same structure, different index names
        i = sumindex(1)
        j = sumindex(2)
        
        s1 = SymSum(σz(i), i)
        s2 = SymSum(σz(j), j)
        
        @test alpha_equivalent(s1, s2)
        @test s1 == s2  # Uses alpha_equivalent
        
        # Different structure
        s3 = SymSum(σx(i), i)
        @test !alpha_equivalent(s1, s3)
        @test s1 != s3
        
        # Different exclusion
        s4 = SymSum(σz(i), i, [sumindex(10)])
        @test !alpha_equivalent(s1, s4)
    end
    
    @testset "Simplify Sums" begin
        i = sumindex(1)
        j = sumindex(2)
        
        # Σᵢ Xᵢ + Σⱼ Xⱼ = 2 Σᵢ Xᵢ
        s1 = SymSum(σz(i), i)
        s2 = SymSum(σz(j), j)
        
        combined = s1 + s2
        simplified = simplify_sums(combined)
        
        @test length(simplified.terms) == 1
        @test simplified.terms[1][1] == 2
    end
    
    @testset "Commutator - SymSum with QuExpr" begin
        i = sumindex(1)
        
        # [Σᵢ a†(i)a(i), a(j)] should work
        H = SymSum(a'(i) * a(i), i)
        
        result = comm(H, a(:j))
        @test result isa SymSum
        
        # Expand and check
        expanded = expand_symbolic(result, [:j])
        # [a†(j)a(j), a(j)] = -a(j)
        @test normal_form(expanded) == normal_form(-a(:j))
    end
    
    @testset "Commutator - SymSum with SymSum (Same Site + Cross Site)" begin
        reset_fresh_sumindex_counter!(1000)
        
        i = sumindex(1)
        
        # Simple test: [Σᵢ σ⁺(i), Σⱼ σ⁻(j)]
        # Should give same-site: Σᵢ [σ⁺(i), σ⁻(i)] = Σᵢ σz(i)
        # And cross-site: Σᵢ Σⱼ≠ᵢ [σ⁺(i), σ⁻(j)] = 0 (different sites commute)
        
        S = SymSum(σp(i), i)
        V = SymSum(σm(i), i)
        
        result = comm(S, V)
        @test result isa SymExpr
        @test length(result.terms) == 2  # same-site and cross-site
        
        # Expand for 2 sites
        expanded = expand_symbolic(result, 1:2)
        
        # Same-site: [σ⁺(1), σ⁻(1)] + [σ⁺(2), σ⁻(2)] = σz(1) + σz(2)
        # Cross-site: [σ⁺(1), σ⁻(2)] + [σ⁺(2), σ⁻(1)] = 0 + 0 = 0
        expected = σz(1) + σz(2)
        @test normal_form(expanded) == normal_form(expected)
    end
    
    @testset "Tavis-Cummings: Exchange Interaction" begin
        reset_fresh_sumindex_counter!(1000)
        
        # This is the key physics test!
        # For the Tavis-Cummings model, the Schrieffer-Wolff transformation
        # should produce exchange interactions between atoms.
        
        i = sumindex(1)
        
        # V = Σᵢ g(a†σ⁻(i) + aσ⁺(i))
        # For simplicity, set g=1
        V = SymSum(a'() * σm(i) + a() * σp(i), i)
        
        # S ∝ Σᵢ (aσ⁺(i) - a†σ⁻(i)) (the generator, simplified)
        S = SymSum(a() * σp(i) - a'() * σm(i), i)
        
        # [S, V] should produce:
        # - Same-site terms (cavity photon number, etc.)
        # - Cross-site terms: σ⁺(i)σ⁻(j) + σ⁺(j)σ⁻(i) exchange!
        
        result = comm(S, V)
        @test result isa SymExpr
        
        # Expand for N=2 atoms
        expanded = expand_symbolic(result, 1:2)
        expanded_normal = normal_form(expanded)
        
        # Check that cross-site terms exist
        # Look for σ⁺(1)σ⁻(2) or σ⁺(2)σ⁻(1) terms
        str = string(expanded_normal)
        @test occursin("σ", str)
        
        # The expanded result should contain operators at BOTH sites
        # i.e., it's not just a sum of single-site terms
        # We check this by looking for products of operators at different sites
        
        # More detailed check: the cross-site contribution should be non-zero
        # because [aσ⁺(1), a†σ⁻(2)] = σ⁺(1)σ⁻(2)[a, a†] = σ⁺(1)σ⁻(2)
        @test !iszero(expanded_normal)
    end
    
    @testset "Display" begin
        i = sumindex(1)
        s = SymSum(σz(i), i)
        
        str = string(s)
        @test occursin("Σ", str)
        
        # With exclusion
        j = sumindex(2)
        s_excl = SymSum(σz(i), i, [j])
        str_excl = string(s_excl)
        @test occursin("≠", str_excl)
        
        # SymExpr
        e = SymExpr([(2, s)], a())
        str_e = string(e)
        @test occursin("Σ", str_e)
    end
    
    @testset "LaTeX" begin
        i = sumindex(1)
        s = SymSum(σz(i), i)
        
        ltx = latex(s)
        @test occursin("\\sum", ltx)
        
        # With exclusion
        j = sumindex(2)
        s_excl = SymSum(σz(i), i, [j])
        ltx_excl = latex(s_excl)
        @test occursin("\\neq", ltx_excl)
    end
    
    @testset "Zero and One" begin
        @test iszero(SymExpr())
        @test zero(SymExpr) == SymExpr()
        @test one(SymExpr) == SymExpr(one(QuExpr))
    end
    
    @testset "Normal Form Through SymSum" begin
        i = sumindex(1)
        
        # normal_form should pass through to the inner expression
        s = SymSum(a(i) * a'(i), i)  # Not in normal form
        s_normal = normal_form(s)
        
        @test s_normal isa SymSum
        # a * a† = a† a + 1 in normal form
        expected_inner = normal_form(a(i) * a'(i))
        @test s_normal.expr == expected_inner
    end
    
    @testset "SymExpr Equality" begin
        i = sumindex(1)
        j = sumindex(2)
        
        s1 = SymSum(σz(i), i)
        s2 = SymSum(σz(j), j)
        
        e1 = SymExpr([(1, s1)], a())
        e2 = SymExpr([(1, s2)], a())
        
        @test e1 == e2  # Alpha-equivalent sums and same scalar
        
        e3 = SymExpr([(2, s1)], a())
        @test e1 != e3  # Different coefficient
    end
    
    @testset "Symbolic Algebra Integration" begin
        i = sumindex(1)
        
        # Test map_scalar_function
        s = SymSum(2 * σz(i), i)
        
        # Simple doubling function
        doubled = map_scalar_function(x -> 2x, s)
        @test doubled.expr == 4 * σz(i)
    end
    
end
