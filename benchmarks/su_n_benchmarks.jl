# =============================================================================
# SU(N) Performance Benchmarks
# =============================================================================
#
# This script benchmarks the SU(N) Lie algebra implementation against
# the existing TLS (SU(2) Pauli) implementation to identify performance
# characteristics and potential bottlenecks.
#
# Prerequisites:
#   import Pkg; Pkg.add("BenchmarkTools")
#
# Run with: julia --project=. benchmarks/su_n_benchmarks.jl
#
# =============================================================================

using QuantumAlgebra
using BenchmarkTools
using Printf

# Disable auto normal form for controlled benchmarking
QuantumAlgebra.auto_normal_form(false)

println("="^70)
println("         SU(N) Performance Benchmarks")
println("="^70)
println()

# =============================================================================
# 1. BaseOperator Creation
# =============================================================================

println("1. Operator Creation Time")
println("-"^50)

print("   Boson a():              ")
t_boson = @belapsed a()
@printf("%8.2f ns\n", t_boson * 1e9)

print("   TLS σx():               ")
t_tls = @belapsed σx()
@printf("%8.2f ns\n", t_tls * 1e9)

print("   SU(2) generator:        ")
t_su2 = @belapsed su_generator(2, :T, 1)
@printf("%8.2f ns\n", t_su2 * 1e9)

print("   SU(3) generator:        ")
t_su3 = @belapsed su_generator(3, :λ, 1)
@printf("%8.2f ns\n", t_su3 * 1e9)

print("   SU(8) generator:        ")
t_su8 = @belapsed su_generator(8, :G, 1)
@printf("%8.2f ns\n", t_su8 * 1e9)

println()

# =============================================================================
# 2. Simple Commutator [A, B]
# =============================================================================

println("2. Simple Commutator [A, B]")
println("-"^50)

# TLS
σ_x, σ_y, σ_z = σx(), σy(), σz()
print("   [σx, σy] (TLS):         ")
t_comm_tls = @belapsed normal_form(comm($σ_x, $σ_y))
@printf("%8.2f μs\n", t_comm_tls * 1e6)

# SU(2) generators
T2 = su_generators(2, :T)
print("   [T¹, T²] (SU(2)):       ")
t_comm_su2 = @belapsed normal_form(comm($T2[1], $T2[2]))
@printf("%8.2f μs\n", t_comm_su2 * 1e6)

# SU(3) generators
T3 = su_generators(3, :λ)
print("   [λ¹, λ²] (SU(3)):       ")
t_comm_su3 = @belapsed normal_form(comm($T3[1], $T3[2]))
@printf("%8.2f μs\n", t_comm_su3 * 1e6)

# SU(4) generators
T4 = su_generators(4, :G)
print("   [G¹, G²] (SU(4)):       ")
t_comm_su4 = @belapsed normal_form(comm($T4[1], $T4[2]))
@printf("%8.2f μs\n", t_comm_su4 * 1e6)

println()

# =============================================================================
# 3. Product Normal Ordering (T^a T^b → normal form)
# =============================================================================

println("3. Product Normal Ordering (A * B)")
println("-"^50)

print("   σx * σy (TLS):          ")
t_prod_tls = @belapsed normal_form($σ_x * $σ_y)
@printf("%8.2f μs\n", t_prod_tls * 1e6)

print("   T¹ * T² (SU(2)):        ")
t_prod_su2 = @belapsed normal_form($T2[1] * $T2[2])
@printf("%8.2f μs\n", t_prod_su2 * 1e6)

print("   λ¹ * λ² (SU(3)):        ")
t_prod_su3 = @belapsed normal_form($T3[1] * $T3[2])
@printf("%8.2f μs\n", t_prod_su3 * 1e6)

print("   G¹ * G² (SU(4)):        ")
t_prod_su4 = @belapsed normal_form($T4[1] * $T4[2])
@printf("%8.2f μs\n", t_prod_su4 * 1e6)

println()

# =============================================================================
# 4. Triple Product (A * B * C)
# =============================================================================

println("4. Triple Product Normal Ordering (A * B * C)")
println("-"^50)

print("   σx * σy * σz (TLS):     ")
t_triple_tls = @belapsed normal_form($σ_x * $σ_y * $σ_z)
@printf("%8.2f μs\n", t_triple_tls * 1e6)

print("   T¹ * T² * T³ (SU(2)):   ")
t_triple_su2 = @belapsed normal_form($T2[1] * $T2[2] * $T2[3])
@printf("%8.2f μs\n", t_triple_su2 * 1e6)

print("   λ¹ * λ² * λ³ (SU(3)):   ")
t_triple_su3 = @belapsed normal_form($T3[1] * $T3[2] * $T3[3])
@printf("%8.2f μs\n", t_triple_su3 * 1e6)

println()

# =============================================================================
# 5. Quadratic Casimir (Σ T^a T^a)
# =============================================================================

println("5. Quadratic Casimir C₂ = Σₐ (T^a)²")
println("-"^50)

# TLS: σx² + σy² + σz² = 3
casimir_tls = σ_x^2 + σ_y^2 + σ_z^2
print("   TLS (3 terms):          ")
t_casimir_tls = @belapsed normal_form($casimir_tls)
@printf("%8.2f μs\n", t_casimir_tls * 1e6)

# SU(2): 3 generators
casimir_su2 = sum(T2[a]^2 for a in 1:3)
print("   SU(2) (3 terms):        ")
t_casimir_su2 = @belapsed normal_form($casimir_su2)
@printf("%8.2f μs\n", t_casimir_su2 * 1e6)

# SU(3): 8 generators
casimir_su3 = sum(T3[a]^2 for a in 1:8)
print("   SU(3) (8 terms):        ")
t_casimir_su3 = @belapsed normal_form($casimir_su3)
@printf("%8.2f μs\n", t_casimir_su3 * 1e6)

# SU(4): 15 generators
casimir_su4 = sum(T4[a]^2 for a in 1:15)
print("   SU(4) (15 terms):       ")
t_casimir_su4 = @belapsed normal_form($casimir_su4)
@printf("%8.2f μs\n", t_casimir_su4 * 1e6)

println()

# =============================================================================
# 6. Mixed Boson + Spin Expressions
# =============================================================================

println("6. Mixed Boson + Spin Expressions")
println("-"^50)

# Jaynes-Cummings type: (a† + a) * σ
jc_tls = (a'() + a()) * σ_x
print("   (a† + a)σx (TLS):       ")
t_jc_tls = @belapsed normal_form($jc_tls)
@printf("%8.2f μs\n", t_jc_tls * 1e6)

jc_su2 = (a'() + a()) * T2[1]
print("   (a† + a)T¹ (SU(2)):     ")
t_jc_su2 = @belapsed normal_form($jc_su2)
@printf("%8.2f μs\n", t_jc_su2 * 1e6)

jc_su3 = (a'() + a()) * T3[1]
print("   (a† + a)λ¹ (SU(3)):     ")
t_jc_su3 = @belapsed normal_form($jc_su3)
@printf("%8.2f μs\n", t_jc_su3 * 1e6)

println()

# =============================================================================
# 7. Heisenberg Equation of Motion
# =============================================================================

println("7. Heisenberg Equation of Motion")
println("-"^50)

H_tls = Pr"ω" * a'() * a() + Pr"g" * (a'() + a()) * σ_x
print("   d/dt a (TLS H):         ")
t_eom_tls = @belapsed heisenberg_eom(a(), $H_tls)
@printf("%8.2f μs\n", t_eom_tls * 1e6)

H_su2 = Pr"ω" * a'() * a() + Pr"g" * (a'() + a()) * T2[1]
print("   d/dt a (SU(2) H):       ")
t_eom_su2 = @belapsed heisenberg_eom(a(), $H_su2)
@printf("%8.2f μs\n", t_eom_su2 * 1e6)

H_su3 = Pr"ω" * a'() * a() + Pr"g" * (a'() + a()) * T3[1]
print("   d/dt a (SU(3) H):       ")
t_eom_su3 = @belapsed heisenberg_eom(a(), $H_su3)
@printf("%8.2f μs\n", t_eom_su3 * 1e6)

print("   d/dt λ¹ (SU(3) H):      ")
t_eom_su3_gen = @belapsed heisenberg_eom($T3[1], $H_su3)
@printf("%8.2f μs\n", t_eom_su3_gen * 1e6)

println()

# =============================================================================
# 8. Scaling with Expression Size
# =============================================================================

println("8. Scaling: (T¹ + T²)^n Normal Ordering")
println("-"^50)

for n in 2:6
    expr_tls = (σ_x + σ_y)^n
    expr_su2 = (T2[1] + T2[2])^n
    expr_su3 = (T3[1] + T3[2])^n
    
    t_tls = @belapsed normal_form($expr_tls)
    t_su2 = @belapsed normal_form($expr_su2)
    t_su3 = @belapsed normal_form($expr_su3)
    
    @printf("   n=%d: TLS %8.2f μs | SU(2) %8.2f μs | SU(3) %8.2f μs\n", 
            n, t_tls*1e6, t_su2*1e6, t_su3*1e6)
end

println()

# =============================================================================
# 9. Many-Term Expressions (Stress Test)
# =============================================================================

println("9. Stress Test: Sum of Many Products")
println("-"^50)

# Create sum of N random products
function make_sum_of_products_tls(N)
    ops = [σx(), σy(), σz()]
    sum(ops[mod1(i,3)] * ops[mod1(i+1,3)] for i in 1:N)
end

function make_sum_of_products_su3(N)
    T = su_generators(3, :λ)
    sum(T[mod1(i,8)] * T[mod1(i+1,8)] for i in 1:N)
end

for N in [10, 50, 100]
    expr_tls = make_sum_of_products_tls(N)
    expr_su3 = make_sum_of_products_su3(N)
    
    t_tls = @belapsed normal_form($expr_tls)
    t_su3 = @belapsed normal_form($expr_su3)
    
    @printf("   N=%3d terms: TLS %10.2f μs | SU(3) %10.2f μs | ratio %.1fx\n", 
            N, t_tls*1e6, t_su3*1e6, t_su3/t_tls)
end

println()

# =============================================================================
# 10. Structure Constant Lookup (Micro-benchmark)
# =============================================================================

println("10. Structure Constant Lookup (internal)")
println("-"^50)

# Get algebras
alg2 = QuantumAlgebra.get_algebra(QuantumAlgebra.get_or_create_su(2))
alg3 = QuantumAlgebra.get_algebra(QuantumAlgebra.get_or_create_su(3))
alg4 = QuantumAlgebra.get_algebra(QuantumAlgebra.get_or_create_su(4))

print("   SU(2) f[1,2]:           ")
t_lookup_su2 = @belapsed QuantumAlgebra.structure_constants($alg2, 1, 2)
@printf("%8.2f ns\n", t_lookup_su2 * 1e9)

print("   SU(3) f[1,4]:           ")
t_lookup_su3 = @belapsed QuantumAlgebra.structure_constants($alg3, 1, 4)
@printf("%8.2f ns\n", t_lookup_su3 * 1e9)

print("   SU(4) f[1,5]:           ")
t_lookup_su4 = @belapsed QuantumAlgebra.structure_constants($alg4, 1, 5)
@printf("%8.2f ns\n", t_lookup_su4 * 1e9)

print("   SU(3) product_coeffs:   ")
t_prod_coeffs = @belapsed QuantumAlgebra.product_coefficients($alg3, 1, 2)
@printf("%8.2f ns\n", t_prod_coeffs * 1e9)

println()

# =============================================================================
# 11. Memory Allocation Analysis
# =============================================================================

println("11. Memory Allocations")
println("-"^50)

print("   normal_form(σx*σy):     ")
alloc_tls = @allocated normal_form(σ_x * σ_y)
@printf("%8d bytes\n", alloc_tls)

print("   normal_form(T¹*T²) SU2: ")
alloc_su2 = @allocated normal_form(T2[1] * T2[2])
@printf("%8d bytes\n", alloc_su2)

print("   normal_form(λ¹*λ²) SU3: ")
alloc_su3 = @allocated normal_form(T3[1] * T3[2])
@printf("%8d bytes\n", alloc_su3)

casimir_su3_fresh = sum(su_generators(3, :λ)[a]^2 for a in 1:8)
print("   normal_form(C₂) SU(3):  ")
alloc_casimir = @allocated normal_form(casimir_su3_fresh)
@printf("%8d bytes\n", alloc_casimir)

println()

# =============================================================================
# Summary
# =============================================================================

println("="^70)
println("                           Summary")
println("="^70)
println()
println("Key observations:")
println("  - Compare TLS vs SU(2) to see overhead of our implementation")
println("  - Compare SU(2) vs SU(3) to see multi-term result overhead")
println("  - Check scaling behavior for large expressions")
println("  - Memory allocations indicate GC pressure")
println()
println("If SU(2) >> TLS: Our infrastructure has overhead")
println("If SU(3) >> SU(2): Multi-term results cause blowup")
println()
