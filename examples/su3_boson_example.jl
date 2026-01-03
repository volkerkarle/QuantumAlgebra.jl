# =============================================================================
# SU(3) ⊗ Boson System Example
# =============================================================================
#
# This example demonstrates the SU(N) Lie algebra extension in QuantumAlgebra.jl
# We model a three-level atom (qutrit) coupled to a bosonic cavity mode.
#
# Physical system: A Λ-type three-level atom interacting with a quantized
# electromagnetic field, relevant for:
#   - Electromagnetically induced transparency (EIT)
#   - Stimulated Raman adiabatic passage (STIRAP)
#   - Quantum memories
#
# =============================================================================

using QuantumAlgebra

# Enable automatic normal ordering for cleaner output
QuantumAlgebra.auto_normal_form(true)

println("="^70)
println("         SU(3) ⊗ Boson System: Three-Level Atom in a Cavity")
println("="^70)
println()

# =============================================================================
# 1. Define the SU(3) generators for the three-level atom
# =============================================================================

println("1. SU(3) Generators")
println("-"^40)

# Create all 8 Gell-Mann generators with name λ (lambda)
λ = su_generators(3, :λ)

println("   The 8 Gell-Mann matrices T^a = λᵃ/2:")
println("   λ = $(λ)")
println()

# Verify commutation relations: [λᵃ, λᵇ] = i fᵃᵇᶜ λᶜ
# Note: Generalized Gell-Mann ordering differs from standard λ₁-λ₈
println("   Commutation relations [λᵃ, λᵇ] = i fᵃᵇᶜ λᶜ:")
println("   [λ¹, λ⁴] = $(comm(λ[1], λ[4]))")
println("   [λ¹, λ⁵] = $(comm(λ[1], λ[5]))")
println("   [λ⁴, λ⁵] = $(comm(λ[4], λ[5]))")
println()

# =============================================================================
# 2. Define the bosonic cavity mode
# =============================================================================

println("2. Bosonic Cavity Mode")
println("-"^40)

# Cavity photon operators - using the default a, a†
println("   Bosonic commutator: [a, a†] = $(comm(a(), a'()))")
println()

# =============================================================================
# 3. Build the Hamiltonian
# =============================================================================

println("3. Hamiltonian: H = H_atom + H_cavity + H_interaction")
println("-"^40)

# Parameters with nice unicode names
ω_c = Pr"ω_c"      # Cavity frequency
ω₁ = Pr"ω_1"       # Energy of level 1 (ground)
ω₂ = Pr"ω_2"       # Energy of level 2 (excited)
ω₃ = Pr"ω_3"       # Energy of level 3 (metastable)
g = Pr"g"          # Atom-cavity coupling strength

# Atomic Hamiltonian using diagonal generators λ⁷ and λ⁸
# The diagonal generators give energy differences between levels
# H_atom = ω₁|1⟩⟨1| + ω₂|2⟩⟨2| + ω₃|3⟩⟨3|
# Can be expressed using λ⁷ and λ⁸ (plus identity which we ignore as constant)
H_atom = ω₁ * λ[7] + ω₂ * λ[8]

# Cavity Hamiltonian  
H_cavity = ω_c * a'() * a()

# Interaction: Jaynes-Cummings type coupling between levels 1↔2
# Using λ¹ (symmetric) and λ⁴ (antisymmetric) to form transition operators
# The transition |1⟩⟨2| + |2⟩⟨1| corresponds to λ¹
H_int = g * (a'() + a()) * λ[1]

# Total Hamiltonian
H = H_atom + H_cavity + H_int

println("   H_atom    = $H_atom")
println("   H_cavity  = $H_cavity")
println("   H_int     = $H_int")
println()
println("   H_total   = $H")
println()

# =============================================================================
# 4. Heisenberg Equations of Motion
# =============================================================================

println("4. Heisenberg Equations of Motion")
println("-"^40)

# Equation of motion for the cavity field
println("   d/dt ⟨a⟩:")
da_dt = heisenberg_eom(a(), H)
println("   $da_dt")
println()

# Equation of motion for an SU(3) generator
println("   d/dt ⟨λ¹⟩:")
dλ1_dt = heisenberg_eom(λ[1], H)
println("   $dλ1_dt")
println()

println("   d/dt ⟨λ⁷⟩:")
dλ7_dt = heisenberg_eom(λ[7], H)
println("   $dλ7_dt")
println()

# =============================================================================
# 5. Expectation Values and Correlations
# =============================================================================

println("5. Expectation Values and Correlations")
println("-"^40)

# Mixed atom-cavity correlations
println("   Atom-cavity correlations:")
corr1 = expval(a'() * λ[1])
println("   ⟨a† λ¹⟩ = $corr1")

# Cumulant expansion
println()
println("   Cumulant expansion of ⟨a† λ¹ λ⁷⟩:")
mixed_expval = expval_as_corrs(a'() * λ[1] * λ[7])
println("   $mixed_expval")
println()

# =============================================================================
# 6. Vacuum Expectation Values
# =============================================================================

println("6. Vacuum Expectation Values")
println("-"^40)
println("   (Vacuum = lowest weight state |N⟩, i.e., |3⟩ for SU(3), |0⟩ for bosons)")
println()

# For SU(3), vacuum is the lowest weight state |3⟩
# The diagonal generator λ⁸ has ⟨3|λ⁸|3⟩ = -1/√3 ≈ -0.577
println("   ⟨vac| λ⁷ |vac⟩ = $(vacExpVal(λ[7]))")
println("   ⟨vac| λ⁸ |vac⟩ = $(vacExpVal(λ[8]))  (= -1/√3)")
println("   ⟨vac| λ¹ |vac⟩ = $(vacExpVal(λ[1]))")  # Off-diagonal → 0
println()

# Mixed vacuum expectation values
println("   ⟨vac| a† a |vac⟩ = $(vacExpVal(a'() * a()))")
println("   ⟨vac| λ¹ λ¹ |vac⟩ = $(vacExpVal(λ[1] * λ[1]))")
println()

# =============================================================================
# 7. SU(3) Algebra Properties
# =============================================================================

println("7. SU(3) Algebra Properties")
println("-"^40)

# Casimir operator: C₂ = Σₐ λᵃ λᵃ
# For fundamental representation of SU(3): C₂ = 4/3
println("   Quadratic Casimir C₂ = Σₐ (λᵃ)²:")
C₂ = sum(λ[a] * λ[a] for a in 1:8)
println("   C₂ = $C₂")
println("   (Expected: 4/3 for fundamental representation)")
println()

# Anticommutator example
println("   Anticommutator {λ¹, λ²}:")
anticomm = λ[1] * λ[2] + λ[2] * λ[1]
println("   {λ¹, λ²} = $anticomm")
println()

# =============================================================================
# 8. Sums over Indices
# =============================================================================

println("8. Sums over Site Indices")
println("-"^40)

# Array of atoms, each with SU(3) structure
println("   Array of three-level atoms coupled to cavity:")
println()

# Create indexed generators
λ_i = su_generators(3, :λ, :i)
println("   λⁱ = $(λ_i[1])  (generator with site index)")
println()

# Collective coupling Hamiltonian
H_collective = g * (a'() + a()) * ∑(:i, λ_i[1])
println("   H_collective = g (a† + a) Σᵢ λ¹ᵢ")
println("   = $H_collective")
println()

# =============================================================================
# 9. Comparison: SU(2) ↔ TLS
# =============================================================================

println("9. SU(2) ↔ TLS Correspondence")
println("-"^40)

# For comparison, show how SU(2) relates to TLS operators
T = su_generators(2, :T)
println("   SU(2) generators: T = $(T)")
println()

# Convert TLS to SU(2) - note: preserves operator name from TLS
println("   TLS → SU(2) conversion (σᵃ = 2Tᵃ):")
println("   σˣ → $(tls_to_su2(σx()))  (= 2T¹ with name σ)")
println("   σʸ → $(tls_to_su2(σy()))  (= 2T² with name σ)")
println("   σᶻ → $(tls_to_su2(σz()))  (= 2T³ with name σ)")
println()

# Ladder operators
Tplus, Tminus, Tz = su2_ladder_operators(:T)
println("   SU(2) ladder operators:")
println("   T⁺ = T¹ + iT² = $Tplus")
println("   T⁻ = T¹ - iT² = $Tminus")
println("   [T⁺, T⁻] = $(comm(Tplus, Tminus))")
println()

# =============================================================================
# 10. Building Equations of Motion System
# =============================================================================

println("10. System of Equations for Mean-Field Dynamics")
println("-"^40)

# Simple mean-field Hamiltonian for demonstration
H_mf = ω_c * a'() * a() + g * (a'() + a()) * λ[1]

# Add cavity decay
κ = Pr"κ"
Ls = ((κ, a()),)

println("   Hamiltonian: H = ωc a†a + g(a† + a)λ¹")
println("   Lindblad: L = √κ a (cavity decay)")
println()

# Get equations for cavity amplitude
eom_a = heisenberg_eom(a(), H_mf, Ls)
println("   d⟨a⟩/dt = $eom_a")
println()

# =============================================================================
println()
println("="^70)
println("                          Example Complete!")
println("="^70)
println()
println("This example demonstrated:")
println("  - SU(3) generators for three-level atoms")
println("  - Coupling to bosonic modes")
println("  - Heisenberg equations of motion")
println("  - Expectation values and correlations")
println("  - Vacuum expectation values")
println("  - Sum notation for atom arrays")
println("  - Comparison with SU(2)/TLS")
println()
