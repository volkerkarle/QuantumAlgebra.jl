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
# 5. Computing Expectation Values in Excited States
# =============================================================================

println("5. Expectation Values in Excited States")
println("-"^40)

# We use vacExpVal(A, S) to compute ⟨ψ|A|ψ⟩ where |ψ⟩ = S|vac⟩
# For bosons: |n⟩ = (a†)^n / √(n!) |0⟩

println("   a) Fock states |n⟩ for cavity:")
n̂ = a'() * a()

# |1⟩ state
S_fock1 = a'()
println("      ⟨1|n̂|1⟩ = $(vacExpVal(n̂, S_fock1))")
println("      ⟨1|a|1⟩ = $(vacExpVal(a(), S_fock1))")

# |2⟩ state  
S_fock2 = a'()^2 / sqrt(2)
println("      ⟨2|n̂|2⟩ = $(vacExpVal(n̂, S_fock2))")
println("      ⟨2|a†a†aa|2⟩ = $(vacExpVal(a'()^2 * a()^2, S_fock2))")

# |3⟩ state
S_fock3 = a'()^3 / sqrt(6)
println("      ⟨3|n̂|3⟩ = $(vacExpVal(n̂, S_fock3))")
println("      ⟨3|n̂²|3⟩ = $(vacExpVal(n̂^2, S_fock3))")
println()

println("   b) Variance in Fock states (should be 0):")
println("      ⟨2|(Δn)²|2⟩ = ⟨n²⟩ - ⟨n⟩² = $(vacExpVal(n̂^2, S_fock2)) - $(vacExpVal(n̂, S_fock2))² = 0")
println()

println("   c) Mixed boson-atom expectation values:")
# Cavity in |1⟩, atom in vacuum |3⟩
println("      Create state: |ψ⟩ = a†|0⟩ ⊗ |3⟩_atom")
println("      ⟨ψ| a†λ¹a |ψ⟩ = $(vacExpVal(a'() * λ[1] * a(), a'()))")
println("      ⟨ψ| n̂ ⊗ λ⁸ |ψ⟩ = $(vacExpVal(n̂ * λ[8], a'()))")
println("      ⟨ψ| a†λ⁸ + λ⁸a |ψ⟩ = $(vacExpVal(a'()*λ[8] + λ[8]*a(), a'()))")
println()

println("   d) Higher Fock state with atom:")
# |n=2⟩ ⊗ |3⟩
println("      State: |ψ⟩ = (a†)²/√2 |0⟩ ⊗ |3⟩_atom")
println("      ⟨ψ| n̂ ⊗ λ⁸ |ψ⟩ = $(vacExpVal(n̂ * λ[8], S_fock2))")
println("      ⟨ψ| (a†a)² ⊗ (λ⁸)² |ψ⟩ = $(vacExpVal(n̂^2 * λ[8]^2, S_fock2))")
println()

# =============================================================================
# 6. Cumulant Expansions and Correlations
# =============================================================================

println("6. Cumulant Expansions and Correlations")
println("-"^40)

# The cumulant expansion decomposes ⟨ABC...⟩ into products of correlators
println("   Cumulant expansion ⟨AB⟩ = ⟨A⟩⟨B⟩ + ⟨AB⟩c:")
println()

# Two-operator expansion
expr_2op = a'() * λ[1]
println("   ⟨a† λ¹⟩ expands to:")
println("      $(expval_as_corrs(expr_2op))")
println()

# Three-operator expansion  
expr_3op = a'() * a() * λ[1]
println("   ⟨a†a λ¹⟩ expands to:")
println("      $(expval_as_corrs(expr_3op))")
println()

# Four-operator mixed expansion
expr_4op = a'() * a() * λ[1] * λ[7]
println("   ⟨a†a λ¹λ⁷⟩ expands to:")
println("      $(expval_as_corrs(expr_4op))")
println()

# =============================================================================
# 7. Vacuum State Properties
# =============================================================================

println("7. Vacuum State Properties")
println("-"^40)
println("   Vacuum = |0⟩_boson ⊗ |3⟩_atom (lowest weight state for SU(3))")
println()

# For SU(3), vacuum is the lowest weight state |3⟩
# The diagonal generator λ⁸ has ⟨3|λ⁸|3⟩ = -1/√3 ≈ -0.577
println("   Diagonal generators in vacuum:")
println("   ⟨vac| λ⁷ |vac⟩ = $(vacExpVal(λ[7]))")
println("   ⟨vac| λ⁸ |vac⟩ = $(vacExpVal(λ[8]))  (= -1/√3)")
println()

println("   Off-diagonal generators (transitions) in vacuum:")
println("   ⟨vac| λ¹ |vac⟩ = $(vacExpVal(λ[1]))")
println("   ⟨vac| λ⁴ |vac⟩ = $(vacExpVal(λ[4]))")
println()

println("   Products of generators in vacuum:")
println("   ⟨vac| λ¹λ¹ |vac⟩ = $(vacExpVal(λ[1] * λ[1]))")
println("   ⟨vac| λ⁷λ⁷ |vac⟩ = $(vacExpVal(λ[7] * λ[7]))")
println("   ⟨vac| λ⁸λ⁸ |vac⟩ = $(vacExpVal(λ[8] * λ[8]))  (= 1/3)")
println()

println("   Bosonic vacuum:")
println("   ⟨vac| a†a |vac⟩ = $(vacExpVal(a'() * a()))")
println("   ⟨vac| aa† |vac⟩ = $(vacExpVal(a() * a'()))")
println()

# =============================================================================
# 8. SU(3) Algebra Properties
# =============================================================================

println("8. SU(3) Algebra Properties")
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
# 9. Sums over Site Indices
# =============================================================================

println("9. Sums over Site Indices")
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
# 10. Comparison: SU(2) ↔ TLS
# =============================================================================

println("10. SU(2) ↔ TLS Correspondence")
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
# 11. Building Equations of Motion System
# =============================================================================

println("11. System of Equations for Mean-Field Dynamics")
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
println("  - SU(3) generators for three-level atoms (qutrits)")
println("  - Coupling to bosonic cavity modes")
println("  - Heisenberg equations of motion")
println("  - Computing expectation values in Fock states")
println("  - Mixed atom-photon expectation values")
println("  - Cumulant expansions for correlations")
println("  - Vacuum state properties")
println("  - Sum notation for atom arrays")
println("  - SU(2)/TLS correspondence")
println()
