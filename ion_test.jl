# ion_test.jl
# Spin-dependent squeezing simulation for trapped ions
# Implements stroboscopic pulse sequence from arXiv:2510.25870 (Bond et al.)
# Hamiltonian Eq.(23), pulse protocol Fig.4(c), parameters from Fig.5
# Usage: julia --project=/Users/sjq/Code/UNM_workspace ion_test.jl

using LinearAlgebra
using OrdinaryDiffEq
using Printf

# ===== OPERATOR CONSTRUCTION =====

"""Bosonic annihilation operator in Fock basis {|0⟩,...,|nmax⟩}"""
function boson_a(nmax::Int)
    a = zeros(ComplexF64, nmax + 1, nmax + 1)
    for n in 1:nmax
        a[n, n + 1] = sqrt(n)
    end
    return a
end

"""Collective spin operators Jx, Jy, Jz in Dicke basis |m⟩, m = -J,...,+J.
   Basis ordering: |−J⟩, |−J+1⟩, ..., |+J⟩  (dimension N+1, J=N/2)."""
function spin_ops(N::Int)
    J = N / 2
    dim = N + 1
    Jz = zeros(ComplexF64, dim, dim)
    Jp = zeros(ComplexF64, dim, dim)  # J+

    for idx in 1:dim
        m = -J + (idx - 1)
        Jz[idx, idx] = m
        if idx < dim
            mp = m + 1
            Jp[idx + 1, idx] = sqrt(J * (J + 1) - m * mp)
        end
    end

    Jm = Jp'  # J-
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2im)
    return Jx, Jy, Jz
end

# ===== SQUEEZED VACUUM STATE =====

"""Squeezed vacuum S(r)|0⟩ in Fock basis via matrix exponentiation.
   S(r) = exp(r/2 * (a² − a†²)) for real r."""
function squeezed_vacuum(r::Float64, nmax::Int)
    a = boson_a(nmax)
    ad = a'
    # Generator: r/2 * (a² - a†²)
    G = (r / 2) * (a^2 - ad^2)
    S = exp(Matrix(G))
    psi = S[:, 1]  # S|0⟩
    return psi
end

# ===== TARGET STATE CONSTRUCTION =====

"""Build target state S(ζ Jz)|0⟩_b|ψ₀⟩_s in the full tensor product space.
   Hilbert space: |n⟩_b ⊗ |m⟩_s, dimension (nmax+1)*(N+1).
   init: :GHZ or :polarized"""
function build_target(ζ::Float64, N::Int, nmax::Int; init::Symbol=:GHZ)
    J = N / 2
    dim_b = nmax + 1
    dim_s = N + 1

    # Spin coefficients c_m in Dicke basis
    c = zeros(ComplexF64, dim_s)
    if init == :GHZ
        c[1] = 1.0 / sqrt(2)        # |−J⟩
        c[end] = 1.0 / sqrt(2)      # |+J⟩
    elseif init == :polarized
        c[end] = 1.0                 # |+J⟩
    else
        error("Unknown init state: $init")
    end

    # Build: ψ_target = Σ_m c_m |ζm⟩_b ⊗ |m⟩_s
    psi = zeros(ComplexF64, dim_b * dim_s)
    for idx_s in 1:dim_s
        m = -J + (idx_s - 1)
        if abs(c[idx_s]) < 1e-15
            continue
        end
        # Squeezed vacuum with parameter ζ*m
        psi_b = squeezed_vacuum(ζ * m, nmax)
        # Tensor product: |n⟩_b ⊗ |m⟩_s  →  index (n-1)*dim_s + m
        for idx_b in 1:dim_b
            psi[(idx_b - 1) * dim_s + idx_s] = c[idx_s] * psi_b[idx_b]
        end
    end
    normalize!(psi)
    return psi
end

# ===== PULSE SEQUENCE =====

"""Determine (Δ_eff, ϕ_eff, g_eff) at time t for the stroboscopic protocol.
   τ = 2πℓ/|Δ|, 4 segments per cycle, P cycles total."""
function pulse_params(t::Float64, Δ::Float64, ϕ1::Float64, ϕ2::Float64,
                      g0::Float64, τ::Float64)
    t_mod = mod(t, 4τ)
    if t_mod < τ
        return (+Δ, ϕ1, +g0)
    elseif t_mod < 2τ
        return (-Δ, ϕ2, +g0)
    elseif t_mod < 3τ
        return (+Δ, ϕ1, -g0)
    else
        return (-Δ, ϕ2, -g0)
    end
end

# ===== HAMILTONIAN AND ODE =====

"""Precompute the 4 static operator matrices for the Hamiltonian."""
struct HamiltonianOps
    aJx::Matrix{ComplexF64}   # a ⊗ Jx
    aJy::Matrix{ComplexF64}   # a ⊗ Jy
    adJx::Matrix{ComplexF64}  # a† ⊗ Jx
    adJy::Matrix{ComplexF64}  # a† ⊗ Jy
end

function build_hamiltonian_ops(N::Int, nmax::Int)
    a = boson_a(nmax)
    ad = a'
    Jx, Jy, _ = spin_ops(N)
    aJx  = kron(Matrix(a),  Jx)
    aJy  = kron(Matrix(a),  Jy)
    adJx = kron(Matrix(ad), Jx)
    adJy = kron(Matrix(ad), Jy)

    return HamiltonianOps(aJx, aJy, adJx, adJy)
end

"""Apply H(t)|ψ⟩ in-place.
   H(t) = g(t) * a * [Jx e^{-iΔt} + Jy e^{+iΔt} e^{-iϕ}] + h.c."""
function hamiltonian_action!(Hpsi::Vector{ComplexF64}, psi::Vector{ComplexF64},
                             ops::HamiltonianOps, t::Float64,
                             Δ_eff::Float64, ϕ_eff::Float64, g_eff::Float64)
    # Coefficients for the two terms
    c1 = g_eff * exp(-1im * Δ_eff * t)                   # coefficient of a⊗Jx
    c2 = g_eff * exp(+1im * Δ_eff * t) * exp(-1im * ϕ_eff) # coefficient of a⊗Jy

    # H|ψ⟩ = c1*(a⊗Jx)|ψ⟩ + c2*(a⊗Jy)|ψ⟩ + conj(c1)*(a†⊗Jx)|ψ⟩ + conj(c2)*(a†⊗Jy)|ψ⟩
    mul!(Hpsi, ops.aJx, psi)
    Hpsi .*= c1
    Hpsi .+= c2 .* (ops.aJy * psi) .+ conj(c1) .* (ops.adJx * psi) .+ conj(c2) .* (ops.adJy * psi)
end

"""Schrödinger equation: dψ/dt = -i H(t) ψ"""
function schrodinger!(dpsi, psi, params, t)
    ops, Δ_abs, ϕ1, ϕ2, g0, τ = params
    Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ)
    hamiltonian_action!(dpsi, psi, ops, t, Δ_eff, ϕ_eff, g_eff)
    dpsi .*= -1im
end

# ===== MAIN SIMULATION =====

"""Run the spin-dependent squeezing simulation.

Parameters:
- N: number of ions
- nmax: Fock space truncation
- z_target: target squeezing parameter z = |ζ|*N/2
- P: number of stroboscopic cycles
- ℓ: number of phase-space loops per segment
- ϕ1, ϕ2: laser phases
- init: initial spin state (:GHZ or :polarized)
"""
function simulate(; N::Int=1, nmax::Int=20, z_target::Float64=1.0,
                   P::Int=5, ℓ::Int=1, ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                   init::Symbol=:GHZ)
    J = N / 2
    dim_b = nmax + 1
    dim_s = N + 1
    dim = dim_b * dim_s

    # Coupling: g = 2π × 5/√N kHz  (angular frequency in kHz units → multiply by 2π)
    g0 = 2π * 5.0 / sqrt(N)  # kHz (angular)

    # Squeezing parameter: |ζ| = 2*z_target/N
    ζ = 2 * z_target / N

    # From Eq.31: |ζ| = 16π g² ℓ P / Δ²  →  Δ = sqrt(16π g² ℓ P / |ζ|)
    Δ_abs = sqrt(16π * g0^2 * ℓ * P / ζ)  # kHz (angular)

    # Segment duration: τ = 2πℓ/|Δ|  (in ms since g and Δ are in kHz)
    τ = 2π * ℓ / Δ_abs

    # Total time
    tf = 4 * P * τ

    @printf("=== Spin-dependent squeezing simulation ===\n")
    @printf("N = %d, J = %.1f, nmax = %d\n", N, J, nmax)
    @printf("z_target = %.3f, P = %d, ℓ = %d\n", z_target, P, ℓ)
    @printf("ϕ₁ = %.4f, ϕ₂ = %.4f\n", ϕ1, ϕ2)
    @printf("Init state: %s\n", init)
    @printf("g = 2π × %.3f kHz\n", g0 / (2π))
    @printf("|ζ| = %.6f\n", ζ)
    @printf("|Δ| = 2π × %.3f kHz\n", Δ_abs / (2π))
    @printf("τ = %.6f ms\n", τ)
    @printf("tf = %.6f ms\n", tf)
    @printf("Hilbert space dim = %d × %d = %d\n", dim_b, dim_s, dim)

    # Build operators
    @printf("\nBuilding operators...\n")
    ops = build_hamiltonian_ops(N, nmax)

    # Build initial state: |0⟩_b ⊗ |ψ₀⟩_s
    psi0 = zeros(ComplexF64, dim)
    if init == :GHZ
        # GHZ: (|−J⟩ + |+J⟩)/√2
        # |0⟩_b is idx_b = 1; |−J⟩ is idx_s = 1; |+J⟩ is idx_s = dim_s
        psi0[(1 - 1) * dim_s + 1] = 1.0 / sqrt(2)       # |0⟩_b|−J⟩
        psi0[(1 - 1) * dim_s + dim_s] = 1.0 / sqrt(2)   # |0⟩_b|+J⟩
    elseif init == :polarized
        psi0[(1 - 1) * dim_s + dim_s] = 1.0              # |0⟩_b|+J⟩
    end

    # Build target state
    @printf("Building target state...\n")
    psi_target = build_target(ζ, N, nmax; init=init)

    # Expected mean photon number for target
    a_op = boson_a(nmax)
    n_op = kron(a_op' * a_op, Matrix{ComplexF64}(I, dim_s, dim_s))
    n_target = real(dot(psi_target, n_op * psi_target))
    @printf("Target ⟨n⟩ = %.4f (expected sinh²(ζJ) = %.4f)\n",
            n_target, sinh(ζ * J)^2)

    # Segment boundaries for tstops
    tstops = Float64[]
    for p in 0:(P - 1)
        t0 = 4p * τ
        push!(tstops, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops, tf)
    unique!(sort!(tstops))

    # Dense save times for smooth plotting + segment boundaries
    n_save = max(500, 100 * P)
    saveat_dense = collect(range(0.0, tf, length=n_save))
    saveat_all = sort!(unique!(vcat(saveat_dense, tstops)))

    # ODE integration
    @printf("\nIntegrating Schrödinger equation...\n")
    params = (ops, Δ_abs, ϕ1, ϕ2, g0, τ)
    prob = ODEProblem(schrodinger!, psi0, (0.0, tf), params)
    sol = solve(prob, Tsit5();
                abstol=1e-10, reltol=1e-10,
                saveat=saveat_all,
                tstops=tstops,
                maxiters=10_000_000)

    @printf("Integration complete. %d saved points.\n", length(sol.t))

    # Compute observables at all saved times
    times = sol.t
    fidelities = Float64[]
    n_avgs = Float64[]
    norms = Float64[]
    for psi_t in sol.u
        push!(fidelities, abs2(dot(psi_target, psi_t)))
        push!(n_avgs, real(dot(psi_t, n_op * psi_t)))
        push!(norms, norm(psi_t))
    end

    # Print per-cycle results
    @printf("\n--- Results per stroboscopic cycle ---\n")
    @printf("%5s  %12s  %10s  %10s\n", "Cycle", "Time (ms)", "Fidelity", "⟨n⟩")
    for p in 1:P
        t_cycle = 4p * τ
        idx = findfirst(t -> abs(t - t_cycle) < 1e-12, sol.t)
        if idx !== nothing
            @printf("%5d  %12.6f  %10.6f  %10.4f\n", p, t_cycle, fidelities[idx], n_avgs[idx])
        end
    end

    # Final state analysis
    F_final = fidelities[end]
    n_final = n_avgs[end]
    norm_final = norms[end]

    @printf("\n--- Final state ---\n")
    @printf("Fidelity:        %.8f\n", F_final)
    @printf("⟨n⟩:             %.6f\n", n_final)
    @printf("‖ψ‖:             %.10f\n", norm_final)
    @printf("Squeezing (dB):  %.2f\n", -10 * log10(exp(-2 * ζ * J)))

    return (; sol, times, fidelities, n_avgs, norms,
              F_final, psi_target, psi_final=sol.u[end],
              N, nmax, P, τ, tf, ζ, Δ_abs, g0, ϕ1, ϕ2, J,
              n_op, dim_s, dim_b)
end

# ===== SWEEP UTILITY =====

"""Sweep over P values to find minimum-time protocol with F ≥ F_threshold."""
function sweep_P(; N::Int=1, nmax::Int=20, z_target::Float64=1.0,
                  P_range=1:20, ℓ::Int=1, ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                  init::Symbol=:GHZ, F_threshold::Float64=0.99)
    @printf("=== Sweep P for N=%d, z=%.2f, F_thresh=%.2f ===\n", N, z_target, F_threshold)
    @printf("%5s  %12s  %10s\n", "P", "tf (ms)", "Fidelity")

    results = Tuple{Int, Float64, Float64}[]
    for P in P_range
        g0 = 2π * 5.0 / sqrt(N)
        ζ = 2 * z_target / N
        Δ_abs = sqrt(16π * g0^2 * ℓ * P / ζ)
        τ = 2π * ℓ / Δ_abs
        tf = 4 * P * τ

        ops = build_hamiltonian_ops(N, nmax)
        psi0 = zeros(ComplexF64, (nmax + 1) * (N + 1))
        if init == :GHZ
            psi0[1] = 1.0 / sqrt(2)
            psi0[N + 1] = 1.0 / sqrt(2)
        elseif init == :polarized
            psi0[N + 1] = 1.0
        end
        psi_target = build_target(ζ, N, nmax; init=init)

        tstops = Float64[]
        for p in 0:(P - 1)
            t0 = 4p * τ
            push!(tstops, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
        end
        push!(tstops, tf)
        unique!(sort!(tstops))

        params = (ops, Δ_abs, ϕ1, ϕ2, g0, τ)
        prob = ODEProblem(schrodinger!, psi0, (0.0, tf), params)
        sol = solve(prob, Tsit5();
                    abstol=1e-10, reltol=1e-10,
                    tstops=tstops,
                    maxiters=10_000_000)

        F = abs2(dot(psi_target, sol.u[end]))
        @printf("%5d  %12.6f  %10.6f\n", P, tf, F)
        push!(results, (P, tf, F))
    end

    # Find minimum P with F ≥ threshold
    passing = filter(r -> r[3] >= F_threshold, results)
    if !isempty(passing)
        best = passing[argmin([r[2] for r in passing])]
        @printf("\nMin-time protocol: P=%d, tf=%.6f ms, F=%.6f\n", best...)
    else
        @printf("\nNo protocol achieved F ≥ %.2f in the scanned range.\n", F_threshold)
    end

    return results
end

# ===== VISUALIZATION =====

using Plots

"""Plot simulation results: 4-panel figure."""
function plot_results(res; save_path::String="ion_squeezing.png")
    (; times, fidelities, n_avgs, norms, P, τ, tf, ζ, J,
       psi_final, psi_target, N, nmax, dim_s, dim_b, n_op) = res

    default(fontfamily="Computer Modern", titlefontsize=13, guidefontsize=11,
            tickfontsize=9, legendfontsize=9, linewidth=2, dpi=200)

    # --- Panel 1: Fidelity vs time ---
    p1 = plot(times, fidelities,
              xlabel="Time (ms)", ylabel="Fidelity",
              title="Fidelity F(t) = |⟨ψ_target|ψ(t)⟩|²",
              label="F(t)", color=:blue, legend=:bottomright)
    hline!([1.0], linestyle=:dash, color=:gray, label="", alpha=0.5)
    # Mark cycle boundaries
    for p in 1:P
        vline!([4p * τ], linestyle=:dot, color=:lightgray, label="", alpha=0.5)
    end

    # --- Panel 2: Mean photon number vs time ---
    n_target_val = sinh(ζ * J)^2
    p2 = plot(times, n_avgs,
              xlabel="Time (ms)", ylabel="⟨n⟩",
              title="Mean photon number",
              label="⟨n⟩(t)", color=:red, legend=:topleft)
    hline!([n_target_val], linestyle=:dash, color=:gray,
           label=@sprintf("target sinh²(ζJ)=%.2f", n_target_val), alpha=0.7)
    for p in 1:P
        vline!([4p * τ], linestyle=:dot, color=:lightgray, label="", alpha=0.5)
    end

    # --- Panel 3: Photon number distribution of final state ---
    # P(n) = Σ_m |⟨n,m|ψ⟩|² — marginal over spin
    pn_final = zeros(dim_b)
    pn_target = zeros(dim_b)
    for ib in 1:dim_b
        for is in 1:dim_s
            idx = (ib - 1) * dim_s + is
            pn_final[ib] += abs2(psi_final[idx])
            pn_target[ib] += abs2(psi_target[idx])
        end
    end
    n_plot_max = min(nmax, findfirst(x -> x < 1e-6, pn_target[2:end] .+ pn_final[2:end]))
    n_plot_max = something(n_plot_max, nmax)
    n_range = 0:n_plot_max
    p3 = bar(n_range, pn_target[1:n_plot_max+1],
             xlabel="Fock state n", ylabel="P(n)",
             title="Photon number distribution",
             label="Target", color=:gray, alpha=0.4, bar_width=0.8)
    bar!(n_range, pn_final[1:n_plot_max+1],
         label="Final ψ(tf)", color=:blue, alpha=0.6, bar_width=0.4)

    # --- Panel 4: Pulse sequence ---
    t_pulse = range(0.0, tf, length=1000)
    Δ_vals = Float64[]
    g_vals = Float64[]
    for t in t_pulse
        Δ_eff, _, g_eff = pulse_params(t, res.Δ_abs, res.ϕ1, res.ϕ2, res.g0, τ)
        push!(Δ_vals, Δ_eff / (2π))
        push!(g_vals, g_eff / (2π))
    end
    p4 = plot(collect(t_pulse), Δ_vals,
              xlabel="Time (ms)", ylabel="Frequency (kHz)",
              title="Stroboscopic pulse sequence",
              label="Δ/(2π)", color=:orange, legend=:topright)
    plot!(collect(t_pulse), g_vals,
          label="g/(2π)", color=:green, linestyle=:dash)
    for p in 1:P
        vline!([4p * τ], linestyle=:dot, color=:lightgray, label="", alpha=0.5)
    end

    # Combine
    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 800),
               plot_title=@sprintf("N=%d, z=%.1f, P=%d, F=%.4f", N, ζ * J, P, fidelities[end]),
               plot_titlefontsize=14, margin=5Plots.mm)

    savefig(fig, save_path)
    println("\nPlot saved to: $save_path")
    return fig
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    # Main simulation: N=1, z=1.0, P=5
    res = simulate(N=1, nmax=20, z_target=1.0, P=5)

    println("\n" * "="^50)
    println("Quick verification:")
    @printf("  Expected ⟨n⟩ ≈ sinh²(1) = %.4f\n", sinh(1.0)^2)
    @printf("  Final fidelity: %.6f\n", res.F_final)

    # Visualize
    plot_results(res; save_path="ion_squeezing_N1_z1.png")

    # Additional: small z, P=1 (near-ideal regime)
    println("\n" * "="^50)
    res2 = simulate(N=1, nmax=20, z_target=0.3, P=1)
    plot_results(res2; save_path="ion_squeezing_N1_z03.png")
end
