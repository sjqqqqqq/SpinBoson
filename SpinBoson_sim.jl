# SpinBoson_sim.jl
# Spin-dependent squeezing simulation for trapped ions, using QuantumOptics.jl.
# Same physics as ion_test.jl: Hamiltonian Eq.(23) of arXiv:2510.25870, pulse
# protocol Fig.4(c), parameters from Fig.5.
# Usage: julia --project=. SpinBoson_sim.jl

using QuantumOptics
using OrdinaryDiffEq
using LinearAlgebra
using Printf
using JLD2

# ===== BASES AND OPERATORS =====

"""Build the boson-spin tensor-product basis and the four static operators
   that appear in H(t) = c1(t)·a⊗Jx + c2(t)·a⊗Jy + h.c."""
function build_spinboson(N::Int, nmax::Int)
    b_fock = FockBasis(nmax)
    b_spin = SpinBasis(N // 2)
    b_full = b_fock ⊗ b_spin

    a  = destroy(b_fock)
    ad = create(b_fock)

    # In QuantumOptics, sigmax/y/z(SpinBasis(j)) = 2·Jx/y/z, sigmap = J+, sigmam = J-.
    # Basis ordering is descending m: basisstate(b_spin, 1) = |+j⟩, …, basisstate(b_spin, 2j+1) = |−j⟩.
    Jx = sigmax(b_spin) / 2
    Jy = sigmay(b_spin) / 2
    Jz = sigmaz(b_spin) / 2

    aJx  = a  ⊗ Jx
    aJy  = a  ⊗ Jy
    adJx = ad ⊗ Jx
    adJy = ad ⊗ Jy
    n_op = (ad * a) ⊗ one(b_spin)

    return (; b_fock, b_spin, b_full, a, ad, Jx, Jy, Jz,
              aJx, aJy, adJx, adJy, n_op)
end

# ===== STATE PREPARATION =====

"""Initial state |0⟩_b ⊗ |ψ₀⟩_s. Convention: :GHZ → (|+J⟩+|−J⟩)/√2, :polarized → |+J⟩."""
function build_initial(N::Int, sb; init::Symbol=:GHZ)
    vac = fockstate(sb.b_fock, 0)
    dim_s = N + 1
    if init == :GHZ
        spin = (basisstate(sb.b_spin, 1) + basisstate(sb.b_spin, dim_s)) / sqrt(2)
    elseif init == :polarized
        spin = basisstate(sb.b_spin, 1)            # |+J⟩
    else
        error("Unknown init state: $init")
    end
    return vac ⊗ spin
end

"""Target state Σ_m c_m S(ζm)|0⟩_b ⊗ |m⟩_s, with QuantumOptics' built-in
   squeeze operator S(z) = exp((conj(z)·a² − z·a†²)/2)."""
function build_target(ζ::Float64, N::Int, sb; init::Symbol=:GHZ)
    j = N / 2
    dim_s = N + 1
    vac = fockstate(sb.b_fock, 0)

    # Spin coefficients in QuantumOptics ordering (idx 1 ↔ m=+j, idx end ↔ m=−j)
    c = zeros(ComplexF64, dim_s)
    if init == :GHZ
        c[1]   = 1 / sqrt(2)
        c[end] = 1 / sqrt(2)
    elseif init == :polarized
        c[1]   = 1.0
    else
        error("Unknown init state: $init")
    end

    psi = Ket(sb.b_full, zeros(ComplexF64, length(sb.b_full)))
    for idx_s in 1:dim_s
        abs(c[idx_s]) < 1e-15 && continue
        m = j - (idx_s - 1)                        # descending m
        S = squeeze(sb.b_fock, ζ * m)
        psi += c[idx_s] * (S * vac) ⊗ basisstate(sb.b_spin, idx_s)
    end
    return normalize(psi)
end

# ===== PULSE SEQUENCE =====

"""Stroboscopic pulse — 4 segments per cycle, each of duration τ = 2πℓ/|Δ|.
   Returns (Δ_eff, ϕ_eff, g_eff) at time t."""
@inline function pulse_params(t::Float64, Δ::Float64, ϕ1::Float64, ϕ2::Float64,
                              g0::Float64, τ::Float64)
    t_mod = mod(t, 4τ)
    if t_mod < τ
        return (+Δ, ϕ1, +g0)       # segment 1
    elseif t_mod < 2τ
        return (-Δ, ϕ2, +g0)       # segment 2
    elseif t_mod < 3τ
        return (-Δ, ϕ2, -g0)       # segment 3 = echo of seg 2 with g→−g
    else
        return (+Δ, ϕ1, -g0)       # segment 4 = echo of seg 1 with g→−g
    end
end

# ===== TIME-DEPENDENT HAMILTONIAN =====

"""Closure that returns H(t) for `timeevolution.schroedinger_dynamic`.
   H(t) = g(t)·a·[Jx e^{−iΔ(t)t} + Jy e^{+iΔ(t)t} e^{−iϕ(t)}] + h.c."""
function make_H_dynamic(sb, Δ_abs::Float64, ϕ1::Float64, ϕ2::Float64,
                       g0::Float64, τ::Float64)
    aJx, aJy, adJx, adJy = sb.aJx, sb.aJy, sb.adJx, sb.adJy
    return function H_at(t, _)
        Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ)
        c1 = g_eff * cis(-Δ_eff * t)                       # coeff of a⊗Jx
        c2 = g_eff * cis(+Δ_eff * t) * cis(-ϕ_eff)         # coeff of a⊗Jy
        return c1 * aJx + c2 * aJy + conj(c1) * adJx + conj(c2) * adJy
    end
end

# ===== PROTOCOL PARAMETERS =====

"""Derive (g0, ζ, Δ_abs, τ, tf) from (N, z_target, P, ℓ) using Eq.(31) of the paper."""
function protocol_params(N::Int, z_target::Float64, P::Int, ℓ::Int)
    g0    = 2π * 5.0 / sqrt(N)
    ζ     = 2 * z_target / N
    Δ_abs = sqrt(16π * g0^2 * ℓ * P / ζ)
    τ     = 2π * ℓ / Δ_abs
    tf    = 4 * P * τ
    return (; g0, ζ, Δ_abs, τ, tf)
end

# ===== MAIN SIMULATION =====

"""Run the spin-dependent squeezing simulation in the QuantumOptics.jl framework.

Parameters mirror ion_test.jl:
  N         number of ions (collective spin J=N/2)
  nmax      Fock-space truncation
  z_target  target squeezing z = |ζ|·N/2
  P, ℓ      stroboscopic cycles and phase-space loops per segment
  ϕ1, ϕ2    laser phases (default ϕ2 = ϕ1 − π = 0)
  init      :GHZ or :polarized
"""
function simulate(; N::Int=1, nmax::Int=20, z_target::Float64=1.0,
                   P::Int=5, ℓ::Int=1, ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                   init::Symbol=:GHZ)
    sb = build_spinboson(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ, tf) = pp

    @printf("=== Spin-dependent squeezing (QuantumOptics.jl) ===\n")
    @printf("N = %d, J = %.1f, nmax = %d\n", N, N/2, nmax)
    @printf("z_target = %.3f, P = %d, ℓ = %d\n", z_target, P, ℓ)
    @printf("ϕ₁ = %.4f, ϕ₂ = %.4f, init = %s\n", ϕ1, ϕ2, init)
    @printf("g = 2π × %.3f kHz, |Δ| = 2π × %.3f kHz\n", g0/(2π), Δ_abs/(2π))
    @printf("|ζ| = %.6f, τ = %.6f ms, tf = %.6f ms\n", ζ, τ, tf)
    @printf("dim(H) = %d\n", length(sb.b_full))

    psi0       = build_initial(N, sb; init)
    psi_target = build_target(ζ, N, sb; init)

    n_target_expect = real(expect(sb.n_op, psi_target))
    @printf("Target ⟨n⟩ = %.4f (analytic sinh²(ζJ) = %.4f)\n",
            n_target_expect, sinh(ζ * N/2)^2)

    # Segment boundaries: solver must not smooth across the discontinuities in g(t),Δ(t),ϕ(t).
    tstops = Float64[]
    for p in 0:(P - 1)
        t0 = 4p * τ
        push!(tstops, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops, tf)
    unique!(sort!(tstops))

    n_save = max(500, 100 * P)
    saveat_dense = collect(range(0.0, tf, length=n_save))
    saveat_all   = sort!(unique!(vcat(saveat_dense, tstops)))

    Hf = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ)
    @printf("\nIntegrating Schrödinger equation via QuantumOptics…\n")
    tout, psi_t = timeevolution.schroedinger_dynamic(
        saveat_all, psi0, Hf;
        alg=Tsit5(), abstol=1e-10, reltol=1e-10,
        tstops=tstops, maxiters=10_000_000,
    )
    @printf("Integration complete. %d saved points.\n", length(tout))

    fidelities = Float64[abs2(dagger(psi_target) * ψ) for ψ in psi_t]
    n_avgs     = Float64[real(expect(sb.n_op, ψ))     for ψ in psi_t]
    norms      = Float64[norm(ψ)                       for ψ in psi_t]

    @printf("\n--- Results per stroboscopic cycle ---\n")
    @printf("%5s  %12s  %10s  %10s\n", "Cycle", "Time (ms)", "Fidelity", "⟨n⟩")
    for p in 1:P
        t_cycle = 4p * τ
        idx = findfirst(t -> abs(t - t_cycle) < 1e-9, tout)
        if idx !== nothing
            @printf("%5d  %12.6f  %10.6f  %10.4f\n",
                    p, t_cycle, fidelities[idx], n_avgs[idx])
        end
    end

    @printf("\n--- Final state ---\n")
    @printf("Fidelity:        %.8f\n", fidelities[end])
    @printf("⟨n⟩:             %.6f\n", n_avgs[end])
    @printf("‖ψ‖:             %.10f\n", norms[end])
    @printf("Squeezing (dB):  %.2f\n", -10 * log10(exp(-2 * ζ * N/2)))

    return (; sb, tout, psi_t, fidelities, n_avgs, norms,
              psi0, psi_target, psi_final = psi_t[end],
              F_final = fidelities[end],
              N, nmax, P, ℓ, ϕ1, ϕ2, init,
              g0, ζ, Δ_abs, τ, tf)
end

# ===== SWEEP UTILITY =====

"""Sweep over P to find the minimum-time protocol meeting F ≥ F_threshold."""
function sweep_P(; N::Int=1, nmax::Int=20, z_target::Float64=1.0,
                  P_range=1:20, ℓ::Int=1, ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                  init::Symbol=:GHZ, F_threshold::Float64=0.99)
    @printf("=== Sweep P  (N=%d, z=%.2f, F≥%.2f) ===\n", N, z_target, F_threshold)
    @printf("%5s  %12s  %10s\n", "P", "tf (ms)", "Fidelity")

    sb = build_spinboson(N, nmax)
    results = Tuple{Int, Float64, Float64}[]
    for P in P_range
        pp = protocol_params(N, z_target, P, ℓ)
        (; g0, ζ, Δ_abs, τ, tf) = pp

        psi0       = build_initial(N, sb; init)
        psi_target = build_target(ζ, N, sb; init)

        tstops = Float64[]
        for p in 0:(P - 1)
            t0 = 4p * τ
            push!(tstops, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
        end
        push!(tstops, tf)
        unique!(sort!(tstops))

        Hf = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ)
        _, psi_t = timeevolution.schroedinger_dynamic(
            [0.0, tf], psi0, Hf;
            alg=Tsit5(), abstol=1e-10, reltol=1e-10,
            tstops=tstops, maxiters=10_000_000,
        )
        F = abs2(dagger(psi_target) * psi_t[end])
        @printf("%5d  %12.6f  %10.6f\n", P, tf, F)
        push!(results, (P, tf, F))
    end

    passing = filter(r -> r[3] >= F_threshold, results)
    if !isempty(passing)
        best = passing[argmin([r[2] for r in passing])]
        @printf("\nMin-time: P=%d, tf=%.6f ms, F=%.6f\n", best...)
    else
        @printf("\nNo protocol reached F ≥ %.2f in the scanned range.\n", F_threshold)
    end
    return results
end

# ===== EXTERNAL-PULSE HAMILTONIAN =====
#
# A second time-dependent Hamiltonian path used when the four control
# amplitudes (ε₁..ε₄) come from a file (e.g. GRAPE output). The lab-frame
# decomposition is
#       H(t) = ε₁(t)·X̂Jx + ε₂(t)·P̂Jx + ε₃(t)·X̂Jy + ε₄(t)·P̂Jy,
# with X̂ = a + a†, P̂ = i(a† − a). Pulses are sampled on `tlist` and
# linearly interpolated between nodes — matches the GRAPE/ExpProp convention.

"""Closure returning H(t) for `timeevolution.schroedinger_dynamic`,
   given four piecewise-linear control amplitudes on `tlist`."""
function make_H_dynamic_from_pulses(sb,
                                    ε1::Vector{Float64}, ε2::Vector{Float64},
                                    ε3::Vector{Float64}, ε4::Vector{Float64},
                                    tlist::Vector{Float64})
    @assert length(ε1) == length(ε2) == length(ε3) == length(ε4) == length(tlist)
    a, ad = sb.a, sb.ad
    X̂ = a + ad
    P̂ = 1im * (ad - a)
    XJx = X̂ ⊗ sb.Jx
    PJx = P̂ ⊗ sb.Jx
    XJy = X̂ ⊗ sb.Jy
    PJy = P̂ ⊗ sb.Jy

    n = length(tlist)
    t0, tf = tlist[1], tlist[end]

    @inline function interp(arr, t)
        if t <= t0;  return arr[1]
        elseif t >= tf; return arr[end]; end
        i = searchsortedlast(tlist, t)
        i = clamp(i, 1, n - 1)
        α = (t - tlist[i]) / (tlist[i+1] - tlist[i])
        return arr[i] * (1 - α) + arr[i+1] * α
    end

    return function H_at(t, _)
        e1 = interp(ε1, t); e2 = interp(ε2, t)
        e3 = interp(ε3, t); e4 = interp(ε4, t)
        return e1 * XJx + e2 * PJx + e3 * XJy + e4 * PJy
    end
end

"""Run the GHZ-fidelity simulation under control pulses loaded from a JLD2
file produced by ion_GRAPE.jl. The file is expected to hold the keys
`ε1, ε2, ε3, ε4, tlist, T, ζ, N, nmax`. Returns the same kind of
result tuple as `simulate(...)`."""
function simulate_from_pulses(pulse_file::String;
                              init::Symbol=:GHZ,
                              nmax::Union{Nothing,Int}=nothing)
    data = load(pulse_file)
    ε1, ε2, ε3, ε4 = data["ε1"], data["ε2"], data["ε3"], data["ε4"]
    tlist = collect(Float64, data["tlist"])
    T     = Float64(data["T"])
    ζ     = Float64(data["ζ"])
    N     = Int(data["N"])
    nmax  = something(nmax, Int(data["nmax"]))

    sb = build_spinboson(N, nmax)

    @printf("=== Simulation from external pulses (JLD2) ===\n")
    @printf("file = %s\n", pulse_file)
    @printf("N = %d, nmax = %d, init = %s\n", N, nmax, init)
    @printf("ζ = %.6f, T = %.6f ms, nt = %d\n", ζ, T, length(tlist))
    if haskey(data, "F_proc")
        @printf("(stored unitary F_proc from optimisation: %.6f)\n",
                data["F_proc"])
    end

    psi0       = build_initial(N, sb; init)
    psi_target = build_target(ζ, N, sb; init)

    @printf("Target ⟨n⟩ = %.4f (analytic sinh²(ζJ) = %.4f)\n",
            real(expect(sb.n_op, psi_target)), sinh(ζ * N/2)^2)

    Hf = make_H_dynamic_from_pulses(sb, ε1, ε2, ε3, ε4, tlist)

    @printf("\nIntegrating Schrödinger equation under loaded pulses…\n")
    tout, psi_t = timeevolution.schroedinger_dynamic(
        tlist, psi0, Hf;
        alg=Tsit5(), abstol=1e-10, reltol=1e-10,
        tstops=tlist, maxiters=10_000_000,
    )
    @printf("Integration complete. %d saved points.\n", length(tout))

    fidelities = Float64[abs2(dagger(psi_target) * ψ) for ψ in psi_t]
    n_avgs     = Float64[real(expect(sb.n_op, ψ))     for ψ in psi_t]
    norms      = Float64[norm(ψ)                       for ψ in psi_t]

    @printf("\n--- Final state ---\n")
    @printf("Fidelity:  %.8f\n", fidelities[end])
    @printf("⟨n⟩:       %.6f\n", n_avgs[end])
    @printf("‖ψ‖:       %.10f\n", norms[end])

    return (; sb, tout, psi_t, fidelities, n_avgs, norms,
              psi0, psi_target, psi_final=psi_t[end],
              F_final = fidelities[end],
              N, nmax, ζ, T, init,
              # Filler entries so plot_results works without protocol params
              P = 0, ℓ = 0, ϕ1 = 0.0, ϕ2 = 0.0,
              g0 = 0.0, Δ_abs = 0.0, τ = T, tf = T)
end

# ===== UNITARY VERIFICATION =====

"""Build the target unitary of Eq.(5):
   U_target = exp((ζ* â² − ζ â†²) Ĵz / 2).
   For real ζ, this is block-diagonal in the spin basis,
   U_target = Σ_m S(ζm) ⊗ |m⟩⟨m|, since Ĵz|m⟩ = m|m⟩."""
function build_target_unitary(ζ::Float64, N::Int, sb)
    j = N / 2
    dim_s = N + 1
    U = nothing
    for idx_s in 1:dim_s
        m = j - (idx_s - 1)                         # descending m
        S_b  = squeeze(sb.b_fock, ζ * m)            # exp((ζm)(a² − a†²)/2)
        proj = projector(basisstate(sb.b_spin, idx_s))
        term = S_b ⊗ proj
        U = U === nothing ? term : U + term
    end
    return U
end

"""State-independent check that U(T), generated by the stroboscopic protocol,
equals the target Eq.(5). Evolves every basis state |n⟩_b ⊗ |m⟩_s with
n ≤ n_test_max (so the target stays inside the truncation), then forms

    score = Σ_{k∈V} ⟨U_target k | U(T) k⟩ = Tr_V(U_target† U(T)),

and reports

    F_proc = |score|² / d_V²              (process fidelity)
    F_avg  = (d_V·F_proc + 1)/(d_V + 1)   (average gate fidelity)

Both are 1 iff U(T) = U_target on V up to a global phase.
Use a Fock cutoff `nmax` well above the squeezed support of |n_test_max⟩ so
that target leakage is negligible.
"""
function verify_unitary(; N::Int=1, nmax::Int=40, z_target::Float64=0.5,
                         P::Int=5, ℓ::Int=1,
                         ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                         n_test_max::Int=4)
    sb = build_spinboson(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ, tf) = pp

    @printf("=== Unitary verification U(T) vs Eq.(5) ===\n")
    @printf("N = %d, nmax = %d, z_target = %.3f, P = %d, ℓ = %d\n",
            N, nmax, z_target, P, ℓ)
    @printf("ϕ₁ = %.4f, ϕ₂ = %.4f\n", ϕ1, ϕ2)
    @printf("g = 2π × %.3f kHz, |Δ| = 2π × %.3f kHz\n", g0/(2π), Δ_abs/(2π))
    @printf("|ζ| = %.6f, τ = %.6f ms, tf = %.6f ms\n", ζ, τ, tf)

    U_target = build_target_unitary(ζ, N, sb)

    Hf = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ)
    tstops = Float64[]
    for p in 0:(P - 1)
        t0 = 4p * τ
        push!(tstops, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops, tf)
    unique!(sort!(tstops))

    d = length(sb.b_full)
    dim_s = N + 1
    n_test_max = min(n_test_max, nmax)
    test_indices = Int[]
    for idx_b in 1:(n_test_max + 1), idx_s in 1:dim_s
        push!(test_indices, (idx_b - 1) * dim_s + idx_s)
    end
    d_V = length(test_indices)
    @printf("Test subspace V: |n⟩_b ⊗ |m⟩_s with n ≤ %d → dim = %d\n",
            n_test_max, d_V)

    @printf("Integrating %d trajectories…\n", d_V)
    score = ComplexF64(0)
    norm_dev   = 0.0
    target_dev = 0.0     # ‖U_target |k⟩‖² − 1 in the truncated space
    for k in test_indices
        ψ0_data = zeros(ComplexF64, d)
        ψ0_data[k] = 1.0
        ψ0 = Ket(sb.b_full, ψ0_data)

        _, ψt = timeevolution.schroedinger_dynamic(
            [0.0, tf], ψ0, Hf;
            alg=Tsit5(), abstol=1e-10, reltol=1e-10,
            tstops=tstops, maxiters=10_000_000,
        )
        ψ_actual = ψt[end]
        ψ_target = U_target * ψ0

        score      += dagger(ψ_target) * ψ_actual
        norm_dev    = max(norm_dev,   abs(1.0 - norm(ψ_actual)^2))
        target_dev  = max(target_dev, abs(1.0 - norm(ψ_target)^2))
    end

    F_proc = abs2(score) / d_V^2
    F_avg  = (d_V * F_proc + 1) / (d_V + 1)
    @printf("\nProcess fidelity   F_proc = %.8f\n", F_proc)
    @printf("Avg gate fidelity  F_avg  = %.8f\n", F_avg)
    @printf("|score|/d_V               = %.8f\n", abs(score) / d_V)
    @printf("arg(score)                = %+.4f rad  (global phase)\n",
            angle(score))
    @printf("Max ‖U(T)|k⟩‖² − 1        = %.2e   (integrator unitarity)\n",
            norm_dev)
    @printf("Max ‖U_target|k⟩‖² − 1    = %.2e   (target truncation leak)\n",
            target_dev)

    return (; F_proc, F_avg, score, U_target, sb, ζ, tf, n_test_max, d_V)
end

# ===== VISUALIZATION =====

using Plots

"""4-panel diagnostic figure: F(t), ⟨n⟩(t), final P(n) vs target, pulse sequence."""
function plot_results(res; save_path::String="spinboson_sim.png")
    (; tout, fidelities, n_avgs, P, τ, tf, ζ, N, nmax,
       psi_final, psi_target, Δ_abs, g0, ϕ1, ϕ2) = res
    J = N / 2

    default(fontfamily="Computer Modern", titlefontsize=13, guidefontsize=11,
            tickfontsize=9, legendfontsize=9, linewidth=2, dpi=200)

    # Panel 1 — fidelity
    p1 = plot(tout, fidelities,
              xlabel="Time (ms)", ylabel="Fidelity",
              title="F(t) = |⟨ψ_target|ψ(t)⟩|²",
              label="F(t)", color=:blue, legend=:bottomright)
    hline!([1.0], linestyle=:dash, color=:gray, label="", alpha=0.5)
    for p in 1:P
        vline!([4p * τ], linestyle=:dot, color=:lightgray, label="", alpha=0.5)
    end

    # Panel 2 — mean photon number
    n_target_val = sinh(ζ * J)^2
    p2 = plot(tout, n_avgs,
              xlabel="Time (ms)", ylabel="⟨n⟩",
              title="Mean photon number",
              label="⟨n⟩(t)", color=:red, legend=:topleft)
    hline!([n_target_val], linestyle=:dash, color=:gray,
           label=@sprintf("sinh²(ζJ)=%.2f", n_target_val), alpha=0.7)
    for p in 1:P
        vline!([4p * τ], linestyle=:dot, color=:lightgray, label="", alpha=0.5)
    end

    # Panel 3 — Fock-state distribution of the final state vs target
    rho_b_final  = ptrace(psi_final  ⊗ dagger(psi_final),  2)
    rho_b_target = ptrace(psi_target ⊗ dagger(psi_target), 2)
    pn_final  = real.(diag(rho_b_final.data))
    pn_target = real.(diag(rho_b_target.data))
    n_plot_max = something(findfirst(x -> x < 1e-6,
                                     pn_target[2:end] .+ pn_final[2:end]),
                           nmax)
    n_plot_max = min(n_plot_max, nmax)
    n_range = 0:n_plot_max
    p3 = bar(n_range, pn_target[1:n_plot_max+1],
             xlabel="Fock state n", ylabel="P(n)",
             title="Photon number distribution",
             label="Target", color=:gray, alpha=0.4, bar_width=0.8)
    bar!(n_range, pn_final[1:n_plot_max+1],
         label="Final ψ(tf)", color=:blue, alpha=0.6, bar_width=0.4)

    # Panel 4 — pulse sequence
    t_pulse = range(0.0, tf, length=1000)
    Δ_vals = Float64[]; g_vals = Float64[]
    for t in t_pulse
        Δ_eff, _, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ)
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

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 800),
               plot_title=@sprintf("[QO] N=%d, z=%.1f, P=%d, F=%.4f",
                                    N, ζ * J, P, fidelities[end]),
               plot_titlefontsize=14, margin=5Plots.mm)
    savefig(fig, save_path)
    println("\nPlot saved to: $save_path")
    return fig
end

"""4-panel pulse + infidelity figure: Δ(t), ϕ(t), g(t), 1−F(t) on a log scale.
   The trailing sample of g(t) is dropped so the final segment-boundary flip
   isn't drawn."""
function plot_pulse_fidelity(res; save_path::String="spinboson_pulse.png")
    (; tout, fidelities, P, τ, tf, Δ_abs, g0, ϕ1, ϕ2, N, ζ) = res

    default(fontfamily="Computer Modern", titlefontsize=13, guidefontsize=11,
            tickfontsize=9, legendfontsize=9, linewidth=2, dpi=200)

    t_pulse = collect(range(0.0, tf, length=4000))
    Δ_vals = Float64[]; ϕ_vals = Float64[]; g_vals = Float64[]
    for t in t_pulse
        Δe, ϕe, ge = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ)
        push!(Δ_vals, Δe / (2π))
        push!(ϕ_vals, ϕe)
        push!(g_vals, ge / (2π))
    end

    p1 = plot(t_pulse, Δ_vals, xlabel="t (ms)", ylabel="Δ/(2π) [kHz]",
              title="Δ(t)", color=:orange, legend=false)
    p2 = plot(t_pulse, ϕ_vals, xlabel="t (ms)", ylabel="ϕ [rad]",
              title="ϕ(t)", color=:green, legend=false)
    p3 = plot(t_pulse[1:end-1], g_vals[1:end-1],
              xlabel="t (ms)", ylabel="g/(2π) [kHz]",
              title="g(t)", color=:purple, legend=false)

    infid = max.(1 .- fidelities, 1e-16)
    p4 = plot(tout, infid, xlabel="t (ms)", ylabel="1 − F",
              title="1 − F(t)", color=:blue, yscale=:log10, legend=false)

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 800),
               plot_title=@sprintf("N=%d, z=%.1f, P=%d, ℓ=1   F=%.4f",
                                    N, ζ * N/2, P, fidelities[end]),
               plot_titlefontsize=14, margin=5Plots.mm)
    savefig(fig, save_path)
    println("\nPulse plot saved to: $save_path")
    return fig
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^60)
    println("Step 1 — GHZ-state fidelity check  (N=1, z=0.5)")
    println("="^60)
    res = simulate(N=1, nmax=20, z_target=0.5, P=5)
    @printf("\nExpected ⟨n⟩ ≈ sinh²(ζJ) = sinh²(0.5) = %.4f\n", sinh(0.5)^2)
    @printf("Final fidelity (GHZ → S(ζJz)|ψ_GHZ⟩): %.6f\n", res.F_final)
    plot_results(res; save_path="spinboson_QO_N1_z05_GHZ.png")

    println("\n" * "="^60)
    println("Step 2 — State-independent unitary check  U(T) ≟ Eq.(5)")
    println("="^60)
    vres = verify_unitary(N=1, nmax=40, z_target=0.5, P=5, n_test_max=4)
end
