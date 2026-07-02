# SpinBoson_test.jl
# Two-spin extension of protocol_polarized.jl.
#
# Hilbert space:  fock ⊗ spin1 ⊗ spin2.
#   * The stroboscopic SQUEEZE stage couples the boson to spin1
#         H(t) = g(t)·a·[Jx₁ e^{−iΔt} + Jy₁ e^{+iΔt} e^{−iϕ}] + h.c.
#     → produces a spin1-conditional squeeze  S(ζ·Jz₁)  on (fock ⊗ spin1).
#   * The free DISPLACEMENT stage couples the boson to spin2
#         H = g0·P̂·(Jx₂ + Jy₂),   P̂ = i(a† − a)
#     → produces a spin2-conditional displacement on (fock ⊗ spin2).
#
# The pulse schedule (pulse_params), the parameter relations (protocol_params),
# and the π/2 boson rotation between stages are taken straight from
# protocol_polarized.jl / SpinBoson_sim.jl.
#
# Usage: julia --project=. SpinBoson_test.jl

using QuantumOptics
using OrdinaryDiffEq
using LinearAlgebra
using Printf
using Plots

include(joinpath(@__DIR__, "..", "src", "SpinBoson_sim.jl"))   # pulse_params, protocol_params

# ===== BASES AND OPERATORS (fock ⊗ spin1 ⊗ spin2) =====

"""Build the three-subsystem basis and the operators that appear in the two
   Hamiltonian stages.  Spin1 ops are used for the squeeze, spin2 ops for the
   displacement; each is embedded with identity on the other spin."""
function build_spinboson2(N::Int, nmax::Int)
    b_fock  = FockBasis(nmax)
    b_spin1 = SpinBasis(N // 2)
    b_spin2 = SpinBasis(N // 2)
    b_full  = b_fock ⊗ b_spin1 ⊗ b_spin2

    a   = destroy(b_fock)
    ad  = create(b_fock)
    Is1 = one(b_spin1)
    Is2 = one(b_spin2)

    # Spin-1/2 angular-momentum operators on each spin (sigmax = 2Jx, etc.).
    Jx1 = sigmax(b_spin1) / 2; Jy1 = sigmay(b_spin1) / 2; Jz1 = sigmaz(b_spin1) / 2
    Jx2 = sigmax(b_spin2) / 2; Jy2 = sigmay(b_spin2) / 2; Jz2 = sigmaz(b_spin2) / 2

    # Squeeze stage couples a/a† to spin1 (identity on spin2).
    aJx1  = a  ⊗ Jx1 ⊗ Is2
    aJy1  = a  ⊗ Jy1 ⊗ Is2
    adJx1 = ad ⊗ Jx1 ⊗ Is2
    adJy1 = ad ⊗ Jy1 ⊗ Is2

    # Displacement stage couples P̂ = i(a† − a) to spin2 (identity on spin1).
    P̂    = 1im * (ad - a)
    PJx2 = P̂ ⊗ Is1 ⊗ Jx2
    PJy2 = P̂ ⊗ Is1 ⊗ Jy2

    n_op = (ad * a) ⊗ Is1 ⊗ Is2

    return (; b_fock, b_spin1, b_spin2, b_full, a, ad, Is1, Is2,
              Jx1, Jy1, Jz1, Jx2, Jy2, Jz2,
              aJx1, aJy1, adJx1, adJy1, PJx2, PJy2, n_op)
end

"""Initial state |0⟩_b ⊗ |s⟩₁ ⊗ |s⟩₂ with spins = :up (|+J⟩, default) or
   :down (|−J⟩)."""
function build_initial2(sb; spins::Symbol=:up)
    vac   = fockstate(sb.b_fock, 0)
    spin1 = spins === :down ? spindown(sb.b_spin1) : spinup(sb.b_spin1)
    spin2 = spins === :down ? spindown(sb.b_spin2) : spinup(sb.b_spin2)
    return vac ⊗ spin1 ⊗ spin2
end

# ===== HAMILTONIAN STAGES =====

"""Stroboscopic squeeze Hamiltonian (boson ↔ spin1)."""
function make_H_squeeze(sb, Δ_abs::Float64, ϕ1::Float64, ϕ2::Float64,
                        g0::Float64, τ::Float64; t_strobo::Float64=Inf)
    aJx1, aJy1, adJx1, adJy1 = sb.aJx1, sb.aJy1, sb.adJx1, sb.adJy1
    return function H_at(t, _)
        Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo)
        c1 = g_eff * cis(-Δ_eff * t)                   # coeff of a⊗Jx₁
        c2 = g_eff * cis(+Δ_eff * t) * cis(-ϕ_eff)     # coeff of a⊗Jy₁
        return c1 * aJx1 + c2 * aJy1 + conj(c1) * adJx1 + conj(c2) * adJy1
    end
end

"""Free-segment displacement Hamiltonian (boson ↔ spin2), H = g0·P̂·(Jx₂+Jy₂)."""
function make_H_disp(sb, g0::Float64)
    H_free = g0 * (sb.PJx2 + sb.PJy2)
    return function H_at(t, _)
        return H_free
    end
end

# ===== MAIN =====

function main(; N::Int=1, nmax::Int=20, z_target::Float64=0.5, P::Int=1, ℓ::Int=1,
                xrange=range(-6.0, 6.0, length=201),
                prange=range(-6.0, 6.0, length=201),
                save_path::String="results/figures/spinboson_test.png")
    sb = build_spinboson2(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free   = t_strobo
    tf       = t_strobo + t_free
    ϕ1, ϕ2   = Float64(π), 0.0

    @printf("=== Two-spin squeeze(spin1) + displacement(spin2) ===\n")
    @printf("N=%d, z=%.2f, P=%d, ℓ=%d, dim(H)=%d\n",
            N, z_target, P, ℓ, length(sb.b_full))
    @printf("g0=2π·%.3f kHz, |Δ|=2π·%.3f kHz, τ=%.4f ms, |ζ|=%.4f\n",
            g0/(2π), Δ_abs/(2π), τ, ζ)
    @printf("t_strobo=%.4f, t_free=%.4f, tf=%.4f ms\n", t_strobo, t_free, tf)

    ψ0 = build_initial2(sb)

    # Stage 1: stroboscopic squeeze conditioned on spin1.  S(ζ·Jz₁)|0⟩.
    Hf_sq = make_H_squeeze(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=Inf)
    tstops_s = Float64[]
    for p in 0:(P - 1)
        t0 = 4p * τ
        push!(tstops_s, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops_s, t_strobo)
    unique!(sort!(tstops_s))
    _, ψs = timeevolution.schroedinger_dynamic(
        [0.0, t_strobo], ψ0, Hf_sq;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=tstops_s, maxiters=10_000_000)
    ψ_after_squeeze = ψs[end]

    # Stage 2: π/2 phase-space rotation R = exp(−i·π/2·n̂) on the boson.
    R_boson = exp(dense(-1im * (π/2) * (sb.ad * sb.a)))
    R_full  = R_boson ⊗ sb.Is1 ⊗ sb.Is2
    ψ_after_rot = R_full * ψ_after_squeeze

    # Stage 3: free-segment displacement conditioned on spin2.
    Hf_d = make_H_disp(sb, g0)
    _, ψf = timeevolution.schroedinger_dynamic(
        [t_strobo, tf], ψ_after_rot, Hf_d;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=[t_strobo, tf], maxiters=10_000_000)
    ψ_final = ψf[end]

    @printf("\n⟨n⟩: t=0 → %.4f, after squeeze+rot → %.4f, final → %.4f\n",
            real(expect(sb.n_op, ψ0)),
            real(expect(sb.n_op, ψ_after_rot)),
            real(expect(sb.n_op, ψ_final)))

    snapshots = [("t = 0  (|0⟩|↑↑⟩)", ψ0),
                 (@sprintf("t = t_strobo = %.4f ms  (R·S₁|0⟩|↑↑⟩)", t_strobo),
                  ψ_after_rot),
                 (@sprintf("t = tf = %.4f ms  (D₂·R·S₁|0⟩|↑↑⟩)", tf), ψ_final)]

    xvec = collect(Float64, xrange)
    pvec = collect(Float64, prange)

    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=10,
            tickfontsize=8, legendfontsize=8, linewidth=1.4, dpi=200)

    # --- protocol pulses ---
    tgrid = collect(range(0.0, tf, length=4000))
    Δ_vals = Float64[]; ϕ_vals = Float64[]; g_vals = Float64[]
    for t in tgrid
        Δe, ϕe, ge = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo)
        push!(Δ_vals, Δe / (2π)); push!(ϕ_vals, ϕe); push!(g_vals, ge / (2π))
    end
    p_Δ = plot(tgrid, Δ_vals; xlabel="t (ms)", ylabel="Δ/(2π) [kHz]",
               title="Δ(t)", color=:orange, legend=false)
    vline!(p_Δ, [t_strobo]; color=:black, linestyle=:dot, alpha=0.5)
    p_ϕ = plot(tgrid, ϕ_vals; xlabel="t (ms)", ylabel="ϕ [rad]",
               title="ϕ(t)", color=:green, legend=false)
    vline!(p_ϕ, [t_strobo]; color=:black, linestyle=:dot, alpha=0.5)
    p_g = plot(tgrid[1:end-1], g_vals[1:end-1]; xlabel="t (ms)",
               ylabel="g/(2π) [kHz]", title="g(t)", color=:purple, legend=false)
    vline!(p_g, [t_strobo]; color=:black, linestyle=:dot, alpha=0.5)

    # --- Wigners of the bosonic marginal (trace out both spins: subsystems 2,3) ---
    rho_bs = [ptrace(ψ ⊗ dagger(ψ), [2, 3]) for (_, ψ) in snapshots]
    Ws = [wigner(ρ, xvec, pvec) for ρ in rho_bs]
    cmax = maximum(maximum.(abs, Ws))

    p_W = Plots.Plot[]
    for (k, ((title, ψ), W)) in enumerate(zip(snapshots, Ws))
        n̄ = real(expect(sb.n_op, ψ))
        plt = heatmap(xvec, pvec, W';
                      c=:RdBu, clims=(-cmax, cmax),
                      xlabel="x", ylabel="p",
                      title=@sprintf("%s   ⟨n⟩=%.2f", title, n̄),
                      aspect_ratio=:equal,
                      colorbar=(k == 3),
                      xlims=(xvec[1], xvec[end]),
                      ylims=(pvec[1], pvec[end]))
        push!(p_W, plt)
    end

    fig = plot(p_Δ, p_ϕ, p_g, p_W...; layout=(2, 3), size=(1500, 900),
               plot_title=@sprintf("Squeeze(spin1)+Displace(spin2) on |0⟩⊗|↑↑⟩  (N=%d, z=%.2f, ζ=%.3f, P=%d)",
                                    N, z_target, ζ, P),
               plot_titlefontsize=13, margin=4Plots.mm)
    savefig(fig, save_path)
    println("Saved: ", save_path)
    return (; sb, ψ0, ψ_after_squeeze, ψ_after_rot, ψ_final, fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
