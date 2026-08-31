# spinboson_protocol.jl
# Spin-boson simulation of the analytic protocol from arXiv:2510.25870
# (Bond et al.), Hamiltonian Eq.(23), pulse sequence Fig.4(c).
#
#     ψ0 = |0⟩_b ⊗ |↓⟩_1 ⊗ |↓⟩_2
#
# evolved through the two analytic stages, then shown as a Wigner function.
#
# Hilbert space: fock ⊗ spin1 ⊗ spin2.
#
#   Stage 1 — stroboscopic SQUEEZE, boson ↔ spin1:
#       H(t) = g(t)·a·[Jx1·e^{−iΔ(t)t} + Jy1·e^{+iΔ(t)t}·e^{−iϕ(t)}] + h.c.
#     Over t_strobo this generates the spin1-conditional squeeze S(ζ·Jz1).
#     With spin1 in |↓⟩ that is S(−ζ/2): the squeeze axis comes out rotated 90°
#     relative to the |↑⟩ case, which is exactly the orientation the next stage
#     wants — so no π/2 boson rotation is needed between the stages (set
#     `rotate=true` to insert one and see the |↑↑⟩-style alignment instead).
#
#   Stage 2 — free DISPLACEMENT, boson ↔ spin2:
#       H = g0·P̂·(Jx2 + Jy2),   P̂ = i(a† − a)
#     `pulse_params` holds (Δ=0, ϕ=0, g=+g0) here, so the drive is constant.
#     This displaces the boson conditionally on spin2, entangling the two.
#
# Everything needed is in this file — it does not include anything else.
#
# Usage:
#   julia --project=. spinboson_protocol.jl
#
#   julia --project=. -i -e 'include("spinboson_protocol.jl")'
#     res = run_protocol(z_target=0.5, P=1)
#     plot_wigner(res; save_path="results/out.png")

using QuantumOptics
using OrdinaryDiffEq
using LinearAlgebra
using Printf
using Plots

# ===== PROTOCOL PARAMETERS =====

"""Derive (g0, ζ, Δ_abs, τ, tf) from (N, z_target, P, ℓ) — Eq.(31) of the paper.

Angular frequencies in rad/ms, times in ms."""
function protocol_params(N::Int, z_target::Float64, P::Int, ℓ::Int)
    g0    = 2π * 5.0 / sqrt(N)          # 2π × 5 kHz / √N
    ζ     = 2 * z_target / N
    Δ_abs = sqrt(16π * g0^2 * ℓ * P / ζ)
    τ     = 2π * ℓ / Δ_abs
    tf    = 4 * P * τ
    return (; g0, ζ, Δ_abs, τ, tf)
end

"""Stroboscopic pulse — 4 segments per cycle, each of duration τ = 2πℓ/|Δ|.

The signs of Δ and g flip on a (+,+), (−,+), (−,−), (+,−) schedule and the
phase alternates between ϕ1 and ϕ2. For t ≥ t_strobo the controls are held at
(Δ=0, ϕ=0, g=+g0) — that is the free displacement stage."""
@inline function pulse_params(t::Float64, Δ::Float64, ϕ1::Float64, ϕ2::Float64,
                              g0::Float64, τ::Float64;
                              t_strobo::Float64=Inf)
    if t >= t_strobo
        return (0.0, 0.0, +g0)
    end
    t_mod = mod(t, 4τ)
    if t_mod < τ
        return (+Δ, ϕ1, +g0)       # segment 1
    elseif t_mod < 2τ
        return (-Δ, ϕ2, +g0)       # segment 2
    elseif t_mod < 3τ
        return (-Δ, ϕ2, -g0)       # segment 3 = echo of segment 2 with g → −g
    else
        return (+Δ, ϕ1, -g0)       # segment 4 = echo of segment 1 with g → −g
    end
end

# ===== BASES AND OPERATORS (fock ⊗ spin1 ⊗ spin2) =====

"""Build the three-subsystem basis and the operators the two stages need.

Spin1 operators carry the squeeze, spin2 the displacement; each is embedded
with the identity on the other spin."""
function build_system(N::Int, nmax::Int)
    b_fock  = FockBasis(nmax)
    b_spin1 = SpinBasis(N // 2)
    b_spin2 = SpinBasis(N // 2)
    b_full  = b_fock ⊗ b_spin1 ⊗ b_spin2

    a   = destroy(b_fock)
    ad  = create(b_fock)
    Is1 = one(b_spin1)
    Is2 = one(b_spin2)

    # Angular-momentum operators on each spin (sigmax = 2Jx, etc.).
    Jx1 = sigmax(b_spin1) / 2; Jy1 = sigmay(b_spin1) / 2
    Jx2 = sigmax(b_spin2) / 2; Jy2 = sigmay(b_spin2) / 2

    # Squeeze stage: a, a† coupled to spin1.
    aJx1  = a  ⊗ Jx1 ⊗ Is2
    aJy1  = a  ⊗ Jy1 ⊗ Is2
    adJx1 = ad ⊗ Jx1 ⊗ Is2
    adJy1 = ad ⊗ Jy1 ⊗ Is2

    # Displacement stage: P̂ = i(a† − a) coupled to spin2.
    P̂    = 1im * (ad - a)
    PJx2 = P̂ ⊗ Is1 ⊗ Jx2
    PJy2 = P̂ ⊗ Is1 ⊗ Jy2

    n_op = (ad * a) ⊗ Is1 ⊗ Is2

    return (; b_fock, b_spin1, b_spin2, b_full, a, ad, Is1, Is2,
              aJx1, aJy1, adJx1, adJy1, PJx2, PJy2, n_op)
end

"""|0⟩_b ⊗ |s⟩_1 ⊗ |s⟩_2, with `spins = :down` (default) or `:up`."""
function build_initial(sb; spins::Symbol=:down)
    vac   = fockstate(sb.b_fock, 0)
    spin1 = spins === :down ? spindown(sb.b_spin1) : spinup(sb.b_spin1)
    spin2 = spins === :down ? spindown(sb.b_spin2) : spinup(sb.b_spin2)
    return vac ⊗ spin1 ⊗ spin2
end

# ===== HAMILTONIAN STAGES =====

"""Stage-1 Hamiltonian closure for `timeevolution.schroedinger_dynamic`."""
function make_H_squeeze(sb, Δ_abs::Float64, ϕ1::Float64, ϕ2::Float64,
                        g0::Float64, τ::Float64; t_strobo::Float64=Inf)
    aJx1, aJy1, adJx1, adJy1 = sb.aJx1, sb.aJy1, sb.adJx1, sb.adJy1
    return function H_at(t, _)
        Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo)
        c1 = g_eff * cis(-Δ_eff * t)                   # coefficient of a⊗Jx1
        c2 = g_eff * cis(+Δ_eff * t) * cis(-ϕ_eff)     # coefficient of a⊗Jy1
        return c1 * aJx1 + c2 * aJy1 + conj(c1) * adJx1 + conj(c2) * adJy1
    end
end

"""Stage-2 Hamiltonian closure, H = g0·P̂·(Jx2 + Jy2) — constant in time."""
function make_H_disp(sb, g0::Float64)
    H_free = g0 * (sb.PJx2 + sb.PJy2)
    return (_, _) -> H_free
end

# ===== SIMULATION =====

"""Run the analytic protocol on |0⟩|↓↓⟩ and return the states at each stage.

`t_free_frac` scales the displacement stage relative to the stroboscopic one
(1.0 means equal durations, matching the exported hardware pulse). `rotate`
inserts the π/2 boson rotation between the stages, which the |↑↑⟩ start needs
and the |↓↓⟩ start does not."""
function run_protocol(; N::Int=1, nmax::Int=60, z_target::Float64=0.5,
                        P::Int=1, ℓ::Int=1, spins::Symbol=:down,
                        t_free_frac::Float64=1.0, rotate::Bool=false,
                        ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                        verbose::Bool=true)
    sb = build_system(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ) = pp

    t_strobo = pp.tf
    t_free   = t_free_frac * t_strobo
    tf       = t_strobo + t_free

    if verbose
        @printf("=== Analytic protocol on |0>|%s> ===\n",
                spins === :down ? "dd" : "uu")
        @printf("N = %d, nmax = %d, z = %.3f, P = %d, l = %d, dim(H) = %d\n",
                N, nmax, z_target, P, ℓ, length(sb.b_full))
        @printf("g0 = 2pi x %.3f kHz, |D| = 2pi x %.3f kHz, tau = %.4f ms, zeta = %.4f\n",
                g0 / (2π), Δ_abs / (2π), τ, ζ)
        @printf("t_strobo = %.4f ms, t_free = %.4f ms, T = %.4f ms\n",
                t_strobo, t_free, tf)
        rotate && println("(inserting the pi/2 boson rotation between stages)")
    end

    ψ0 = build_initial(sb; spins=spins)

    # --- Stage 1: stroboscopic squeeze conditioned on spin1. ---
    # The solver is handed the segment boundaries as tstops so it does not
    # smooth across the pulse discontinuities.
    tstops = Float64[]
    for p in 0:(P - 1)
        t0 = 4p * τ
        push!(tstops, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops, t_strobo)
    unique!(sort!(tstops))

    H_sq = make_H_squeeze(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=Inf)
    _, ψs = timeevolution.schroedinger_dynamic(
        [0.0, t_strobo], ψ0, H_sq;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=tstops, maxiters=10_000_000)
    ψ_squeeze = ψs[end]

    # --- Optional π/2 phase-space rotation R = exp(−i·(π/2)·n̂) on the boson. ---
    ψ_mid = ψ_squeeze
    if rotate
        R = exp(dense(-1im * (π / 2) * (sb.ad * sb.a))) ⊗ sb.Is1 ⊗ sb.Is2
        ψ_mid = R * ψ_squeeze
    end

    # --- Stage 2: free displacement conditioned on spin2. ---
    H_d = make_H_disp(sb, g0)
    _, ψf = timeevolution.schroedinger_dynamic(
        [t_strobo, tf], ψ_mid, H_d;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=[t_strobo, tf], maxiters=10_000_000)
    ψ_final = ψf[end]

    n̄ = ψ -> real(expect(sb.n_op, ψ))
    # A Fock cutoff that is too low shows up as population piling into the top
    # levels, so check it rather than trusting nmax.
    tail = sum(abs2, ψ_final.data[end - 2 * (N + 1)^2 + 1:end])

    if verbose
        @printf("\n<n>: t=0 -> %.4f, after squeeze -> %.4f, final -> %.4f\n",
                n̄(ψ0), n̄(ψ_mid), n̄(ψ_final))
        # The spin-conditional squeeze is S(zeta*Jz1); on |down> that is
        # S(-zeta/2), so an ideal stage would leave sinh^2(zeta/2) phonons.
        # Finite P undershoots this — that gap is the protocol's discretization
        # error, not a bug.
        @printf("     ideal squeeze <n> = sinh^2(zeta/2) = %.4f  (P=%d undershoot: %.1f%%)\n",
                sinh(ζ / 2)^2, P,
                100 * (1 - n̄(ψ_squeeze) / sinh(ζ / 2)^2))
        @printf("norm: %.12f, population in top 2 Fock levels: %.2e",
                norm(ψ_final), tail)
        println(tail > 1e-6 ? "  <-- raise nmax" : "  (cutoff ok)")
    end

    return (; sb, pp, ψ0, ψ_squeeze, ψ_mid, ψ_final,
              t_strobo, t_free, tf, N, nmax, z_target, P, ℓ, ϕ1, ϕ2,
              spins, rotate, n_tail=tail)
end

# ===== WIGNER PLOT =====

"""Bosonic Wigner function of `ψ` with both spins traced out."""
function boson_wigner(ψ, xvec, pvec)
    ρ = ptrace(ψ ⊗ dagger(ψ), [2, 3])
    return wigner(ρ, xvec, pvec)
end

"""Wigner snapshots at t = 0, after the squeeze stage, and at the end, with the
pulse sequence that produced them.

Plot text is deliberately ASCII: the Computer Modern font used across the deck
has no subscript or arrow glyphs, and missing ones render as boxes."""
function plot_wigner(res; xrange=range(-8.0, 8.0, length=241),
                          prange=range(-8.0, 8.0, length=241),
                          save_path::String="results/spinboson_wigner.png")
    (; sb, pp, ψ0, ψ_mid, ψ_final, t_strobo, tf) = res
    (; g0, ζ, Δ_abs, τ) = pp

    xvec = collect(Float64, xrange)
    pvec = collect(Float64, prange)

    spin_txt = res.spins === :down ? "dd" : "uu"
    snapshots = [
        (@sprintf("t = 0   vacuum |0>|%s>", spin_txt), ψ0),
        (@sprintf("t = t_strobo = %.3f ms   squeezed", t_strobo), ψ_mid),
        (@sprintf("t = T = %.3f ms   + displaced", tf), ψ_final),
    ]

    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=10,
            tickfontsize=8, legendfontsize=8, linewidth=1.4, dpi=200)

    # --- the pulse sequence ---
    tgrid = collect(range(0.0, tf, length=4000))
    Δ_vals = Float64[]; ϕ_vals = Float64[]; g_vals = Float64[]
    for t in tgrid
        Δe, ϕe, ge = pulse_params(t, Δ_abs, res.ϕ1, res.ϕ2, g0, τ;
                                  t_strobo=t_strobo)
        push!(Δ_vals, Δe / (2π)); push!(ϕ_vals, ϕe); push!(g_vals, ge / (2π))
    end
    mark!(p) = vline!(p, [t_strobo]; color=:black, linestyle=:dot, alpha=0.5)
    p_Δ = mark!(plot(tgrid, Δ_vals; xlabel="t (ms)", ylabel="D/(2pi) [kHz]",
                     title="detuning D(t)", color=:orange, legend=false))
    p_ϕ = mark!(plot(tgrid, ϕ_vals; xlabel="t (ms)", ylabel="phi [rad]",
                     title="phase phi(t)", color=:green, legend=false))
    p_g = mark!(plot(tgrid[1:end-1], g_vals[1:end-1]; xlabel="t (ms)",
                     ylabel="g/(2pi) [kHz]", title="coupling g(t)",
                     color=:purple, legend=false))

    # --- Wigner panels ---
    Ws = [boson_wigner(ψ, xvec, pvec) for (_, ψ) in snapshots]

    # Each panel is scaled to its own peak. A shared scale would be set by the
    # vacuum (max W = 1/pi) and leave the squeezed and displaced states nearly
    # invisible; the colorbars carry the absolute numbers. The scale stays
    # symmetric about zero so any negativity would read as red — there is none
    # here, because tracing out spin2 leaves a classical mixture of the two
    # conditional displacements rather than a cat.
    p_W = Plots.Plot[]
    for ((title, ψ), W) in zip(snapshots, Ws)
        n̄ = real(expect(sb.n_op, ψ))
        cmax = maximum(abs, W)
        push!(p_W, heatmap(xvec, pvec, W';
                           c=:RdBu, clims=(-cmax, cmax),
                           xlabel="x", ylabel="p",
                           title=@sprintf("%s\n<n> = %.2f", title, n̄),
                           aspect_ratio=:equal, colorbar=true,
                           xlims=(xvec[1], xvec[end]),
                           ylims=(pvec[1], pvec[end])))
    end

    # Height chosen so the square heatmaps roughly fill their cells once the
    # colorbar and title are accounted for — otherwise they float in whitespace.
    fig = plot(p_Δ, p_ϕ, p_g, p_W...;
               layout=grid(2, 3; heights=[0.30, 0.70]), size=(1650, 820),
               plot_title=@sprintf("Analytic protocol on |0>|%s>   N=%d, z=%.2f, zeta=%.3f, P=%d",
                                   spin_txt, res.N, res.z_target, ζ, res.P),
               plot_titlefontsize=14, leftmargin=6Plots.mm,
               bottommargin=5Plots.mm, topmargin=3Plots.mm)

    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: ", save_path)
    return fig
end

# ===== RUN =====

if abspath(PROGRAM_FILE) == @__FILE__
    res = run_protocol()
    plot_wigner(res)
end
