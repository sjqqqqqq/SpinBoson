# protocol_polarized.jl
# Apply the analytic squeeze+displacement protocol (P=1 strobo + 4τ free) to
# ψ₀ = |0⟩_b ⊗ |+J⟩ and produce a single figure with:
#   * top row     pulse sequence Δ(t), ϕ(t), g(t)
#   * bottom row  bosonic-marginal Wigner W(x,p) at t = 0, t_strobo, tf.
#
# Usage: julia --project=. protocol_polarized.jl

using QuantumOptics
using OrdinaryDiffEq
using LinearAlgebra
using Printf
using Plots

include("SpinBoson_sim.jl")

"""Same stroboscopic Hamiltonian as `make_H_dynamic` for t < t_strobo, but in
   the free segment uses `g0·P̂·(Jx+Jy)` instead of `g0·X̂·(Jx+Jy)`, which
   rotates the conditional displacement from the p-axis onto the x-axis."""
function make_H_dynamic_xdisp(sb, Δ_abs::Float64, ϕ1::Float64, ϕ2::Float64,
                              g0::Float64, τ::Float64; t_strobo::Float64)
    aJx, aJy, adJx, adJy = sb.aJx, sb.aJy, sb.adJx, sb.adJy
    a, ad = sb.a, sb.ad
    P̂ = 1im * (ad - a)
    H_free = g0 * (P̂ ⊗ sb.Jx + P̂ ⊗ sb.Jy)
    return function H_at(t, _)
        if t >= t_strobo
            return H_free
        end
        Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ)
        c1 = g_eff * cis(-Δ_eff * t)
        c2 = g_eff * cis(+Δ_eff * t) * cis(-ϕ_eff)
        return c1 * aJx + c2 * aJy + conj(c1) * adJx + conj(c2) * adJy
    end
end

function main(; N::Int=1, nmax::Int=20, z_target::Float64=0.5, P::Int=1, ℓ::Int=1,
                xrange=range(-6.0, 6.0, length=201),
                prange=range(-6.0, 6.0, length=201),
                save_path::String="protocol_polarized.png")
    sb = build_spinboson(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free   = t_strobo
    tf       = t_strobo + t_free
    ϕ1, ϕ2   = Float64(π), 0.0

    @printf("N=%d, z=%.2f, P=%d, ℓ=%d\n", N, z_target, P, ℓ)
    @printf("g0=2π·%.3f kHz, |Δ|=2π·%.3f kHz, τ=%.4f ms\n",
            g0/(2π), Δ_abs/(2π), τ)
    @printf("t_strobo=%.4f, t_free=%.4f, tf=%.4f ms\n", t_strobo, t_free, tf)

    ψ0 = build_initial(N, sb; init=:polarized)

    # Stage 1: stroboscopic squeeze (the original strobo protocol, untouched).
    #   Produces S|0⟩|↑⟩ with the squeezing axis along x (vertical ellipse).
    Hf_strobo = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=Inf)
    tstops_s = Float64[]
    for p in 0:(P-1)
        t0 = 4p * τ
        push!(tstops_s, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops_s, t_strobo)
    unique!(sort!(tstops_s))
    _, ψs = timeevolution.schroedinger_dynamic(
        [0.0, t_strobo], ψ0, Hf_strobo;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=tstops_s, maxiters=10_000_000)
    ψ_after_squeeze = ψs[end]

    # Stage 2: π/2 phase-space rotation, R = exp(−i·π/2·n̂).
    #   In Wigner picture: (x, p) → (−p, x).  The squeezing axis rotates from
    #   x onto p — so the ellipse goes from vertical to horizontal.
    R_boson = exp(dense(-1im * (π/2) * (sb.ad * sb.a)))
    R_full  = R_boson ⊗ one(sb.b_spin)
    ψ_after_rot = R_full * ψ_after_squeeze

    # Stage 3: free-segment displacement along x, H = g0·P̂·(Jx+Jy).
    Hf_free = make_H_dynamic_xdisp(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo)
    _, ψf = timeevolution.schroedinger_dynamic(
        [t_strobo, tf], ψ_after_rot, Hf_free;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=[t_strobo, tf], maxiters=10_000_000)
    ψ_final = ψf[end]

    snapshots = [("t = 0  (|0⟩|↑⟩)", ψ0),
                 (@sprintf("t = t_strobo = %.4f ms  (R·S|0⟩|↑⟩)", t_strobo),
                  ψ_after_rot),
                 (@sprintf("t = tf = %.4f ms  (D·R·S|0⟩|↑⟩)", tf), ψ_final)]

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

    # --- Wigners (shared symmetric colour scale) ---
    rho_bs = [ptrace(ψ ⊗ dagger(ψ), 2) for (_, ψ) in snapshots]
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
               plot_title=@sprintf("Protocol on |0⟩⊗|↑⟩  (N=%d, z=%.2f, ζ=%.3f, P=%d)",
                                    N, z_target, ζ, P),
               plot_titlefontsize=13, margin=4Plots.mm)
    savefig(fig, save_path)
    println("Saved: ", save_path)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
