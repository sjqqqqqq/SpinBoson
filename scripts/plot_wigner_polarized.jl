# plot_wigner_polarized.jl
# Wigner functions of the bosonic marginal ρ_b = Tr_s |ψ(T)⟩⟨ψ(T)| for the
# polarized initial state |0⟩_b ⊗ |+J⟩ propagated under
#   * the analytic squeeze + displacement protocol (target), and
#   * each GRAPE pulse from the T_frac sweep.
#
# Usage: julia --project=. plot_wigner_polarized.jl

using QuantumOptics
using OrdinaryDiffEq
using LinearAlgebra
using Printf
using JLD2
using Plots

include(joinpath(@__DIR__, "..", "src", "SpinBoson_sim.jl"))        # build_spinboson, build_initial, protocol_*

# ----- propagation helpers -----

function final_state_protocol(sb, N, z_target, P, ℓ; init::Symbol=:polarized)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free   = t_strobo
    tf       = t_strobo + t_free
    ϕ1, ϕ2   = Float64(π), 0.0

    ψ0 = build_initial(N, sb; init=init)
    Hf = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo)

    tstops = Float64[]
    for p in 0:(P-1)
        t0 = 4p * τ
        push!(tstops, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops, t_strobo, tf)
    unique!(sort!(tstops))

    _, ψt = timeevolution.schroedinger_dynamic(
        [0.0, tf], ψ0, Hf;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=tstops, maxiters=10_000_000)
    return ψt[end]
end

function final_state_pulses(sb, jld_path::String; init::Symbol=:polarized)
    data = load(jld_path)
    ε1, ε2, ε3, ε4 = data["ε1"], data["ε2"], data["ε3"], data["ε4"]
    tlist = collect(Float64, data["tlist"])
    N     = Int(data["N"])
    ψ0    = build_initial(N, sb; init=init)

    Hf = make_H_dynamic_from_pulses(sb, ε1, ε2, ε3, ε4, tlist)
    _, ψt = timeevolution.schroedinger_dynamic(
        tlist, ψ0, Hf;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=tlist, maxiters=10_000_000)
    return ψt[end]
end

# ----- Wigner plot of the bosonic marginal -----

function wigner_panel(ρ_b, xvec, pvec; title::String="", clim=nothing)
    W = wigner(ρ_b, xvec, pvec)
    cmax = clim === nothing ? maximum(abs, W) : clim
    heatmap(xvec, pvec, W';
            c=:RdBu, clims=(-cmax, cmax),
            xlabel="x", ylabel="p", title=title,
            aspect_ratio=:equal, colorbar=false,
            xlims=(xvec[1], xvec[end]), ylims=(pvec[1], pvec[end]))
end

function main(; N::Int=1, nmax::Int=20, z_target::Float64=0.5, P::Int=1, ℓ::Int=1,
                xrange=range(-6.0, 6.0, length=201),
                prange=range(-6.0, 6.0, length=201))
    sb = build_spinboson(N, nmax)

    pulses = [
        ("protocol",          nothing),
        ("GRAPE T_frac=1",    "results/data/ion_GRAPE_displace_pol_controls_Tfrac1000.jld2"),
        ("GRAPE T_frac=0.9",  "results/data/ion_GRAPE_displace_pol_controls_Tfrac900.jld2"),
        ("GRAPE T_frac=0.75", "results/data/ion_GRAPE_displace_pol_controls_Tfrac750.jld2"),
        ("GRAPE T_frac=0.5",  "results/data/ion_GRAPE_displace_pol_controls_Tfrac500.jld2"),
        ("GRAPE T_frac=1/3",  "results/data/ion_GRAPE_displace_pol_controls_Tfrac333.jld2"),
    ]

    xvec = collect(Float64, xrange)
    pvec = collect(Float64, prange)

    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=9,
            tickfontsize=8, legendfontsize=8, dpi=200)

    # Pass 1: compute ρ_b + per-panel max for shared color scale.
    rho_bs = Any[]
    panel_titles = String[]
    cmax_global = 0.0
    for (label, path) in pulses
        ψ = path === nothing ?
            final_state_protocol(sb, N, z_target, P, ℓ) :
            final_state_pulses(sb, path)
        ρ = ψ ⊗ dagger(ψ)
        ρ_b = ptrace(ρ, 2)
        n̄ = real(expect(sb.n_op, ψ))
        push!(rho_bs, ρ_b)
        push!(panel_titles, @sprintf("%s   ⟨n⟩=%.2f", label, n̄))
        cmax_global = max(cmax_global, maximum(abs, wigner(ρ_b, xvec, pvec)))
        @printf("computed %-22s ⟨n⟩=%.4f\n", label, n̄)
    end

    plts = [wigner_panel(ρ_b, xvec, pvec; title=t, clim=cmax_global)
            for (ρ_b, t) in zip(rho_bs, panel_titles)]

    fig = plot(plts...; layout=(2, 3), size=(1500, 950),
               plot_title="Bosonic Wigner W(x, p)  —  |0⟩_b ⊗ |+J⟩ initial state",
               plot_titlefontsize=13, margin=4Plots.mm)
    savefig(fig, "results/figures/wigner_polarized.png")
    println("\nSaved: results/figures/wigner_polarized.png")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
