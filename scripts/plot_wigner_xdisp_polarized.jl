# plot_wigner_xdisp_polarized.jl
# Wigner functions of the bosonic marginal ρ_b = Tr_s |ψ(T)⟩⟨ψ(T)| for the
# polarized initial state |0⟩_b ⊗ |+J⟩ propagated under
#   * the squeeze + R(π/2) + x-displace target protocol, and
#   * each GRAPE pulse from the xdisp T_frac sweep
#     (ion_GRAPE_xdisp_pol_controls_Tfrac*.jld2).
#
# Usage: julia --project=. plot_wigner_xdisp_polarized.jl

using QuantumOptics
using OrdinaryDiffEq
using LinearAlgebra
using Printf
using JLD2
using Plots

include(joinpath(@__DIR__, "..", "src", "ion_GRAPE_displace.jl"))   # build_target_ext_xdisp, SpinBoson helpers

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

function wigner_panel(ρ_b, xvec, pvec; title::String="", clim=nothing,
                      colorbar::Bool=false)
    W = wigner(ρ_b, xvec, pvec)
    cmax = clim === nothing ? maximum(abs, W) : clim
    heatmap(xvec, pvec, W';
            c=:RdBu, clims=(-cmax, cmax),
            xlabel="x", ylabel="p", title=title,
            aspect_ratio=:equal, colorbar=colorbar,
            xlims=(xvec[1], xvec[end]), ylims=(pvec[1], pvec[end]))
end

function main(; N::Int=1, nmax::Int=20, z_target::Float64=0.5, P::Int=1, ℓ::Int=1,
                xrange=range(-6.0, 6.0, length=201),
                prange=range(-6.0, 6.0, length=201),
                save_path::String="results/figures/wigner_xdisp_polarized.png")
    sb = build_spinboson(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free   = t_strobo
    ϕ1, ϕ2   = Float64(π), 0.0

    ψ0_ket = build_initial(N, sb; init=:polarized)
    ψ_target = build_target_ext_xdisp(sb, ψ0_ket, Δ_abs, ϕ1, ϕ2, g0, τ,
                                       t_strobo, t_free, P)

    pulses = [
        ("protocol (target)",  nothing),
        ("GRAPE T_frac=1",     "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac1000.jld2"),
        ("GRAPE T_frac=0.9",   "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac900.jld2"),
        ("GRAPE T_frac=0.75",  "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac750.jld2"),
        ("GRAPE T_frac=0.5",   "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac500.jld2"),
        ("GRAPE T_frac=1/3",   "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac333.jld2"),
    ]

    xvec = collect(Float64, xrange)
    pvec = collect(Float64, prange)

    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=9,
            tickfontsize=8, legendfontsize=8, dpi=200)

    rho_bs = Any[]
    titles = String[]
    cmax_global = 0.0
    for (label, path) in pulses
        ψ = path === nothing ? ψ_target : final_state_pulses(sb, path)
        ρ_b = ptrace(ψ ⊗ dagger(ψ), 2)
        n̄ = real(expect(sb.n_op, ψ))
        F = abs2(dagger(ψ_target) * ψ)
        push!(rho_bs, ρ_b)
        push!(titles, @sprintf("%s   ⟨n⟩=%.2f   F=%.4f", label, n̄, F))
        cmax_global = max(cmax_global, maximum(abs, wigner(ρ_b, xvec, pvec)))
        @printf("computed %-22s ⟨n⟩=%.4f   F=%.6f\n", label, n̄, F)
    end

    plts = [wigner_panel(ρ_b, xvec, pvec; title=t, clim=cmax_global,
                         colorbar=(k == 6))
            for (k, (ρ_b, t)) in enumerate(zip(rho_bs, titles))]

    fig = plot(plts...; layout=(2, 3), size=(1500, 950),
               plot_title="Bosonic Wigner W(x, p) — squeeze + R(π/2) + x-displace target  (|0⟩_b ⊗ |+J⟩)",
               plot_titlefontsize=13, margin=4Plots.mm)
    savefig(fig, save_path)
    println("\nSaved: ", save_path)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
