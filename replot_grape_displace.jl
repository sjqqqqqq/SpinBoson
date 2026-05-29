# replot_grape_displace.jl
# Re-render the GRAPE T_frac sweep pulse figures with:
#   * x-axis normalised to t/T_ref ∈ [0, T_frac]  (T_ref = full reference
#     protocol t_strobo + t_free; T_frac = T_GRAPE / T_ref)
#   * vertical dashed marker at t/T_ref = T_frac/2  (midpoint of the GRAPE
#     horizon in T_ref units).
#
# Reads the JLD2 files saved by ion_GRAPE_displace.jl's sweep_T_frac.
#
# Usage: julia --project=. replot_grape_displace.jl

using JLD2
using LinearAlgebra
using Printf
using Plots

include("ion_GRAPE_displace.jl")   # protocol_amplitudes_ext, protocol_params

function rebuild_initial_guess(data)
    tlist     = collect(Float64, data["tlist"])
    T_total   = Float64(data["T"])
    t_strobo  = Float64(data["t_strobo"])
    t_free    = Float64(data["t_free"])
    T_full    = t_strobo + t_free
    α         = T_total / T_full
    N         = Int(data["N"])
    nmax      = Int(data["nmax"])
    ζ         = Float64(data["ζ"])
    # Recover protocol parameters consistent with how the run was set up
    # (z_target = ζ·N/2, P deduced from t_strobo = 4Pτ later if needed).
    z_target  = ζ * N / 2
    P         = 1                                # sweep used P=1 throughout
    pp        = protocol_params(N, z_target, P, 1)
    g0, Δ_abs, τ = pp.g0, pp.Δ_abs, pp.τ
    ϕ1, ϕ2    = Float64(π), 0.0

    init = [Float64[protocol_amplitudes_ext(t / α, Δ_abs, ϕ1, ϕ2, g0, τ,
                                            t_strobo)[k] for t in tlist]
            for k in 1:4]
    return tlist, T_total, init
end

function replot_one(jld_path::String; save_path::String=replace(jld_path,
                                                                ".jld2" => "_norm.png"))
    data       = load(jld_path)
    tlist, T_total, init = rebuild_initial_guess(data)
    opt        = [data["ε1"], data["ε2"], data["ε3"], data["ε4"]]
    T_frac     = Float64(get(data, "T_frac", NaN))
    F_opt      = Float64(data["F"])
    t_strobo   = Float64(data["t_strobo"])
    t_free     = Float64(data["t_free"])
    T_ref      = t_strobo + t_free      # full reference protocol horizon
    t_norm     = tlist ./ T_ref         # x ∈ [0, T_frac] in units of T_ref

    labels = ["ε1 : X⊗Jx", "ε2 : P⊗Jx", "ε3 : X⊗Jy", "ε4 : P⊗Jy"]
    colors = [:blue, :red, :green, :orange]

    default(fontfamily="Computer Modern", titlefontsize=12, guidefontsize=10,
            tickfontsize=8, legendfontsize=8, linewidth=1.2, dpi=200)

    plts = Plots.Plot[]
    for k in 1:4
        plt = plot(t_norm, init[k] ./ (2π);
                   label="protocol",
                   color=:gray, linestyle=:dash,
                   xlabel="t / T_ref", ylabel=labels[k] * " /(2π)  [kHz]",
                   legend=:bottomright, xlims=(0.0, T_frac))
        plot!(plt, t_norm, opt[k] ./ (2π); label="GRAPE", color=colors[k])
        vline!(plt, [T_frac / 2]; color=:black, linestyle=:dot, alpha=0.6, label="")
        push!(plts, plt)
    end

    fig = plot(plts...; layout=(2, 2), size=(1100, 750),
               plot_title=@sprintf("GRAPE squeeze+displacement   T_frac=%.4f   T=%.4f ms   F=%.5f",
                                    T_frac, T_total, F_opt),
               plot_titlefontsize=12, margin=4Plots.mm)
    savefig(fig, save_path)
    println("Saved: $save_path")
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    # GHZ files
    for tag in ("Tfrac900", "Tfrac750", "Tfrac500", "Tfrac333")
        replot_one("ion_GRAPE_displace_controls_$(tag).jld2";
                   save_path="ion_GRAPE_displace_pulses_$(tag)_norm.png")
    end
    # Polarized files
    for tag in ("Tfrac1000", "Tfrac900", "Tfrac750", "Tfrac500", "Tfrac333")
        replot_one("ion_GRAPE_displace_pol_controls_$(tag).jld2";
                   save_path="ion_GRAPE_displace_pol_pulses_$(tag)_norm.png")
    end
end
