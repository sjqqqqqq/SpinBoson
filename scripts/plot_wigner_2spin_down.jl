# plot_wigner_2spin_down.jl
# Side-by-side Wigner functions of the final boson marginal (spins traced out)
# for the down-down / no-rotation configuration of ion_GRAPE_2spin.jl:
#
#   left  — T_frac = 1.0, analytic two-stage pulse (GRAPE leaves it unchanged,
#           F ≈ 0.9999)
#   right — T_frac = 0.5, GRAPE-optimised controls loaded from
#           results/data/ion_GRAPE_2spin_down_controls_Tfrac05.jld2
#
# Usage: julia --project=. scripts/plot_wigner_2spin_down.jl

include(joinpath(@__DIR__, "ion_GRAPE_2spin.jl"))

"""Propagate `controls` on the problem in `pd`; return the boson marginal of
   the final state, the transfer fidelity, and ⟨n⟩."""
function final_boson_marginal(pd, controls)
    gen0 = pd.problem.trajectories[1].generator
    H = substitute(gen0, IdDict(zip(get_controls(gen0), controls)))
    ψf_vec = propagate(pd.init_state, H, pd.tlist; method=ExpProp)
    F  = abs2(dot(pd.target_state, ψf_vec))
    ψf = Ket(pd.sb.b_full, ψf_vec)
    ρb = ptrace(ψf ⊗ dagger(ψf), [2, 3])
    n̄  = real(expect(pd.sb.ad * pd.sb.a, ρb))
    return ρb, F, n̄
end

function main_wigner(; xrange=range(-6.0, 6.0, length=201),
                       prange=range(-6.0, 6.0, length=201),
                       controls_path::String="results/data/ion_GRAPE_2spin_down_controls_Tfrac05.jld2",
                       save_path::String="results/figures/wigner_2spin_down_T10_vs_T05.png")
    # Case 1: analytic two-stage pulse at full T.
    pd1 = build_problem2(T_frac=1.0)
    ρ1, F1, n1 = final_boson_marginal(pd1, initial_controls2(pd1))

    # Case 2: GRAPE-optimised controls at half T.
    data = load(controls_path)
    pd2 = build_problem2(T_frac=0.5)
    @assert length(pd2.tlist) == length(data["tlist"]) &&
            isapprox(pd2.tlist[end], data["tlist"][end]; rtol=1e-12)
    ρ2, F2, n2 = final_boson_marginal(pd2, data["controls"])

    @printf("T_frac=1.0 (analytic):  F = %.6f, ⟨n⟩ = %.4f\n", F1, n1)
    @printf("T_frac=0.5 (GRAPE):     F = %.6f, ⟨n⟩ = %.4f\n", F2, n2)

    xvec = collect(Float64, xrange)
    pvec = collect(Float64, prange)
    W1 = wigner(ρ1, xvec, pvec)
    W2 = wigner(ρ2, xvec, pvec)
    cmax = max(maximum(abs, W1), maximum(abs, W2))

    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=10,
            tickfontsize=8, linewidth=1.4, dpi=200)

    panels = [(W1, @sprintf("analytic pulse, T = %.4f ms   F = %.4f, <n> = %.2f",
                            pd1.T, F1, n1), false),
              (W2, @sprintf("GRAPE, T = %.4f ms   F = %.4f, <n> = %.2f",
                            pd2.T, F2, n2), true)]
    plts = [heatmap(xvec, pvec, W';
                    c=:RdBu, clims=(-cmax, cmax),
                    xlabel="x", ylabel="p", title=title,
                    aspect_ratio=:equal, colorbar=cbar,
                    xlims=(xvec[1], xvec[end]), ylims=(pvec[1], pvec[end]))
            for (W, title, cbar) in panels]

    fig = plot(plts...; layout=(1, 2), size=(1250, 560),
               plot_title="Final boson Wigner (spins traced out):  D2(cond)*S1(cond) on |0>|dd>",
               plot_titlefontsize=12, margin=5Plots.mm)
    savefig(fig, save_path)
    println("Saved: ", save_path)
    return (; fig, ρ1, ρ2, F1, F2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_wigner()
end
