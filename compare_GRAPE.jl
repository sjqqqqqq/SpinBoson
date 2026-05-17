# compare_GRAPE.jl
# Compare ion_GRAPE_state_controls.jld2 (1 trajectory: |GHZ⟩)
# with     ion_GRAPE_2.jld2              (2 trajectories: |↑⟩, |↓⟩).
# Loads both, prints summary, and overlays the four ε's in one figure.

using JLD2, Plots, Printf, LinearAlgebra

function load_run(path)
    d = load(path)
    return (; ε = [d["ε1"], d["ε2"], d["ε3"], d["ε4"]],
              tlist = d["tlist"], T = d["T"], T_ref = d["T_ref"],
              ζ = d["ζ"], N = d["N"], nmax = d["nmax"], F = d["F"])
end

A = load_run("ion_GRAPE_state_controls.jld2")  # 1-traj GHZ
B = load_run("ion_GRAPE_2.jld2")                # 2-traj |↑⟩,|↓⟩

@printf("\n=== Metadata ===\n")
@printf("                          %-28s  %-28s\n",
        "ion_GRAPE_state (1-traj)", "ion_GRAPE_2 (2-traj)")
@printf("  N, nmax              :  %-28s  %-28s\n",
        "$(A.N), $(A.nmax)", "$(B.N), $(B.nmax)")
@printf("  ζ                    :  %-28.6f  %-28.6f\n", A.ζ, B.ζ)
@printf("  T  (ms)              :  %-28.6f  %-28.6f\n", A.T, B.T)
@printf("  T_ref (ms)           :  %-28.6f  %-28.6f\n", A.T_ref, B.T_ref)
@printf("  T / T_ref            :  %-28.4f  %-28.4f\n",
        A.T/A.T_ref, B.T/B.T_ref)
@printf("  saved fidelity F     :  %-28.6f  %-28.6f\n", A.F, B.F)
@printf("  control samples nt   :  %-28d  %-28d\n",
        length(A.tlist), length(B.tlist))

# Per-pulse comparison (only meaningful if grids match)
grids_match = length(A.tlist) == length(B.tlist) &&
              maximum(abs.(A.tlist .- B.tlist)) < 1e-12 * max(A.T, B.T)

@printf("\n=== Per-pulse statistics  (kHz, /(2π)) ===\n")
@printf("            ε1            ε2            ε3            ε4\n")
for (label, R) in (("state(1) ‖ε‖∞", A), ("two(2)   ‖ε‖∞", B))
    @printf("  %-14s  ", label)
    for k in 1:4
        @printf("%+.4f      ", maximum(abs, R.ε[k]) / (2π))
    end
    println()
end
for (label, R) in (("state(1) ‖ε‖₂", A), ("two(2)   ‖ε‖₂", B))
    @printf("  %-14s  ", label)
    for k in 1:4
        @printf("%+.4f      ", norm(R.ε[k]) / sqrt(length(R.ε[k])) / (2π))
    end
    println()
end

if grids_match
    @printf("\n  Grids match — ‖ε_state − ε_two‖∞ /(2π)  per channel:\n  ")
    for k in 1:4
        @printf("ε%d: %.4f kHz   ", k, maximum(abs.(A.ε[k] .- B.ε[k]))/(2π))
    end
    println()
else
    println("\n  (control grids differ — skipping pointwise diff)")
end

# Overlay plot
default(fontfamily="Computer Modern", titlefontsize=12, guidefontsize=10,
        tickfontsize=8, legendfontsize=8, linewidth=1.4, dpi=200)
labels  = ["ε1 : X⊗Jx", "ε2 : P⊗Jx", "ε3 : X⊗Jy", "ε4 : P⊗Jy"]
plts = Plots.Plot[]
for k in 1:4
    plt = plot(A.tlist, A.ε[k] ./ (2π);
               label=@sprintf("state (1-traj, F=%.4f)", A.F),
               color=:steelblue, xlabel="t (ms)",
               ylabel=labels[k]*" /(2π) [kHz]", legend=:topright)
    plot!(plt, B.tlist, B.ε[k] ./ (2π);
          label=@sprintf("two   (2-traj, F=%.4f)", B.F),
          color=:crimson, linestyle=:dash)
    push!(plts, plt)
end
fig = plot(plts...; layout=(2,2), size=(1100, 750), margin=4Plots.mm,
           plot_title="GRAPE controls: 1-trajectory (GHZ) vs 2-trajectory (|↑⟩,|↓⟩)",
           plot_titlefontsize=12)
out = "compare_state_vs_two.png"
savefig(fig, out)
@printf("\nOverlay saved to: %s\n", out)
