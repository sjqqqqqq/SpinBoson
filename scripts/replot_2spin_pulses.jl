# replot_2spin_pulses.jl
# Re-render the three two-spin pulse figures with plot_pulses2 from saved
# GRAPE controls (results/data/ion_GRAPE_2spin_down_controls_Tfrac{10,05}.jld2):
#
#   1. analytic protocol only                         (gray dashed)
#   2. analytic protocol + GRAPE   (T_frac = 1.0)
#   3. GRAPE only                  (T_frac = 0.5)
#
# Usage: julia --project=. scripts/replot_2spin_pulses.jl

using JLD2
include(joinpath(@__DIR__, "ion_GRAPE_2spin.jl"))

const DATA = joinpath(@__DIR__, "..", "results", "data")
const FIGS = joinpath(@__DIR__, "..", "results", "figures")

ctrl(frac) = load(joinpath(DATA, "ion_GRAPE_2spin_down_controls_Tfrac$(frac).jld2"),
                  "controls")

# T_frac = 1 problem supplies the analytic init guess + matching tlist;
# T_frac = 0.5 problem supplies the compressed tlist / t_strobo.
pd10 = build_problem2(; T_frac=1.0)
pd05 = build_problem2(; T_frac=0.5)

# 1. Analytic protocol only.
plot_pulses2(pd10; show_init=true, show_grape=false,
             title="2-spin analytic protocol (initial guess)",
             save_path=joinpath(FIGS, "ion_GRAPE_2spin_down_pulses_analytic.png"))

# 2. Analytic protocol + GRAPE, T_frac = 1.
plot_pulses2(pd10; controls=ctrl("10"), show_init=true, show_grape=true,
             share_spin2_ylims=true,
             title=@sprintf("2-spin analytic + GRAPE   T=%.4f ms (t_strobo=%.4f)",
                            pd10.T, pd10.t_strobo),
             save_path=joinpath(FIGS, "ion_GRAPE_2spin_down_pulses_Tfrac10.png"))

# 3. GRAPE only, T_frac = 0.5.
plot_pulses2(pd05; controls=ctrl("05"), show_init=false, show_grape=true,
             title=@sprintf("2-spin GRAPE   T=%.4f ms (t_strobo=%.4f)",
                            pd05.T, pd05.t_strobo),
             save_path=joinpath(FIGS, "ion_GRAPE_2spin_down_pulses_Tfrac05.png"))

println("\nAll three figures re-rendered.")
