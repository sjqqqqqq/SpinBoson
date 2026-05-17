# plot_ion_GRAPE_2.jl
# Load saved controls from ion_GRAPE_2.jld2 and re-plot via plot_pulses.
# Usage:  julia --project=. plot_ion_GRAPE_2.jl
#
# Notes:
#   • The JLD2 stores ε1..ε4, tlist, T, T_ref, ζ, N, nmax, F — but not P, ℓ,
#     ϕ1, ϕ2 or the initial spin state. To get the analytic-protocol
#     reference curve overlaid by plot_pulses, we reconstruct a problem with
#     the same defaults the 30%-T script uses (P=4, ℓ=1, ϕ1=π, ϕ2=0). If
#     these don't match the file's run, only the gray "protocol (init)"
#     dashed curve will be wrong — the GRAPE traces are still the saved data.

include("ion_GRAPE.jl")

const JLD2_PATH = length(ARGS) ≥ 1 ? ARGS[1] : "ion_GRAPE_2.jld2"
const SAVE_PATH = length(ARGS) ≥ 2 ? ARGS[2] : "ion_GRAPE_2_pulses.png"

data = load(JLD2_PATH)
ε     = [data["ε1"], data["ε2"], data["ε3"], data["ε4"]]
tlist = data["tlist"]
T     = data["T"]
T_ref = data["T_ref"]
ζ     = data["ζ"]
N     = data["N"]
nmax  = data["nmax"]
F_saved = data["F"]

z_target = ζ * N / 2          # ζ = 2·z_target / N
T_factor = T / T_ref

@printf("Loaded %s\n", JLD2_PATH)
@printf("  N=%d  nmax=%d  ζ=%.4f  (z_target=%.4f)\n", N, nmax, ζ, z_target)
@printf("  T=%.6f ms  T_ref=%.6f ms  (T_factor=%.4f)\n", T, T_ref, T_factor)
@printf("  saved F = %.6f\n\n", F_saved)

# Reconstruct a matching problem so plot_pulses can produce its overlay.
# Assumes P=4, ℓ=1, ϕ1=π, ϕ2=0 (defaults of ion_GRAPE_30pct.jl). The
# initial guess is time-compressed onto [0,T] just like in run_30pct.
P_assumed = 4
ℓ_assumed = 1
ϕ1, ϕ2 = Float64(π), 0.0

sb = build_spinboson(N, nmax)
pp = protocol_params(N, z_target, P_assumed, ℓ_assumed)
(; g0, Δ_abs, τ) = pp

s = T_ref / T
ε1(t) = protocol_amplitudes(t * s, Δ_abs, ϕ1, ϕ2, g0, τ)[1]
ε2(t) = protocol_amplitudes(t * s, Δ_abs, ϕ1, ϕ2, g0, τ)[2]
ε3(t) = protocol_amplitudes(t * s, Δ_abs, ϕ1, ϕ2, g0, τ)[3]
ε4(t) = protocol_amplitudes(t * s, Δ_abs, ϕ1, ϕ2, g0, τ)[4]

H1, H2, H3, H4 = control_operators(sb)
H = hamiltonian((H1, ε1), (H2, ε2), (H3, ε3), (H4, ε4))

# A throwaway trajectory just so initial_controls() can fetch the ε callables.
vac   = fockstate(sb.b_fock, 0)
ψ_up  = vac ⊗ basisstate(sb.b_spin, 1)
target_states = [Vector{ComplexF64}(ψ_up.data)]
J_T_fn, chi_fn = make_state_transfer_functionals(target_states)
traj = Trajectory(Vector{ComplexF64}(ψ_up.data), H;
                  target_state=target_states[1], prop_method=ExpProp)
problem = ControlProblem([traj], collect(tlist); J_T=J_T_fn, chi=chi_fn)

prob_data  = (; problem, tlist=collect(tlist))
opt_result = (; optimized_controls = ε)

plot_pulses(prob_data, opt_result; save_path=SAVE_PATH)
