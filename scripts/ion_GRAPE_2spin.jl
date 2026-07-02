# ion_GRAPE_2spin.jl
# GRAPE state-to-state transfer for the two-spin system of SpinBoson_test.jl:
#
#     ψ₀     = |0⟩_b ⊗ |s⟩₁ ⊗ |s⟩₂           s = :down (default) or :up
#     ψ_tgt  = D₂(cond) · [R] · S₁(cond) · ψ₀  (numerically integrated, same
#                                               stages as SpinBoson_test.jl;
#                                               the π/2 boson rotation R only
#                                               if rotate=true)
#
# Default configuration (init_spins=:down, rotate=false): with spin1 in |↓⟩
# the conditional squeeze acts as S(−ζ/2), i.e. the squeeze axis comes out
# rotated 90° relative to the |↑⟩ case, so no boson rotation between the
# stages is needed.  In that case the analytic two-stage sequence — squeeze
# drive on spin1 for t < t_strobo, displacement drive on spin2 afterwards —
# is exactly representable by the controls and the guess fidelity at
# T_frac=1 should be ≈1 (up to time discretization).
#
# GRAPE may use all EIGHT bilinear controls on both spins over the whole
# horizon:
#
#     H(t) = Σ_{s∈{1,2}} ε₁ˢ(t)·X̂⊗Jxˢ + ε₂ˢ(t)·P̂⊗Jxˢ
#                       + ε₃ˢ(t)·X̂⊗Jyˢ + ε₄ˢ(t)·P̂⊗Jyˢ,
#     X̂ = a + a†,   P̂ = i(a† − a).
#
# With rotate=true the instantaneous R is NOT representable by the initial
# guess (no n̂ control), so the guess fidelity is low and GRAPE has to
# synthesise the rotation out of the bilinear couplings.
#
# Usage: julia --project=. scripts/ion_GRAPE_2spin.jl

using QuantumOptics
using QuantumControl
using QuantumControl.Controls: get_controls, substitute, discretize
using GRAPE
const ExpProp = parentmodule(typeof(QuantumControl.init_prop)).ExpProp
using LinearAlgebra
using Printf
using JLD2
using Plots

include(joinpath(@__DIR__, "SpinBoson_test.jl"))   # build_spinboson2, build_initial2,
                                                   # make_H_squeeze, make_H_disp
                                                   # (+ SpinBoson_sim.jl helpers)

# ===== HELPERS =====

asmat(op) = Matrix{ComplexF64}(op.data)

"""Eight control operators: (X̂,P̂)⊗(Jx,Jy) on spin1 then on spin2."""
function control_operators2(sb)
    X̂ = sb.a + sb.ad
    P̂ = 1im * (sb.ad - sb.a)
    return (asmat(X̂ ⊗ sb.Jx1 ⊗ sb.Is2), asmat(P̂ ⊗ sb.Jx1 ⊗ sb.Is2),
            asmat(X̂ ⊗ sb.Jy1 ⊗ sb.Is2), asmat(P̂ ⊗ sb.Jy1 ⊗ sb.Is2),
            asmat(X̂ ⊗ sb.Is1 ⊗ sb.Jx2), asmat(P̂ ⊗ sb.Is1 ⊗ sb.Jx2),
            asmat(X̂ ⊗ sb.Is1 ⊗ sb.Jy2), asmat(P̂ ⊗ sb.Is1 ⊗ sb.Jy2))
end

"""Analytic-sequence initial guess, spin1 channel k ∈ 1:4.
   Strobo quadrature amplitudes for t < t_strobo, zero afterwards."""
@inline function guess_spin1(t::Float64, k::Int, Δ_abs::Float64,
                             ϕ1::Float64, ϕ2::Float64,
                             g0::Float64, τ::Float64, t_strobo::Float64)
    t >= t_strobo && return 0.0
    Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ)
    θ = Δ_eff * t
    vals = (g_eff * cos(θ), g_eff * sin(θ),
            g_eff * cos(θ - ϕ_eff), -g_eff * sin(θ - ϕ_eff))
    return vals[k]
end

"""Analytic-sequence initial guess, spin2 channel k ∈ 1:4.
   Zero during the strobe; H_disp = g0·P̂·(Jx₂+Jy₂) ⇒ (0, g0, 0, g0) after."""
@inline function guess_spin2(t::Float64, k::Int, g0::Float64, t_strobo::Float64)
    t < t_strobo && return 0.0
    return (k == 2 || k == 4) ? g0 : 0.0
end

# ===== TARGET STATE (numerical, same three stages as SpinBoson_test.main) =====

function build_target_2spin(sb, ψ0, Δ_abs, ϕ1, ϕ2, g0, τ,
                            t_strobo::Float64, t_free::Float64, P::Int;
                            rotate::Bool=true)
    tf = t_strobo + t_free
    tstops_s = Float64[]
    for p in 0:(P - 1)
        t0 = 4p * τ
        push!(tstops_s, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops_s, t_strobo)
    unique!(sort!(tstops_s))

    Hf_sq = make_H_squeeze(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=Inf)
    _, ψs = timeevolution.schroedinger_dynamic(
        [0.0, t_strobo], ψ0, Hf_sq;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=tstops_s, maxiters=10_000_000)

    ψ_rot = ψs[end]
    if rotate
        R_full = exp(dense(-1im * (π/2) * (sb.ad * sb.a))) ⊗ sb.Is1 ⊗ sb.Is2
        ψ_rot = R_full * ψ_rot
    end

    Hf_d = make_H_disp(sb, g0)
    _, ψf = timeevolution.schroedinger_dynamic(
        [t_strobo, tf], ψ_rot, Hf_d;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=[t_strobo, tf], maxiters=10_000_000)
    return ψf[end]
end

# ===== STATE-TRANSFER FIDELITY FUNCTIONAL (single trajectory) =====

function make_state_transfer_functionals2(target_state::Vector{ComplexF64})
    J_T(Ψ, _trajectories; kwargs...) = 1.0 - abs2(dot(target_state, Ψ[1]))
    chi(Ψ, _trajectories; kwargs...) =
        [dot(target_state, Ψ[1]) * target_state]
    return J_T, chi
end

# ===== PROBLEM SETUP =====

function build_problem2(; N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                          P::Int=1, ℓ::Int=1,
                          t_free_frac::Float64=1.0,
                          T_frac::Float64=1.0, nt::Int=250,
                          ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                          init_spins::Symbol=:down, rotate::Bool=false,
                          iter_stop::Int=200,
                          F_threshold::Float64=0.99)
    sb = build_spinboson2(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ) = pp
    t_strobo = pp.tf                       # 4Pτ
    t_free   = t_free_frac * t_strobo
    T_full   = t_strobo + t_free
    T_total  = T_frac * T_full
    α_time   = T_total / T_full            # time compression of the guess

    ψ0_ket   = build_initial2(sb; spins=init_spins)
    ψtgt_ket = build_target_2spin(sb, ψ0_ket, Δ_abs, ϕ1, ϕ2, g0, τ,
                                  t_strobo, t_free, P; rotate=rotate)
    init_state   = Vector{ComplexF64}(ψ0_ket.data)
    target_state = Vector{ComplexF64}(ψtgt_ket.data)
    @printf("‖ψ_target‖² = %.10f  (should be 1; <1 signals nmax truncation)\n",
            real(dot(target_state, target_state)))

    J_T_fn, chi_fn = make_state_transfer_functionals2(target_state)

    Hc = control_operators2(sb)

    # Initial guess: analytic two-stage sequence, time-compressed by α.
    ε1(t) = guess_spin1(t / α_time, 1, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
    ε2(t) = guess_spin1(t / α_time, 2, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
    ε3(t) = guess_spin1(t / α_time, 3, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
    ε4(t) = guess_spin1(t / α_time, 4, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
    ε5(t) = guess_spin2(t / α_time, 1, g0, t_strobo)
    ε6(t) = guess_spin2(t / α_time, 2, g0, t_strobo)
    ε7(t) = guess_spin2(t / α_time, 3, g0, t_strobo)
    ε8(t) = guess_spin2(t / α_time, 4, g0, t_strobo)

    H = hamiltonian((Hc[1], ε1), (Hc[2], ε2), (Hc[3], ε3), (Hc[4], ε4),
                    (Hc[5], ε5), (Hc[6], ε6), (Hc[7], ε7), (Hc[8], ε8))

    tlist = collect(range(0.0, T_total, length=nt))

    trajectories = [Trajectory(init_state, H;
                               target_state=target_state,
                               prop_method=ExpProp)]

    J_T_threshold = 1.0 - F_threshold
    check_convergence = res -> (res.J_T ≤ J_T_threshold) &&
                                @sprintf("F ≥ %.4f (J_T = %.3e)",
                                         F_threshold, res.J_T)

    problem = ControlProblem(
        trajectories, tlist;
        J_T = J_T_fn,
        chi = chi_fn,
        iter_stop = iter_stop,
        check_convergence = check_convergence,
    )

    return (; problem, sb, ζ, T=T_total, t_strobo, t_free, tlist,
              init_state, target_state, control_ops=Hc,
              g0, Δ_abs, τ, ϕ1, ϕ2, P, ℓ, init_spins, rotate)
end

# ===== POST-OPTIMISATION ANALYSIS =====

function evaluate_fidelity2(prob_data, controls)
    (; problem, init_state, target_state, tlist) = prob_data
    gen0   = problem.trajectories[1].generator
    H_eval = substitute(gen0, IdDict(zip(get_controls(gen0), controls)))
    ψf = propagate(init_state, H_eval, tlist; method=ExpProp)
    ov = dot(target_state, ψf)
    return (; F=abs2(ov), overlap=ov,
              norm_dev=abs(1.0 - real(dot(ψf, ψf))))
end

function initial_controls2(prob_data)
    gen0  = prob_data.problem.trajectories[1].generator
    return [discretize(c, prob_data.tlist) for c in get_controls(gen0)]
end

"""Re-propagate optimised controls in a larger Fock space to verify the
   result does not exploit the nmax truncation."""
function validate_truncation(prob_data, controls; nmax_big::Int=30)
    (; ζ, t_strobo, t_free, tlist, Δ_abs, ϕ1, ϕ2, g0, τ, P,
       init_spins, rotate) = prob_data
    N  = length(prob_data.sb.b_spin1) - 1
    sb = build_spinboson2(N, nmax_big)
    ψ0_ket   = build_initial2(sb; spins=init_spins)
    ψtgt_ket = build_target_2spin(sb, ψ0_ket, Δ_abs, ϕ1, ϕ2, g0, τ,
                                  t_strobo, t_free, P; rotate=rotate)
    Hc = control_operators2(sb)
    H  = hamiltonian(collect(zip(Hc, controls))...)
    ψf = propagate(Vector{ComplexF64}(ψ0_ket.data), H, tlist; method=ExpProp)
    return abs2(dot(Vector{ComplexF64}(ψtgt_ket.data), ψf))
end

# ===== VISUALISATION =====

"""4×2 panel plot: spin1 controls (left column) and spin2 controls (right).

By default it overlays the analytic-sequence initial guess (gray dashed) and
the GRAPE result (solid).  Toggle `show_init` / `show_grape` to draw only one
of them; pass `controls` to plot externally supplied (e.g. loaded) pulses
instead of `opt_result.optimized_controls`, and `title` to override the
auto-generated supertitle."""
function plot_pulses2(prob_data, opt_result=nothing;
                      save_path::String="results/figures/ion_GRAPE_2spin_down_pulses.png",
                      show_init::Bool=true, show_grape::Bool=true,
                      controls=nothing, title::Union{Nothing,String}=nothing,
                      share_spin2_ylims::Bool=false)
    tlist  = prob_data.tlist
    pulses = controls === nothing ?
             (opt_result === nothing ? nothing : opt_result.optimized_controls) :
             controls
    init   = initial_controls2(prob_data)

    labels = ["ε1 : X⊗Jx1", "ε2 : P⊗Jx1", "ε3 : X⊗Jy1", "ε4 : P⊗Jy1",
              "ε5 : X⊗Jx2", "ε6 : P⊗Jx2", "ε7 : X⊗Jy2", "ε8 : P⊗Jy2"]
    colors = [:blue, :red, :green, :orange,
              :purple, :crimson, :teal, :brown]

    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=9,
            tickfontsize=7, legendfontsize=7, linewidth=1.1, dpi=200)

    # Optionally put the whole spin2 column on one y-scale so the tiny ε5/ε7
    # channels read as ~flat next to the displacement drives ε6/ε8.
    spin2_ylims = nothing
    if share_spin2_ylims
        vals = Float64[]
        for k in 5:8
            show_init  && append!(vals, init[k] ./ (2π))
            show_grape && pulses !== nothing && append!(vals, pulses[k] ./ (2π))
        end
        lo, hi = extrema(vals)
        pad = 0.05 * max(hi - lo, eps())
        spin2_ylims = (lo - pad, hi + pad)
    end

    plts = Plots.Plot[]
    for k in [1, 5, 2, 6, 3, 7, 4, 8]           # spin1 left, spin2 right
        plt = plot(; xlabel="t (ms)", ylabel=labels[k] * " /(2π) [kHz]",
                   legend=:topright)
        if show_init
            plot!(plt, tlist, init[k] ./ (2π);
                  label="analytic seq", color=:gray, linestyle=:dash)
        end
        if show_grape && pulses !== nothing
            plot!(plt, tlist, pulses[k] ./ (2π); label="GRAPE", color=colors[k])
        end
        spin2_ylims !== nothing && k >= 5 && ylims!(plt, spin2_ylims)
        vline!(plt, [prob_data.t_strobo]; color=:black, linestyle=:dot,
               alpha=0.5, label="")
        push!(plts, plt)
    end
    sup = title === nothing ?
          @sprintf("2-spin controls   T=%.4f ms (t_strobo=%.4f)",
                   prob_data.T, prob_data.t_strobo) : title
    fig = plot(plts...; layout=(4, 2), size=(1300, 1300),
               plot_title=sup, plot_titlefontsize=12, margin=4Plots.mm)
    savefig(fig, save_path)
    println("Pulse plot saved to: $save_path")
    return fig
end

# ===== MAIN =====

function main_grape2(; N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                       P::Int=1, ℓ::Int=1,
                       t_free_frac::Float64=1.0, T_frac::Float64=1.0,
                       nt::Int=250, iter_stop::Int=200,
                       init_spins::Symbol=:down, rotate::Bool=false,
                       F_threshold::Float64=0.99)
    pd = build_problem2(; N, nmax, z_target, P, ℓ,
                          t_free_frac, T_frac, nt, init_spins, rotate,
                          iter_stop, F_threshold)
    spin_str = init_spins === :down ? "↓↓" : "↑↑"
    rot_str  = rotate ? "·R" : ""
    @printf("=== ion_GRAPE_2spin: |0⟩|%s⟩ → D₂(cond)%s·S₁(cond)|0⟩|%s⟩ ===\n",
            spin_str, rot_str, spin_str)
    @printf("N = %d, nmax = %d, z_target = %.3f, P = %d, ℓ = %d, dim(H) = %d\n",
            N, nmax, z_target, P, ℓ, length(pd.init_state))
    @printf("ζ = %.6f, τ = %.6f ms\n", pd.ζ, pd.τ)
    @printf("t_strobo = %.6f ms, t_free = %.6f ms, T = %.6f ms, nt = %d\n",
            pd.t_strobo, pd.t_free, pd.T, length(pd.tlist))

    init_diag = evaluate_fidelity2(pd, initial_controls2(pd))
    @printf("\nInitial-guess (analytic two-stage sequence):  F = %.6f\n",
            init_diag.F)

    @printf("\nRunning GRAPE (iter_stop = %d, F_threshold = %.4f)…\n",
            iter_stop, F_threshold)
    res = optimize(pd.problem; method=GRAPE, iter_stop=iter_stop)

    @printf("\n--- Optimisation summary ---\n")
    @printf("Iterations: %d   Final J_T: %.6e   ⇒ F ≈ %.6f\n",
            res.iter, res.J_T, 1 - res.J_T)

    diag = evaluate_fidelity2(pd, res.optimized_controls)
    @printf("\n--- Re-propagation diagnostics ---\n")
    @printf("F:                  %.8f\n", diag.F)
    @printf("|⟨φ|ψ(T)⟩|:         %.6f\n", abs(diag.overlap))
    @printf("arg⟨φ|ψ(T)⟩:        %+.4f rad   (global phase)\n", angle(diag.overlap))
    @printf("Max |1 − ‖ψ(T)‖²|:  %.2e\n", diag.norm_dev)

    F_big = validate_truncation(pd, res.optimized_controls)
    @printf("F re-propagated at nmax=30:  %.8f  (truncation check)\n", F_big)

    return (; problem_data=pd, opt_result=res, diagnostics=diag,
              init_diagnostics=init_diag, F_big)
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    out = main_grape2()
    plot_pulses2(out.problem_data, out.opt_result)

    save_path = "results/data/ion_GRAPE_2spin_down_controls.jld2"
    jldsave(save_path;
            controls = out.opt_result.optimized_controls,
            tlist    = out.problem_data.tlist,
            T        = out.problem_data.T,
            t_strobo = out.problem_data.t_strobo,
            t_free   = out.problem_data.t_free,
            ζ        = out.problem_data.ζ,
            N        = length(out.problem_data.sb.b_spin1) - 1,
            nmax     = length(out.problem_data.sb.b_fock) - 1,
            F        = out.diagnostics.F)
    println("\nOptimised controls saved to: $save_path")
end
