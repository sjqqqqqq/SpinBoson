# ion_GRAPE.jl
# GRAPE state-transfer optimal control of the trapped-ion spin-boson system,
# matching the metrology setting of arXiv:2510.25870: prepare the
# spin-dependent squeezed reference state from the bosonic vacuum and a
# fixed spin state.
#
#     |ψ_target⟩ = U_target · |0⟩_b ⊗ |ψ_spin⟩,
#     U_target   = exp( (ζ* â² − ζ â†²) · Ĵz / 2 )                  (Eq.5)
#
# For N = 1 with |ψ_spin⟩ = |ψ_GHZ⟩ = (|+½⟩ + |−½⟩)/√2 this is a single
# trajectory, ψ₀ = |0⟩_b ⊗ |ψ_GHZ⟩ → ψ_target. nmax only needs to bound the
# squeezed-vacuum support (P(n=2k) ∝ tanh(ζ/4)^{2k}/cosh(ζ/2)) — modest.
#
# Control parameterisation. We use the bilinear decomposition shared with
# SpinBoson_sim.jl, with four real piecewise-constant amplitudes on a fine
# time grid:
#
#     H(t) = ε₁(t)·X̂⊗Jx + ε₂(t)·P̂⊗Jx + ε₃(t)·X̂⊗Jy + ε₄(t)·P̂⊗Jy,
#     X̂ = a + a†,        P̂ = i(a† − a).
#
# These are the in/out-of-phase quadrature components of the physical
# (g(t), Δ(t), ϕ(t)) of Eq.(23):
#
#     ε₁ =  g·cos(Δ·t),       ε₂ =  g·sin(Δ·t),
#     ε₃ =  g·cos(Δ·t − ϕ),   ε₄ = −g·sin(Δ·t − ϕ).
#
# Figure of merit — squared modulus state fidelity:
#
#     F = |⟨ψ_target | ψ(T)⟩|²,    J_T = 1 − F.
#     χ = ⟨ψ_target | ψ(T)⟩ · |ψ_target⟩  (co-state for J_T_sm).
#
# Anchoring. The initial control guess is the analytic stroboscopic protocol
# (P=1, ℓ=1 — already gives F ≈ 0.996 for N=1, z=0.5) sampled on the GRAPE
# time grid. T = 4τ for that protocol.
#
# Usage: julia --project=. ion_GRAPE.jl

using QuantumOptics
using QuantumControl
using QuantumControl.Controls: get_controls, substitute, discretize
using GRAPE
const ExpProp = parentmodule(typeof(QuantumControl.init_prop)).ExpProp
using LinearAlgebra
using Printf
using JLD2

include("SpinBoson_sim.jl")  # build_spinboson, build_target_unitary,
                             # protocol_params, pulse_params

# ===== HELPERS =====

asmat(op) = Matrix{ComplexF64}(op.data)
asvec(ψ)  = Vector{ComplexF64}(ψ.data)

"""Four control operators (X̂Jx, P̂Jx, X̂Jy, P̂Jy)."""
function control_operators(sb)
    a, ad = sb.a, sb.ad
    X̂ = a + ad
    P̂ = 1im * (ad - a)
    return asmat(X̂ ⊗ sb.Jx), asmat(P̂ ⊗ sb.Jx),
           asmat(X̂ ⊗ sb.Jy), asmat(P̂ ⊗ sb.Jy)
end

"""Decompose the analytic protocol (g, Δ, ϕ) at time t into the four ε's."""
@inline function protocol_amplitudes(t::Float64, Δ_abs::Float64,
                                     ϕ1::Float64, ϕ2::Float64,
                                     g0::Float64, τ::Float64)
    Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ)
    θ = Δ_eff * t
    return ( g_eff * cos(θ),
             g_eff * sin(θ),
             g_eff * cos(θ - ϕ_eff),
            -g_eff * sin(θ - ϕ_eff) )
end

# ===== STATE-TRANSFER FIDELITY FUNCTIONAL =====
#
# Squared-modulus state fidelity for a single trajectory:
#
#     J_T = 1 − |⟨φ | ψ(T)⟩|²,
#     χ   = ⟨φ | ψ(T)⟩ · |φ⟩          (= −∂J_T/∂ψ* via Wirtinger calculus).
#
# Generalises to multiple trajectories by averaging J_T (here we use a
# single trajectory).

function make_state_transfer_functionals(target_states::Vector{Vector{ComplexF64}})
    Nt = length(target_states)

    function J_T(Ψ, _trajectories; kwargs...)
        F = 0.0
        @inbounds for k in 1:Nt
            F += abs2(dot(target_states[k], Ψ[k]))
        end
        return 1.0 - F / Nt
    end

    function chi(Ψ, _trajectories; kwargs...)
        out = Vector{Vector{ComplexF64}}(undef, Nt)
        @inbounds for k in 1:Nt
            ov = dot(target_states[k], Ψ[k])      # ⟨φ|ψ⟩
            out[k] = (ov / Nt) * target_states[k]
        end
        return out
    end

    return J_T, chi
end

# ===== PROBLEM SETUP =====

function build_problem(; N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                        P::Int=1, ℓ::Int=1,
                        T::Union{Nothing,Float64}=nothing, nt::Int=400,
                        ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                        init::Symbol=:GHZ,
                        iter_stop::Int=200)
    sb = build_spinboson(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ, tf) = pp
    T = something(T, tf)

    # Single-trajectory state transfer: ψ₀ = |0⟩_b ⊗ |ψ_spin⟩,
    #                                   ψ_target = U_target · ψ₀.
    U_target_op = build_target_unitary(ζ, N, sb)
    U_target    = U_target_op.data

    ψ0_ket     = build_initial(N, sb; init)
    ψtarget_ket = build_target(ζ, N, sb; init)
    init_states   = [Vector{ComplexF64}(ψ0_ket.data)]
    target_states = [Vector{ComplexF64}(ψtarget_ket.data)]
    d_V = 1

    J_T_fn, chi_fn = make_state_transfer_functionals(target_states)

    H1, H2, H3, H4 = control_operators(sb)

    # Initial guess: analytic stroboscopic protocol → four ε(t) callables.
    ε1(t) = protocol_amplitudes(t, Δ_abs, ϕ1, ϕ2, g0, τ)[1]
    ε2(t) = protocol_amplitudes(t, Δ_abs, ϕ1, ϕ2, g0, τ)[2]
    ε3(t) = protocol_amplitudes(t, Δ_abs, ϕ1, ϕ2, g0, τ)[3]
    ε4(t) = protocol_amplitudes(t, Δ_abs, ϕ1, ϕ2, g0, τ)[4]

    H = hamiltonian((H1, ε1), (H2, ε2), (H3, ε3), (H4, ε4))

    tlist = collect(range(0.0, T, length=nt))

    trajectories = Trajectory[]
    for k in 1:d_V
        push!(trajectories,
              Trajectory(init_states[k], H;
                         target_state=target_states[k],
                         prop_method=ExpProp))
    end

    problem = ControlProblem(
        trajectories, tlist;
        J_T = J_T_fn,
        chi = chi_fn,
        iter_stop = iter_stop,
    )

    return (; problem, sb, ζ, T, tf, tlist, U_target_op, target_states,
              init_states, control_ops=(H1, H2, H3, H4),
              d_V, init, g0, Δ_abs, τ, ϕ1, ϕ2)
end

# ===== POST-OPTIMISATION ANALYSIS =====

"""Re-propagate the trajectory(ies) under given controls (vector-of-vectors
matching `get_controls` order) and report state fidelity F = |⟨φ|ψ(T)⟩|²."""
function evaluate_fidelity(prob_data, controls)
    (; problem, init_states, target_states, tlist, d_V) = prob_data
    gen0 = problem.trajectories[1].generator
    H_eval = substitute(gen0, IdDict(zip(get_controls(gen0), controls)))

    F_total = 0.0
    norm_dev = 0.0
    overlaps = ComplexF64[]
    for k in 1:d_V
        ψf = propagate(init_states[k], H_eval, tlist; method=ExpProp)
        ov = dot(target_states[k], ψf)
        push!(overlaps, ov)
        F_total += abs2(ov)
        norm_dev = max(norm_dev, abs(1.0 - real(dot(ψf, ψf))))
    end
    F = F_total / d_V
    return (; F, overlaps, norm_dev)
end

"""Discretise the (callable) initial-guess controls onto tlist."""
function initial_controls(prob_data)
    gen0  = prob_data.problem.trajectories[1].generator
    ctrls = get_controls(gen0)
    return [discretize(c, prob_data.tlist) for c in ctrls]
end

# ===== VISUALISATION =====

using Plots

"""Four-panel comparison of GRAPE-optimised pulses against the analytic
   stroboscopic initial guess (in kHz, /(2π))."""
function plot_pulses(prob_data, opt_result; save_path::String="ion_GRAPE_pulses.png")
    tlist  = prob_data.tlist
    pulses = opt_result.optimized_controls
    init   = initial_controls(prob_data)
    @assert length(pulses[1]) == length(tlist)

    labels = ["ε1 : X⊗Jx", "ε2 : P⊗Jx", "ε3 : X⊗Jy", "ε4 : P⊗Jy"]
    colors = [:blue, :red, :green, :orange]

    default(fontfamily="Computer Modern", titlefontsize=12, guidefontsize=10,
            tickfontsize=8, legendfontsize=8, linewidth=1.2, dpi=200)

    plts = Plots.Plot[]
    for k in 1:4
        plt = plot(tlist, init[k] ./ (2π);
                   label="protocol (init)", color=:gray, linestyle=:dash,
                   xlabel="t (ms)", ylabel=labels[k] * " /(2π)  [kHz]",
                   legend=:topright)
        plot!(plt, tlist, pulses[k] ./ (2π);
              label="GRAPE", color=colors[k])
        push!(plts, plt)
    end
    fig = plot(plts...; layout=(2, 2), size=(1100, 750),
               plot_title="GRAPE-optimised spin-boson controls",
               plot_titlefontsize=13, margin=4Plots.mm)
    savefig(fig, save_path)
    println("Pulse plot saved to: $save_path")
    return fig
end

"""In-place phase unwrap: shift each successive value by 2π until the
   step is in (−π, π]."""
function unwrap!(a::Vector{Float64})
    @inbounds for i in 2:length(a)
        while a[i] - a[i-1] >  π;  a[i] -= 2π; end
        while a[i] - a[i-1] < -π;  a[i] += 2π; end
    end
    return a
end
unwrap(a::Vector{Float64}) = unwrap!(copy(a))

"""Three-panel pulse-sequence plot in the style of arXiv:2510.25870 Fig.4(c):
   Δ(t), ϕ(t), g(t) recovered from the four optimised ε's, with the analytic
   stroboscopic protocol overlaid as a gray-dashed reference.

   Recovery (real ε's, with c₁ = ε₁ − iε₂, c₂ = ε₃ − iε₄):
       g(t)   = |c₁(t)| ≈ |c₂(t)|
       Δ(t)   = −d/dt arg(c₁(t))            (numerical derivative on
                                              unwrapped phase)
       ϕ(t)   = arg(c₁(t))·(−1) − arg(c₂(t)) = Δt − arg(c₂)

   Caveat: a sign flip of g(t) (segments 3,4 of the protocol) is gauge-equivalent
   to a π shift in arg(c₁)+arg(c₂), so |c₁|, |c₂| are always plotted as positive
   while the ±g sign appears as π-jumps absorbed into Δ(t)·t and ϕ(t)."""
function plot_fig4c_style(prob_data, opt_result;
                          save_path::String="ion_GRAPE_fig4c.png")
    tlist = prob_data.tlist
    p     = opt_result.optimized_controls
    e1, e2, e3, e4 = p[1], p[2], p[3], p[4]

    # |c1|, |c2|, and unwrapped phases
    c1_abs = sqrt.(e1.^2 .+ e2.^2)
    c2_abs = sqrt.(e3.^2 .+ e4.^2)
    g_rec  = (c1_abs .+ c2_abs) ./ 2

    arg_c1 = unwrap(atan.(-e2, e1))     # arg(c1) = −Δ·t  (analytically)
    arg_c2 = unwrap(atan.(-e4, e3))     # arg(c2) = +Δ·t − ϕ
    Δt_rec = -arg_c1                    # Δ·t (unwrapped)
    ϕ_rec  = Δt_rec .- arg_c2           # ϕ(t) = Δt − arg(c2), unwrapped

    # Centred difference for instantaneous Δ(t)
    Δ_inst = similar(Δt_rec)
    @inbounds for i in eachindex(Δt_rec)
        if i == 1
            Δ_inst[i] = (Δt_rec[i+1] - Δt_rec[i]) / (tlist[i+1] - tlist[i])
        elseif i == length(Δt_rec)
            Δ_inst[i] = (Δt_rec[i] - Δt_rec[i-1]) / (tlist[i] - tlist[i-1])
        else
            Δ_inst[i] = (Δt_rec[i+1] - Δt_rec[i-1]) / (tlist[i+1] - tlist[i-1])
        end
    end

    # Fold ϕ to (−π, π] for display
    ϕ_disp = mod.(ϕ_rec .+ π, 2π) .- π

    # Analytic-protocol references
    (; Δ_abs, ϕ1, ϕ2, g0, τ) = prob_data
    Δ_ref  = Float64[]; ϕ_ref = Float64[]; g_ref = Float64[]
    for t in tlist
        Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ)
        push!(Δ_ref, Δ_eff)
        push!(ϕ_ref, ϕ_eff)
        push!(g_ref, g_eff)
    end

    default(fontfamily="Computer Modern", titlefontsize=12, guidefontsize=11,
            tickfontsize=10, legendfontsize=9, linewidth=1.6, dpi=200)

    # Segment boundaries to mark the protocol's stroboscopic structure
    P_cycles = round(Int, prob_data.T / (4 * τ))
    seg_lines = [k * τ for k in 0:(4 * P_cycles)]

    p1 = plot(tlist, Δ_ref ./ (2π);
              color=:gray, linestyle=:dash, label="protocol",
              ylabel="Δ(t) /(2π)  [kHz]",
              xticks=:none, legend=:topright,
              ylims=(-1.2 * Δ_abs/(2π), 1.2 * Δ_abs/(2π)))
    plot!(p1, tlist, Δ_inst ./ (2π); color=:darkorange, label="GRAPE")
    for sl in seg_lines
        vline!(p1, [sl]; color=:lightgray, linestyle=:dot, alpha=0.4, label="")
    end

    p2 = plot(tlist, ϕ_ref;
              color=:gray, linestyle=:dash, label="protocol",
              ylabel="ϕ(t)  [rad]",
              xticks=:none, legend=:topright,
              ylims=(-π - 0.4, π + 0.4))
    plot!(p2, tlist, ϕ_disp; color=:darkgreen, label="GRAPE")
    for sl in seg_lines
        vline!(p2, [sl]; color=:lightgray, linestyle=:dot, alpha=0.4, label="")
    end

    p3 = plot(tlist, g_ref ./ (2π);
              color=:gray, linestyle=:dash, label="protocol  (signed)",
              ylabel="g(t) /(2π)  [kHz]",
              xlabel="t  [ms]", legend=:bottomright,
              ylims=(-1.4 * g0/(2π), 1.4 * g0/(2π)))
    plot!(p3, tlist,  g_rec ./ (2π); color=:purple,
          label="GRAPE  |c₁|=|c₂|")
    plot!(p3, tlist, -g_rec ./ (2π); color=:purple,
          linestyle=:dashdot, alpha=0.4, label="")
    for sl in seg_lines
        vline!(p3, [sl]; color=:lightgray, linestyle=:dot, alpha=0.4, label="")
    end

    fig = plot(p1, p2, p3; layout=(3, 1), size=(900, 850), link=:x,
               margin=4Plots.mm,
               plot_title="GRAPE-optimised pulse sequence  (Fig.4(c) style)",
               plot_titlefontsize=12)
    savefig(fig, save_path)
    println("Fig 4(c)-style plot saved to: $save_path")
    return fig
end

# ===== MAIN =====

function main(; N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                P::Int=1, ℓ::Int=1, init::Symbol=:GHZ,
                nt::Int=400, iter_stop::Int=200)
    pd = build_problem(; N, nmax, z_target, P, ℓ, init, nt, iter_stop)
    @printf("=== ion_GRAPE: state transfer |0⟩_b⊗|ψ_spin⟩ → U_target·|0⟩_b⊗|ψ_spin⟩ ===\n")
    @printf("N = %d, nmax = %d, z_target = %.3f, P = %d, ℓ = %d, init = %s\n",
            N, nmax, z_target, P, ℓ, init)
    @printf("ζ = %.6f, T = tf = %.6f ms, nt = %d, d_V = %d\n",
            pd.ζ, pd.T, length(pd.tlist), pd.d_V)

    init_diag = evaluate_fidelity(pd, initial_controls(pd))
    @printf("\nInitial-guess (analytic protocol):  F = %.6f\n", init_diag.F)

    @printf("\nRunning GRAPE (iter_stop = %d)…\n", iter_stop)
    res = optimize(pd.problem; method=GRAPE, iter_stop=iter_stop)

    @printf("\n--- Optimisation summary ---\n")
    @printf("Iterations: %d   Final J_T: %.6e   ⇒ F ≈ %.6f\n",
            res.iter, res.J_T, 1 - res.J_T)

    diag = evaluate_fidelity(pd, res.optimized_controls)
    @printf("\n--- Re-propagation diagnostics ---\n")
    @printf("F:                  %.8f\n", diag.F)
    @printf("|⟨φ|ψ(T)⟩|:         %.6f\n", abs(diag.overlaps[1]))
    @printf("arg⟨φ|ψ(T)⟩:        %+.4f rad   (global phase)\n",
            angle(diag.overlaps[1]))
    @printf("Max |1 − ‖ψ(T)‖²|:  %.2e\n", diag.norm_dev)

    return (; problem_data=pd, opt_result=res, diagnostics=diag,
              init_diagnostics=init_diag)
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    out = main(N=1, nmax=20, z_target=0.5, P=1, init=:GHZ,
               nt=200, iter_stop=200)
    plot_pulses(out.problem_data, out.opt_result)
    plot_fig4c_style(out.problem_data, out.opt_result)

    save_path = "ion_GRAPE_controls.jld2"
    jldsave(save_path;
            ε1    = out.opt_result.optimized_controls[1],
            ε2    = out.opt_result.optimized_controls[2],
            ε3    = out.opt_result.optimized_controls[3],
            ε4    = out.opt_result.optimized_controls[4],
            tlist = out.problem_data.tlist,
            T     = out.problem_data.T,
            ζ     = out.problem_data.ζ,
            N     = length(out.problem_data.sb.b_spin) - 1,
            nmax  = length(out.problem_data.sb.b_fock) - 1,
            F     = out.diagnostics.F)
    println("\nOptimised controls saved to: $save_path")
end
