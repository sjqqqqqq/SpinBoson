# ion_GRAPE_displace.jl
# GRAPE state transfer for the "squeeze + displacement" extended pulse:
# a P-cycle stroboscopic stage (squeezing) followed by a free segment of
# duration t_free with controls (Δ=0, ϕ=0, g=+g0). In the free segment,
#       H = g0·(a+a†)⊗Jx + g0·(a+a†)⊗Jy  =  √2·g0·X̂⊗(Jx+Jy)/√2,
# i.e. a spin-conditioned displacement along the (Jx+Jy)/√2 direction.
#
# Initial:  ψ₀     = |0⟩_b ⊗ |GHZ⟩
# Target:   ψ_tgt  = U_extended(t_strobo + t_free) · ψ₀
#                  = D(α·(Jx+Jy)) · S(ζJz) · ψ₀     (numerically extracted)
# Horizon:  T      = t_strobo + t_free
# Guess:    extended analytic pulse, sampled on the GRAPE time grid.
#
# Usage: julia --project=. ion_GRAPE_displace.jl

using QuantumOptics
using QuantumControl
using QuantumControl.Controls: get_controls, substitute, discretize
using GRAPE
const ExpProp = parentmodule(typeof(QuantumControl.init_prop)).ExpProp
using LinearAlgebra
using Printf
using JLD2

include("SpinBoson_sim.jl")   # build_spinboson, build_initial, protocol_params,
                              # pulse_params, make_H_dynamic

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

"""Extended-protocol (squeeze + free displacement) decomposed into four ε's.
   Inside the strobo stage (t < t_strobo): same as ion_GRAPE.jl.
   In the free segment (t ≥ t_strobo): Δ=ϕ=0, g=+g0 ⇒ (ε₁,ε₂,ε₃,ε₄)=(g0,0,g0,0)."""
@inline function protocol_amplitudes_ext(t::Float64, Δ_abs::Float64,
                                         ϕ1::Float64, ϕ2::Float64,
                                         g0::Float64, τ::Float64,
                                         t_strobo::Float64)
    Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ;
                                       t_strobo=t_strobo)
    θ = Δ_eff * t
    return ( g_eff * cos(θ),
             g_eff * sin(θ),
             g_eff * cos(θ - ϕ_eff),
            -g_eff * sin(θ - ϕ_eff) )
end

"""Same as protocol_amplitudes_ext for t < t_strobo, but in the free segment
   uses (ε₁,ε₂,ε₃,ε₄) = (0, g0, 0, g0), i.e. H_free = g0·P̂·(Jx+Jy) — which
   shifts the conditional displacement axis from p onto x."""
@inline function protocol_amplitudes_ext_xdisp(t::Float64, Δ_abs::Float64,
                                                ϕ1::Float64, ϕ2::Float64,
                                                g0::Float64, τ::Float64,
                                                t_strobo::Float64)
    if t >= t_strobo
        return (0.0, g0, 0.0, g0)
    end
    return protocol_amplitudes_ext(t, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
end

"""Stroboscopic Hamiltonian (same as `make_H_dynamic`) but with the free
   segment replaced by H_free = g0·P̂·(Jx+Jy), P̂ = i(a†−a)."""
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

"""Three-stage extended target: strobo squeeze (0 → t_strobo) producing
   S|ψ₀⟩ with squeeze axis along x; an instantaneous boson rotation
   R = exp(−i·π/2·n̂) that takes the squeezing axis onto p; then free-segment
   x-displacement (t_strobo → tf) via `make_H_dynamic_xdisp`."""
function build_target_ext_xdisp(sb, ψ0_ket, Δ_abs, ϕ1, ϕ2, g0, τ,
                                t_strobo::Float64, t_free::Float64, P::Int)
    tf = t_strobo + t_free
    tstops_s = Float64[]
    for p in 0:(P-1)
        t0 = 4p * τ
        push!(tstops_s, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops_s, t_strobo)
    unique!(sort!(tstops_s))

    Hf_s = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=Inf)
    _, ψs = timeevolution.schroedinger_dynamic(
        [0.0, t_strobo], ψ0_ket, Hf_s;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=tstops_s, maxiters=10_000_000)
    ψ_after_squeeze = ψs[end]

    R_full = exp(dense(-1im * (π/2) * (sb.ad * sb.a))) ⊗ one(sb.b_spin)
    ψ_after_rot = R_full * ψ_after_squeeze

    Hf_f = make_H_dynamic_xdisp(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo)
    _, ψf = timeevolution.schroedinger_dynamic(
        [t_strobo, tf], ψ_after_rot, Hf_f;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=[t_strobo, tf], maxiters=10_000_000)
    return ψf[end]
end

# ===== STATE-TRANSFER FIDELITY FUNCTIONAL =====

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
            ov = dot(target_states[k], Ψ[k])
            out[k] = (ov / Nt) * target_states[k]
        end
        return out
    end

    return J_T, chi
end

# ===== TARGET STATE (numerical) =====

"""Produce the squeeze+displacement target by integrating the extended
   analytic pulse from ψ₀ = |0⟩_b ⊗ |ψ_spin⟩ for total time t_strobo+t_free."""
function build_target_ext(sb, ψ0_ket, Δ_abs, ϕ1, ϕ2, g0, τ,
                          t_strobo::Float64, t_free::Float64, P::Int)
    tf = t_strobo + t_free
    tstops = Float64[]
    for p in 0:(P - 1)
        t0 = 4p * τ
        push!(tstops, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops, t_strobo, tf)
    unique!(sort!(tstops))

    Hf = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo)
    _, ψt = timeevolution.schroedinger_dynamic(
        [0.0, tf], ψ0_ket, Hf;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=tstops, maxiters=10_000_000,
    )
    return ψt[end]
end

# ===== PROBLEM SETUP =====

function build_problem(; N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                        P::Int=1, ℓ::Int=1,
                        t_free_frac::Float64=1.0,
                        t_free::Union{Nothing,Float64}=nothing,
                        T::Union{Nothing,Float64}=nothing,
                        T_frac::Float64=1.0, nt::Int=400,
                        ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                        init::Symbol=:GHZ,
                        iter_stop::Int=200,
                        F_threshold::Float64=0.99,
                        xdisp::Bool=false)
    sb = build_spinboson(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ) = pp
    t_strobo = pp.tf                                # 4Pτ
    t_free = something(t_free, t_free_frac * t_strobo)
    T_full = t_strobo + t_free
    T_total = something(T, T_frac * T_full)
    # Time-compression factor: ε_init(t') = ε_proto(t' / α), α = T_total/T_full.
    α_time = T_total / T_full

    ψ0_ket = build_initial(N, sb; init)
    init_state = Vector{ComplexF64}(ψ0_ket.data)

    target_builder = xdisp ? build_target_ext_xdisp : build_target_ext
    amplitudes_fn  = xdisp ? protocol_amplitudes_ext_xdisp : protocol_amplitudes_ext

    ψtgt_ket = target_builder(sb, ψ0_ket, Δ_abs, ϕ1, ϕ2, g0, τ,
                              t_strobo, t_free, P)
    target_state = Vector{ComplexF64}(ψtgt_ket.data)
    @printf("‖ψ_target‖² = %.10f  (should be 1, xdisp=%s)\n",
            real(dot(target_state, target_state)), xdisp)

    init_states   = [init_state]
    target_states = [target_state]
    d_V = 1

    J_T_fn, chi_fn = make_state_transfer_functionals(target_states)

    H1, H2, H3, H4 = control_operators(sb)

    # Initial guess: extended squeeze+displace protocol, time-compressed by α.
    # At α=1 this is the protocol itself; for α<1 it's stretched onto a
    # shorter horizon, oscillating faster but keeping the same segment shape.
    ε1(t) = amplitudes_fn(t / α_time, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)[1]
    ε2(t) = amplitudes_fn(t / α_time, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)[2]
    ε3(t) = amplitudes_fn(t / α_time, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)[3]
    ε4(t) = amplitudes_fn(t / α_time, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)[4]

    H = hamiltonian((H1, ε1), (H2, ε2), (H3, ε3), (H4, ε4))

    tlist = collect(range(0.0, T_total, length=nt))

    trajectories = Trajectory[]
    for k in 1:d_V
        push!(trajectories,
              Trajectory(init_states[k], H;
                         target_state=target_states[k],
                         prop_method=ExpProp))
    end

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
              target_states, init_states,
              control_ops=(H1, H2, H3, H4),
              d_V, init, g0, Δ_abs, τ, ϕ1, ϕ2, P, ℓ)
end

# ===== POST-OPTIMISATION ANALYSIS =====

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

function initial_controls(prob_data)
    gen0  = prob_data.problem.trajectories[1].generator
    ctrls = get_controls(gen0)
    return [discretize(c, prob_data.tlist) for c in ctrls]
end

# ===== VISUALISATION =====

using Plots

function plot_pulses(prob_data, opt_result; save_path::String="ion_GRAPE_displace_pulses.png")
    tlist  = prob_data.tlist
    pulses = opt_result.optimized_controls
    init   = initial_controls(prob_data)
    @assert length(pulses[1]) == length(tlist)

    labels = ["ε1 : X⊗Jx", "ε2 : P⊗Jx", "ε3 : X⊗Jy", "ε4 : P⊗Jy"]
    colors = [:blue, :red, :green, :orange]

    default(fontfamily="Computer Modern", titlefontsize=12, guidefontsize=10,
            tickfontsize=8, legendfontsize=8, linewidth=1.2, dpi=200)

    T_total = prob_data.T
    T_ref   = prob_data.t_strobo + prob_data.t_free   # full reference protocol
    T_frac  = T_total / T_ref
    t_norm  = tlist ./ T_ref          # x ∈ [0, T_frac] in units of T_ref

    plts = Plots.Plot[]
    for k in 1:4
        plt = plot(t_norm, init[k] ./ (2π);
                   label="extended protocol (init)", color=:gray, linestyle=:dash,
                   xlabel="t / T_ref", ylabel=labels[k] * " /(2π)  [kHz]",
                   xlims=(0.0, T_frac), legend=:topright)
        plot!(plt, t_norm, pulses[k] ./ (2π);
              label="GRAPE", color=colors[k])
        vline!(plt, [T_frac / 2]; color=:black, linestyle=:dot,
               alpha=0.5, label="")
        push!(plts, plt)
    end
    fig = plot(plts...; layout=(2, 2), size=(1100, 750),
               plot_title=@sprintf("GRAPE: squeeze + displacement   T=%.3f ms (t_strobo=%.3f, t_free=%.3f)",
                                    prob_data.T, prob_data.t_strobo, prob_data.t_free),
               plot_titlefontsize=12, margin=4Plots.mm)
    savefig(fig, save_path)
    println("Pulse plot saved to: $save_path")
    return fig
end

# ===== MAIN =====

function main(; N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                P::Int=1, ℓ::Int=1, init::Symbol=:GHZ,
                t_free_frac::Float64=1.0,
                t_free::Union{Nothing,Float64}=nothing,
                T_frac::Float64=1.0,
                nt::Int=400, iter_stop::Int=200,
                F_threshold::Float64=0.99,
                xdisp::Bool=false)
    pd = build_problem(; N, nmax, z_target, P, ℓ, init,
                         t_free_frac, t_free, T_frac, nt, iter_stop,
                         F_threshold, xdisp)
    @printf("=== ion_GRAPE_displace: |0⟩_b⊗|%s⟩ → %s squeeze+displace target ===\n",
            init, xdisp ? "x-displaced (rotated)" : "p-displaced")
    @printf("N = %d, nmax = %d, z_target = %.3f, P = %d, ℓ = %d\n",
            N, nmax, z_target, P, ℓ)
    @printf("ζ = %.6f, τ = %.6f ms\n", pd.ζ, pd.τ)
    @printf("t_strobo = %.6f ms, t_free = %.6f ms, T = %.6f ms, nt = %d\n",
            pd.t_strobo, pd.t_free, pd.T, length(pd.tlist))

    init_diag = evaluate_fidelity(pd, initial_controls(pd))
    @printf("\nInitial-guess (extended protocol):  F = %.6f\n", init_diag.F)

    @printf("\nRunning GRAPE (iter_stop = %d, F_threshold = %.4f)…\n",
            iter_stop, F_threshold)
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

# ===== SWEEP T_frac =====

"""Sweep GRAPE optimisations over a list of horizon fractions α = T_new / T_full.
   Target is fixed (extended-protocol final state); only T shrinks. Initial
   guess is the same extended protocol time-compressed onto each shorter T."""
function sweep_T_frac(T_fracs::Vector{Float64};
                      N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                      P::Int=1, ℓ::Int=1, init::Symbol=:GHZ,
                      t_free_frac::Float64=1.0,
                      nt::Int=400, iter_stop::Int=400,
                      F_threshold::Float64=0.99,
                      save_prefix::String="ion_GRAPE_displace",
                      xdisp::Bool=false)
    results = NamedTuple[]
    for α in T_fracs
        println("\n" * "="^60)
        @printf("  T_frac = %.4f\n", α)
        println("="^60)
        out = main(; N, nmax, z_target, P, ℓ, init,
                     t_free_frac, T_frac=α, nt, iter_stop, F_threshold, xdisp)
        tag = @sprintf("Tfrac%03d", round(Int, 1000*α))
        plot_pulses(out.problem_data, out.opt_result;
                    save_path="$(save_prefix)_pulses_$tag.png")
        jldsave("$(save_prefix)_controls_$tag.jld2";
                ε1=out.opt_result.optimized_controls[1],
                ε2=out.opt_result.optimized_controls[2],
                ε3=out.opt_result.optimized_controls[3],
                ε4=out.opt_result.optimized_controls[4],
                tlist=out.problem_data.tlist,
                T=out.problem_data.T,
                t_strobo=out.problem_data.t_strobo,
                t_free=out.problem_data.t_free,
                T_frac=α,
                ζ=out.problem_data.ζ,
                N=length(out.problem_data.sb.b_spin) - 1,
                nmax=length(out.problem_data.sb.b_fock) - 1,
                F=out.diagnostics.F)
        push!(results, (; T_frac=α,
                          T=out.problem_data.T,
                          F_init=out.init_diagnostics.F,
                          F_opt=out.diagnostics.F,
                          iter=out.opt_result.iter))
    end
    println("\n" * "="^60)
    println("  Sweep summary")
    println("="^60)
    @printf("%8s %12s %12s %12s %6s\n",
            "T_frac", "T (ms)", "F_init", "F_opt", "iter")
    for r in results
        @printf("%8.4f %12.6f %12.6f %12.8f %6d\n",
                r.T_frac, r.T, r.F_init, r.F_opt, r.iter)
    end
    return results
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    # Same target as spinboson_pulse_extended_preview.png: P=1 strobo + 4τ free.
    # Sweep GRAPE horizon T over fractions of the analytic protocol duration.
    sweep_T_frac([0.9, 0.75, 0.5, 1/3];
                 N=1, nmax=20, z_target=0.5, P=1, init=:GHZ,
                 t_free_frac=1.0, nt=400, iter_stop=400)
end
