# spinboson_grape.jl
# Step 2: GRAPE-optimize the analytic protocol of spinboson_protocol.jl.
#
# The target is exactly the state step 1 produces — the analytic protocol run
# to completion on |0>|dd> — so the analytic pulse itself is a valid solution
# whenever the optimizer is given the full duration.
#
# EIGHT bilinear controls, four per spin, in the quadrature/spin-axis basis:
#
#     H(t) = sum_{s=1,2} [ e1s(t)*X*Jxs + e2s(t)*P*Jxs
#                        + e3s(t)*X*Jys + e4s(t)*P*Jys ],
#     X = a + a†,   P = i(a† − a)
#
# The analytic protocol written in that basis (see `analytic_guess_*` below) is
# the initial guess, so GRAPE starts on the analytic solution and refines it.
#
# Two horizons are tested, set by `T_frac`:
#
#   T_frac = 1.0   full duration. The guess already reaches the target, so the
#                  optimized pulse should sit right on top of the analytic one —
#                  only slightly different, absorbing the piecewise-constant
#                  discretization error.
#   T_frac = 0.5   half duration. The guess is time-compressed, so it accrues
#                  only half the area and falls well short; GRAPE has to find a
#                  genuinely different, stronger pulse.
#
# Usage:
#   julia --project=. spinboson_grape.jl
#
#   julia --project=. -i -e 'include("spinboson_grape.jl")'
#     out = run_grape(T_frac=1.0)
#     plot_pulses(out; save_path="results/out.png")

include(joinpath(@__DIR__, "spinboson_protocol.jl"))   # step 1: system, protocol, pulse_params

using QuantumControl
using QuantumControl.Controls: get_controls, substitute, discretize
using GRAPE
using JLD2
const ExpProp = parentmodule(typeof(QuantumControl.init_prop)).ExpProp

asmat(op) = Matrix{ComplexF64}(op.data)

# ===== CONTROL OPERATORS =====

"""The eight control operators: (X,P) x (Jx,Jy) on spin1, then the same on spin2.

Ordering matches the ε indices used everywhere below and in the JaqalPaw
exporter: (X*Jx1, P*Jx1, X*Jy1, P*Jy1, X*Jx2, P*Jx2, X*Jy2, P*Jy2)."""
function control_operators(sb)
    X = sb.a + sb.ad
    P = 1im * (sb.ad - sb.a)
    Jx1 = sigmax(sb.b_spin1) / 2; Jy1 = sigmay(sb.b_spin1) / 2
    Jx2 = sigmax(sb.b_spin2) / 2; Jy2 = sigmay(sb.b_spin2) / 2
    return (asmat(X ⊗ Jx1 ⊗ sb.Is2), asmat(P ⊗ Jx1 ⊗ sb.Is2),
            asmat(X ⊗ Jy1 ⊗ sb.Is2), asmat(P ⊗ Jy1 ⊗ sb.Is2),
            asmat(X ⊗ sb.Is1 ⊗ Jx2), asmat(P ⊗ sb.Is1 ⊗ Jx2),
            asmat(X ⊗ sb.Is1 ⊗ Jy2), asmat(P ⊗ sb.Is1 ⊗ Jy2))
end

# ===== THE ANALYTIC PULSE AS AN INITIAL GUESS =====
#
# Stage 1 is H = g(t)*a*[Jx1*exp(-i*th) + Jy1*exp(+i*th)*exp(-i*phi)] + h.c.
# with th = D(t)*t. Expanding a and a† onto X and P:
#
#     c*a + conj(c)*a†  =  Re(c)*X + Im(c)*P      (c = g*exp(-i*th))
#
# gives (e1, e2) = g*(cos th, sin th) for the Jx1 channel and
#       (e3, e4) = g*(cos(th - phi), -sin(th - phi)) for the Jy1 channel.
# Stage 2 is H = g0*P*(Jx2 + Jy2), i.e. (e5..e8) = (0, g0, 0, g0).

"""Spin-1 analytic guess, channel `k` in 1:4. Zero once the strobe ends."""
@inline function analytic_guess_spin1(t::Float64, k::Int, Δ_abs::Float64,
                                      ϕ1::Float64, ϕ2::Float64,
                                      g0::Float64, τ::Float64, t_strobo::Float64)
    t >= t_strobo && return 0.0
    Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ)
    θ = Δ_eff * t
    vals = (g_eff * cos(θ), g_eff * sin(θ),
            g_eff * cos(θ - ϕ_eff), -g_eff * sin(θ - ϕ_eff))
    return vals[k]
end

"""Spin-2 analytic guess, channel `k` in 1:4. Zero during the strobe, then the
constant displacement drive."""
@inline function analytic_guess_spin2(t::Float64, k::Int, g0::Float64,
                                      t_strobo::Float64)
    t < t_strobo && return 0.0
    return (k == 2 || k == 4) ? g0 : 0.0
end

# ===== PROBLEM SETUP =====

"""State-transfer functional: J_T = 1 - |<target|psi(T)>|^2."""
function state_transfer_functionals(target::Vector{ComplexF64})
    J_T(Ψ, _traj; kwargs...) = 1.0 - abs2(dot(target, Ψ[1]))
    chi(Ψ, _traj; kwargs...) = [dot(target, Ψ[1]) * target]
    return J_T, chi
end

"""Assemble the GRAPE control problem.

The target comes from `run_protocol` — the same analytic evolution step 1
plots — so step 1 and step 2 cannot drift apart. `T_frac` compresses the
horizon; the guess is compressed with it, which is what makes T_frac < 1 a real
optimization rather than a re-parameterization."""
function build_grape_problem(; N::Int=1, nmax::Int=30, z_target::Float64=0.5,
                               P::Int=1, ℓ::Int=1, spins::Symbol=:down,
                               t_free_frac::Float64=1.0, T_frac::Float64=1.0,
                               nt::Int=250, rotate::Bool=false,
                               ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                               iter_stop::Int=200, F_threshold::Float64=0.99)
    # Step 1 supplies both the initial state and the target.
    ref = run_protocol(; N, nmax, z_target, P, ℓ, spins, t_free_frac, rotate,
                         ϕ1, ϕ2, verbose=false)
    sb = ref.sb
    (; g0, ζ, Δ_abs, τ) = ref.pp

    t_strobo = ref.t_strobo
    t_free   = ref.t_free
    T_full   = ref.tf
    T_total  = T_frac * T_full
    α_time   = T_frac                      # guess is compressed by this factor

    init_state   = Vector{ComplexF64}(ref.ψ0.data)
    target_state = Vector{ComplexF64}(ref.ψ_final.data)

    Hc = control_operators(sb)

    # Initial guess: the analytic pulse, time-compressed onto the new horizon.
    g1(k) = t -> analytic_guess_spin1(t / α_time, k, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
    g2(k) = t -> analytic_guess_spin2(t / α_time, k, g0, t_strobo)
    guesses = (g1(1), g1(2), g1(3), g1(4), g2(1), g2(2), g2(3), g2(4))

    H = hamiltonian(collect(zip(Hc, guesses))...)
    tlist = collect(range(0.0, T_total, length=nt))

    J_T_fn, chi_fn = state_transfer_functionals(target_state)
    J_T_threshold = 1.0 - F_threshold
    check_convergence = res -> (res.J_T ≤ J_T_threshold) &&
                               @sprintf("F >= %.4f (J_T = %.3e)", F_threshold, res.J_T)

    problem = ControlProblem(
        [Trajectory(init_state, H; target_state=target_state, prop_method=ExpProp)],
        tlist;
        J_T=J_T_fn, chi=chi_fn, iter_stop=iter_stop,
        check_convergence=check_convergence,
    )

    return (; problem, sb, ref, control_ops=Hc, tlist, init_state, target_state,
              T=T_total, T_full, T_frac, t_strobo, t_free, α_time,
              g0, ζ, Δ_abs, τ, ϕ1, ϕ2, N, nmax, z_target, P, ℓ, spins, rotate)
end

# ===== DIAGNOSTICS =====

"""Re-propagate `controls` and return the state-transfer fidelity."""
function evaluate_fidelity(pd, controls)
    gen0 = pd.problem.trajectories[1].generator
    H = substitute(gen0, IdDict(zip(get_controls(gen0), controls)))
    ψf = propagate(pd.init_state, H, pd.tlist; method=ExpProp)
    ov = dot(pd.target_state, ψf)
    return (; F=abs2(ov), overlap=ov, norm_dev=abs(1.0 - real(dot(ψf, ψf))))
end

"""The analytic guess, sampled on the problem's time grid."""
function guess_controls(pd)
    gen0 = pd.problem.trajectories[1].generator
    return [discretize(c, pd.tlist) for c in get_controls(gen0)]
end

"""Largest absolute difference between the guess and the optimized controls."""
max_change(guess, opt) = maximum(maximum(abs, o .- g) for (g, o) in zip(guess, opt))

"""Re-propagate optimized controls in a LARGER Fock space.

GRAPE runs at a cutoff chosen for speed, and a pulse that quietly relies on the
truncation would score well there and fail on hardware. Rebuilding the whole
problem at `nmax_big` and re-propagating is the check that it does not."""
function validate_truncation(pd, controls; nmax_big::Int=60)
    ref = run_protocol(; N=pd.N, nmax=nmax_big, z_target=pd.z_target, P=pd.P,
                         ℓ=pd.ℓ, spins=pd.spins, rotate=pd.rotate,
                         ϕ1=pd.ϕ1, ϕ2=pd.ϕ2, verbose=false)
    Hc = control_operators(ref.sb)
    H = hamiltonian(collect(zip(Hc, controls))...)
    ψf = propagate(Vector{ComplexF64}(ref.ψ0.data), H, pd.tlist; method=ExpProp)
    return abs2(dot(Vector{ComplexF64}(ref.ψ_final.data), ψf))
end

# ===== RUN =====

"""Optimize one horizon and report guess vs optimized fidelity."""
function run_grape(; T_frac::Float64=1.0, nmax::Int=30, nt::Int=250,
                     iter_stop::Int=200, F_threshold::Float64=0.99,
                     nmax_big::Int=60, verbose::Bool=true, kwargs...)
    pd = build_grape_problem(; T_frac, nmax, nt, iter_stop, F_threshold, kwargs...)

    if verbose
        @printf("=== GRAPE, T_frac = %.2f ===\n", T_frac)
        @printf("nmax = %d, dim(H) = %d, nt = %d\n", nmax, length(pd.init_state), nt)
        @printf("T = %.4f ms of %.4f ms full, t_strobo = %.4f ms\n",
                pd.T, pd.T_full, pd.t_strobo)
    end

    guess = guess_controls(pd)
    F_guess = evaluate_fidelity(pd, guess).F
    verbose && @printf("\nanalytic guess:  F = %.6f\n", F_guess)

    verbose && @printf("\nrunning GRAPE (iter_stop = %d, F_threshold = %.4f)...\n",
                       iter_stop, F_threshold)
    t_start = time()
    res = optimize(pd.problem; method=GRAPE, iter_stop=iter_stop)
    elapsed = time() - t_start

    opt = res.optimized_controls
    diag = evaluate_fidelity(pd, opt)
    F_big = validate_truncation(pd, opt; nmax_big=nmax_big)

    if verbose
        @printf("\n--- summary (T_frac = %.2f) ---\n", T_frac)
        @printf("iterations:        %d in %.1f s (%.1f s/iter)\n",
                res.iter, elapsed, elapsed / max(res.iter, 1))
        @printf("F (guess):         %.6f\n", F_guess)
        @printf("F (GRAPE):         %.6f\n", diag.F)
        @printf("F at nmax=%d:      %.6f   (truncation check)\n", nmax_big, F_big)
        @printf("max |1-norm|:      %.2e\n", diag.norm_dev)
        @printf("max |pulse change|: %.4f rad/ms  (%.1f%% of g0)\n",
                max_change(guess, opt), 100 * max_change(guess, opt) / pd.g0)
    end

    return (; pd, res, guess, opt, F_guess, F=diag.F, F_big, elapsed)
end

# ===== VISUALISATION =====

const CH_LABELS = ("e1  X*Jx", "e2  P*Jx", "e3  X*Jy", "e4  P*Jy")

"""4x2 panel figure: spin-1 controls on the left, spin-2 on the right, with the
analytic guess (gray dashed) under the GRAPE result (solid).

Rates are plotted in kHz; the stored controls are angular, in rad/ms."""
function plot_pulses(out; save_path::String="results/spinboson_grape_pulses.png",
                          title::Union{Nothing,String}=nothing)
    (; pd, guess, opt) = out
    t = pd.tlist

    default(fontfamily="Computer Modern", titlefontsize=10, guidefontsize=9,
            tickfontsize=7, legendfontsize=8, linewidth=1.5, dpi=200)

    # One y-scale for all eight panels, always. Channels 5 and 7 are identically
    # zero in the analytic protocol, so per-panel auto-scaling blows their
    # sub-percent corrections up to fill the frame and the pulse reads as
    # "completely different" when it is within 1% of the analytic one. The
    # per-panel deviation in each title carries that detail instead.
    ymax = maximum(maximum(abs, v) for v in vcat(guess, opt)) / (2π)
    ylims = (-1.08 * ymax, 1.08 * ymax)

    panels = Plots.Plot[]
    for row in 1:4, (col, spin) in enumerate((1, 2))
        k = (spin - 1) * 4 + row
        dev = maximum(abs, opt[k] .- guess[k]) / pd.g0
        p = plot(; xlabel=(row == 4 ? "t (ms)" : ""),
                   ylabel=@sprintf("%s / 2pi [kHz]", CH_LABELS[row]),
                   ylims=ylims,
                   title=(row == 1 ? "spin $spin\n" : "") *
                         @sprintf("max|GRAPE - guess| = %.2f%% of g0", 100 * dev),
                   legend=(row == 1 && col == 1 ? :topright : false))
        plot!(p, t, guess[k] ./ (2π); color=:gray55, linestyle=:dash,
              linewidth=2.0, label="analytic guess")
        plot!(p, t, opt[k] ./ (2π); color=(spin == 1 ? :crimson : :royalblue),
              label="GRAPE")
        # The strobe/free boundary only exists on the uncompressed horizon.
        pd.T_frac == 1.0 && vline!(p, [pd.t_strobo]; color=:black,
                                   linestyle=:dot, alpha=0.4, label="")
        push!(panels, p)
    end
    ttl = something(title,
        @sprintf("GRAPE from the analytic guess   T_frac = %.2f, T = %.4f ms   F: %.6f -> %.6f",
                 pd.T_frac, pd.T, out.F_guess, out.F))
    fig = plot(panels...; layout=grid(4, 2), size=(1300, 1150),
               plot_title=ttl, plot_titlefontsize=12,
               leftmargin=7Plots.mm, bottommargin=3Plots.mm)

    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: ", save_path)
    return fig
end

"""Wigner function of the state the optimized pulse actually produces, next to
the analytic target."""
function plot_wigner_compare(out; xrange=range(-8.0, 8.0, length=241),
                                  prange=range(-8.0, 8.0, length=241),
                                  save_path::String="results/spinboson_grape_wigner.png")
    (; pd, opt) = out
    gen0 = pd.problem.trajectories[1].generator
    H = substitute(gen0, IdDict(zip(get_controls(gen0), opt)))
    ψf_vec = propagate(pd.init_state, H, pd.tlist; method=ExpProp)
    ψf = Ket(pd.sb.b_full, ψf_vec)

    xvec = collect(Float64, xrange); pvec = collect(Float64, prange)
    pairs = [("analytic target", pd.ref.ψ_final),
             (@sprintf("GRAPE, T_frac = %.2f", pd.T_frac), ψf)]

    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=10,
            tickfontsize=8, dpi=200)
    panels = Plots.Plot[]
    for (label, ψ) in pairs
        W = boson_wigner(ψ, xvec, pvec)
        cmax = maximum(abs, W)
        n̄ = real(expect(pd.sb.n_op, ψ))
        push!(panels, heatmap(xvec, pvec, W'; c=:RdBu, clims=(-cmax, cmax),
                              xlabel="x", ylabel="p", aspect_ratio=:equal,
                              colorbar=true,
                              title=@sprintf("%s\n<n> = %.2f", label, n̄),
                              xlims=(xvec[1], xvec[end]),
                              ylims=(pvec[1], pvec[end])))
    end
    fig = plot(panels...; layout=grid(1, 2), size=(1150, 560),
               plot_title=@sprintf("F = %.6f", out.F), plot_titlefontsize=12,
               leftmargin=6Plots.mm, bottommargin=5Plots.mm)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: ", save_path)
    return fig
end

# ===== MAIN =====

"""Optimize both horizons, plot each, and save the controls.

Also writes the analytic pulse in the same format (see `save_analytic_controls`)
so `export_jaqalpaw.jl` has all three drives to convert. Only at T_frac = 1: the
time-compressed analytic pulse reaches F = 0.42, so it is a GRAPE starting point,
not a pulse to put on hardware."""
function main(; nmax::Int=30, nt::Int=250, iter_stop::Int=200,
                T_fracs=(1.0, 0.5))
    save_analytic_controls(; T_frac=1.0, nmax, nt)
    outs = Dict{Float64,Any}()
    for T_frac in T_fracs
        out = run_grape(; T_frac, nmax, nt, iter_stop)
        tag = @sprintf("Tfrac%02d", round(Int, 100 * T_frac))
        plot_pulses(out; save_path="results/spinboson_grape_pulses_$tag.png")
        plot_wigner_compare(out; save_path="results/spinboson_grape_wigner_$tag.png")

        jldsave("results/spinboson_grape_controls_$tag.jld2";
                controls=out.opt, tlist=out.pd.tlist, T=out.pd.T,
                t_strobo=out.pd.t_strobo, t_free=out.pd.t_free,
                ζ=out.pd.ζ, N=out.pd.N, nmax=out.pd.nmax,
                F=out.F, F_guess=out.F_guess, T_frac=T_frac)
        println("Saved: results/spinboson_grape_controls_$tag.jld2\n")
        outs[T_frac] = out
    end
    return outs
end

"""Write the ANALYTIC pulse in the same 8-control format GRAPE produces.

The analytic protocol is just a particular point in the same control space, so
it needs no separate export path: discretize `analytic_guess_*` onto the time
grid and save it with the identical JLD2 schema. `export_jaqalpaw.jl` and
`spinboson_pulses.py` then consume it unchanged.

Cheap — no optimization — so this is the way to get the analytic pulse onto
hardware. At T_frac = 1 the result is identical to the T_frac = 1 GRAPE run,
which converges at iteration 0 precisely because this pulse already solves the
problem."""
function save_analytic_controls(; T_frac::Float64=1.0, nmax::Int=30,
                                  nt::Int=250, path=nothing, kwargs...)
    pd = build_grape_problem(; T_frac, nmax, nt, kwargs...)
    controls = guess_controls(pd)
    F = evaluate_fidelity(pd, controls).F

    tag = @sprintf("Tfrac%02d", round(Int, 100 * T_frac))
    path = something(path, "results/spinboson_analytic_controls_$tag.jld2")
    jldsave(path;
            controls, tlist=pd.tlist, T=pd.T,
            t_strobo=pd.t_strobo, t_free=pd.t_free,
            ζ=pd.ζ, N=pd.N, nmax=pd.nmax, F, F_guess=F, T_frac)
    @printf("analytic pulse (8 controls), T_frac = %.2f, F = %.6f -> %s\n",
            T_frac, F, path)
    return (; pd, controls, F, path)
end

"""Rebuild the plot inputs from a saved control file, without re-optimizing.

GRAPE takes tens of minutes per horizon, so figure tweaks go through here."""
function load_run(T_frac::Float64)
    tag = @sprintf("Tfrac%02d", round(Int, 100 * T_frac))
    # nmax and nt come from the file, so the rebuilt problem matches the run.
    d = load("results/spinboson_grape_controls_$tag.jld2")
    pd = build_grape_problem(; T_frac, nmax=d["nmax"], nt=length(d["tlist"]))
    return (; pd, res=nothing, guess=guess_controls(pd), opt=d["controls"],
              F_guess=d["F_guess"], F=d["F"], F_big=NaN, elapsed=NaN)
end

"""Re-render both figures for both horizons from the saved controls."""
function replot(; T_fracs=(1.0, 0.5))
    for T_frac in T_fracs
        out = load_run(T_frac)
        tag = @sprintf("Tfrac%02d", round(Int, 100 * T_frac))
        plot_pulses(out; save_path="results/spinboson_grape_pulses_$tag.png")
        plot_wigner_compare(out; save_path="results/spinboson_grape_wigner_$tag.png")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
