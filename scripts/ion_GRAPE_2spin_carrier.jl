# ion_GRAPE_2spin_carrier.jl
# ion_GRAPE_2spin.jl plus two CARRIER controls on spin2 (the Ĥ_c of
# Matsos et al., ref2.pdf, discussion below Eq. 1), targeting the
# spin-DISENTANGLED product state
#
#     ψ₀    = |0⟩_b ⊗ |↓⟩₁ ⊗ |↓⟩₂
#     ψ_tgt = D(α)·S(−ζ/2)|0⟩_b ⊗ |↓⟩₁ ⊗ |↓⟩₂,   α = g0·t_free/√2
#
# Without the carrier this target is unreachable: all eight bilinear controls
# conserve G = e^{iπn̂}⊗σz⊗σz and a displaced squeezed vacuum is not a parity
# eigenstate, capping F at (1+⟨G⟩_tgt)/2 ≈ 0.505.  The carrier σx₂/σy₂
# anticommutes with σz₂ and breaks G.
#
# Ten controls:  ε₁..ε₈ as in ion_GRAPE_2spin.jl, plus
#     ε₉(t)·1_b⊗Jx₂ + ε₁₀(t)·1_b⊗Jy₂.
#
# Initial guess = analytic sequence with a carrier sandwich around the
# displacement stage:
#   * strobe squeeze on spin1 for t < t_strobo (spin1 |↓⟩ is a Jz₁ eigenstate,
#     so it returns to |↓⟩ on its own);
#   * carrier π/2 rotation taking |↓⟩₂ → |+⟩_{x+y} (rotation about (1,−1,0)/√2);
#   * displacement drive g0·P̂·(Jx₂+Jy₂), now acting on an eigenstate of
#     (Jx₂+Jy₂) ⇒ unconditional D(+α), spin2 untouched;
#   * inverse carrier rotation |+⟩_{x+y} → |↓⟩₂.
#
# Usage: julia --project=. scripts/ion_GRAPE_2spin_carrier.jl

include(joinpath(@__DIR__, "ion_GRAPE_2spin.jl"))

# ===== CONTROLS =====

"""Ten control operators: the eight bilinear ones plus the spin2 carrier."""
function control_operators_carrier(sb)
    Ib = one(sb.b_fock)
    return (control_operators2(sb)...,
            asmat(Ib ⊗ sb.Is1 ⊗ sb.Jx2),
            asmat(Ib ⊗ sb.Is1 ⊗ sb.Jy2))
end

# ===== TARGET =====

"""Product target D(α)S(−ζ/2)|0⟩ ⊗ |↓↓⟩ (down branch of the conditional
   squeeze, + branch of the conditional displacement)."""
function build_target_product(sb, ζ::Float64, α::Float64)
    ψb = displace(sb.b_fock, α) * (squeeze(sb.b_fock, -ζ / 2) *
                                   fockstate(sb.b_fock, 0))
    return ψb ⊗ spindown(sb.b_spin1) ⊗ spindown(sb.b_spin2)
end

# ===== INITIAL GUESS =====

"""Displacement-stage guess for spin2 bilinear channel k ∈ 1:4 (ε₅..ε₈):
   active only inside the carrier sandwich, amplitude rescaled so the total
   displacement area still equals g0·t_free."""
@inline function guess_disp_sandwich(t::Float64, k::Int, g0::Float64,
                                     t_strobo::Float64, t_free::Float64,
                                     t_rot::Float64)
    t1 = t_strobo + t_rot
    t2 = t_strobo + t_free - t_rot
    (t < t1 || t >= t2) && return 0.0
    g_eff = g0 * t_free / (t_free - 2t_rot)
    return (k == 2 || k == 4) ? g_eff : 0.0
end

"""Carrier guess, k = 1 → Jx₂ channel (ε₉), k = 2 → Jy₂ channel (ε₁₀).
   H_rot = ±c·(Jx₂ − Jy₂) with c·t_rot = π/(2√2) rotates the spin2 Bloch
   vector by ±π/2 about (1,−1,0)/√2, i.e. |↓⟩ ↔ |+⟩_{x+y}."""
@inline function guess_carrier(t::Float64, k::Int, t_strobo::Float64,
                               t_free::Float64, t_rot::Float64)
    c  = (π / (2 * sqrt(2))) / t_rot
    t1 = t_strobo + t_rot
    t2 = t_strobo + t_free - t_rot
    T  = t_strobo + t_free
    s = if t_strobo <= t < t1
        1.0                       # |↓⟩ → |+⟩_{x+y}
    elseif t2 <= t <= T
        -1.0                      # |+⟩_{x+y} → |↓⟩
    else
        0.0
    end
    return s * (k == 1 ? c : -c)
end

# ===== PROBLEM SETUP =====

function build_problem_carrier(; N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                                 P::Int=1, ℓ::Int=1,
                                 t_free_frac::Float64=1.0,
                                 T_frac::Float64=1.0, nt::Int=250,
                                 t_rot_frac::Float64=0.03,
                                 ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                                 iter_stop::Int=300,
                                 F_threshold::Float64=0.999)
    sb = build_spinboson2(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free   = t_free_frac * t_strobo
    T_full   = t_strobo + t_free
    T_total  = T_frac * T_full
    α_time   = T_total / T_full
    t_rot    = t_rot_frac * T_full

    α_disp   = g0 * t_free / sqrt(2)
    ψ0_ket   = build_initial2(sb; spins=:down)
    ψtgt_ket = build_target_product(sb, ζ, α_disp)
    init_state   = Vector{ComplexF64}(ψ0_ket.data)
    target_state = Vector{ComplexF64}(ψtgt_ket.data)
    @printf("‖ψ_target‖² = %.10f  (should be 1; <1 signals nmax truncation)\n",
            real(dot(target_state, target_state)))

    J_T_fn, chi_fn = make_state_transfer_functionals2(target_state)

    Hc = control_operators_carrier(sb)

    ε1(t) = guess_spin1(t / α_time, 1, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
    ε2(t) = guess_spin1(t / α_time, 2, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
    ε3(t) = guess_spin1(t / α_time, 3, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
    ε4(t) = guess_spin1(t / α_time, 4, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo)
    ε5(t) = guess_disp_sandwich(t / α_time, 1, g0, t_strobo, t_free, t_rot)
    ε6(t) = guess_disp_sandwich(t / α_time, 2, g0, t_strobo, t_free, t_rot)
    ε7(t) = guess_disp_sandwich(t / α_time, 3, g0, t_strobo, t_free, t_rot)
    ε8(t) = guess_disp_sandwich(t / α_time, 4, g0, t_strobo, t_free, t_rot)
    ε9(t)  = guess_carrier(t / α_time, 1, t_strobo, t_free, t_rot)
    ε10(t) = guess_carrier(t / α_time, 2, t_strobo, t_free, t_rot)

    H = hamiltonian((Hc[1], ε1), (Hc[2], ε2), (Hc[3], ε3), (Hc[4], ε4),
                    (Hc[5], ε5), (Hc[6], ε6), (Hc[7], ε7), (Hc[8], ε8),
                    (Hc[9], ε9), (Hc[10], ε10))

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

    return (; problem, sb, ζ, α_disp, T=T_total, t_strobo, t_free, t_rot,
              tlist, init_state, target_state, control_ops=Hc,
              g0, Δ_abs, τ, ϕ1, ϕ2, P, ℓ)
end

# ===== DIAGNOSTICS =====

"""Reduced-spin diagnostics of the final state reached by `controls`."""
function spin_diagnostics_carrier(pd, controls)
    gen0 = pd.problem.trajectories[1].generator
    H = substitute(gen0, IdDict(zip(get_controls(gen0), controls)))
    ψf_vec = propagate(pd.init_state, H, pd.tlist; method=ExpProp)
    F = abs2(dot(pd.target_state, ψf_vec))
    ψf = Ket(pd.sb.b_full, ψf_vec)
    ρ  = ψf ⊗ dagger(ψf)
    ρs = ptrace(ρ, 1)
    ρ2 = ptrace(ρ, [1, 2])
    dn = spindown(pd.sb.b_spin1) ⊗ spindown(pd.sb.b_spin2)
    p_dd = real(dot(dn.data, (ρs * dn).data))
    b2 = (real(expect(sigmax(pd.sb.b_spin2), ρ2)),
          real(expect(sigmay(pd.sb.b_spin2), ρ2)),
          real(expect(sigmaz(pd.sb.b_spin2), ρ2)))
    return (; F, p_dd, purity_spins=real(tr(ρs * ρs)), bloch2=b2, ψf)
end

"""Re-propagate in a larger Fock space against the product target."""
function validate_truncation_carrier(pd, controls; nmax_big::Int=30)
    N  = length(pd.sb.b_spin1) - 1
    sb = build_spinboson2(N, nmax_big)
    ψ0_ket   = build_initial2(sb; spins=:down)
    ψtgt_ket = build_target_product(sb, pd.ζ, pd.α_disp)
    Hc = control_operators_carrier(sb)
    H  = hamiltonian(collect(zip(Hc, controls))...)
    ψf = propagate(Vector{ComplexF64}(ψ0_ket.data), H, pd.tlist; method=ExpProp)
    return abs2(dot(Vector{ComplexF64}(ψtgt_ket.data), ψf))
end

# ===== VISUALISATION =====

"""5×2 panel plot: spin1 bilinear (left) / spin2 bilinear (right) rows 1–4,
   carrier Jx₂ / Jy₂ row 5.

By default overlays the sandwich guess (gray dashed) and the GRAPE result
(solid, per-channel colors).  `show_init=false` drops the guess trace; pass
`controls` to plot externally supplied pulses instead of
`opt_result.optimized_controls`, `controls_label` to rename them, and `title`
to override the supertitle."""
function plot_pulses_carrier(pd, opt_result=nothing;
                             save_path::String="results/figures/ion_GRAPE_2spin_carrier_pulses.png",
                             controls=nothing, show_init::Bool=true,
                             controls_label::String="GRAPE",
                             title::Union{Nothing,String}=nothing)
    tlist  = pd.tlist
    pulses = controls === nothing ?
             (opt_result === nothing ? nothing : opt_result.optimized_controls) :
             controls
    init   = initial_controls2(pd)   # generic: discretizes whatever pd carries

    labels = ["ε1 : X⊗Jx1", "ε2 : P⊗Jx1", "ε3 : X⊗Jy1", "ε4 : P⊗Jy1",
              "ε5 : X⊗Jx2", "ε6 : P⊗Jx2", "ε7 : X⊗Jy2", "ε8 : P⊗Jy2",
              "ε9 : 1⊗Jx2 (carrier)", "ε10 : 1⊗Jy2 (carrier)"]
    colors = [:blue, :red, :green, :orange,
              :purple, :crimson, :teal, :brown, :black, :magenta]

    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=9,
            tickfontsize=7, legendfontsize=7, linewidth=1.1, dpi=200)

    plts = Plots.Plot[]
    for k in [1, 5, 2, 6, 3, 7, 4, 8, 9, 10]     # spin1 left, spin2 right, carrier last row
        plt = plot(; xlabel="t (ms)", ylabel=labels[k] * " /(2π) [kHz]",
                   legend=:topright)
        if show_init
            plot!(plt, tlist, init[k] ./ (2π);
                  label="guess (carrier sandwich)", color=:gray, linestyle=:dash)
        end
        if pulses !== nothing
            plot!(plt, tlist, pulses[k] ./ (2π);
                  label=controls_label, color=colors[k])
        end
        vline!(plt, [pd.t_strobo]; color=:black, linestyle=:dot,
               alpha=0.5, label="")
        push!(plts, plt)
    end
    sup = title === nothing ?
          @sprintf("2-spin + carrier controls   T=%.4f ms   target D(%.2f)S(%+.2f)|0>⊗|dd>",
                   pd.T, pd.α_disp, -pd.ζ / 2) : title
    fig = plot(plts...; layout=(5, 2), size=(1300, 1600),
               plot_title=sup, plot_titlefontsize=12, margin=4Plots.mm)
    savefig(fig, save_path)
    println("Pulse plot saved to: $save_path")
    return fig
end

# ===== MAIN =====

function main_grape_carrier(; N::Int=1, nmax::Int=20, z_target::Float64=0.5,
                              P::Int=1, ℓ::Int=1,
                              t_free_frac::Float64=1.0, T_frac::Float64=1.0,
                              nt::Int=250, t_rot_frac::Float64=0.03,
                              iter_stop::Int=300,
                              F_threshold::Float64=0.999)
    pd = build_problem_carrier(; N, nmax, z_target, P, ℓ,
                                 t_free_frac, T_frac, nt, t_rot_frac,
                                 iter_stop, F_threshold)
    @printf("=== ion_GRAPE_2spin_carrier: |0,↓↓⟩ → D(α)S(−ζ/2)|0⟩⊗|↓↓⟩ ===\n")
    @printf("N = %d, nmax = %d, z_target = %.3f, P = %d, dim(H) = %d\n",
            N, nmax, z_target, P, length(pd.init_state))
    @printf("ζ = %.4f, α = %.4f, t_strobo = %.6f ms, t_rot = %.6f ms, T = %.6f ms\n",
            pd.ζ, pd.α_disp, pd.t_strobo, pd.t_rot, pd.T)

    init_diag = spin_diagnostics_carrier(pd, initial_controls2(pd))
    @printf("\nInitial guess (analytic + carrier sandwich):  F = %.6f\n",
            init_diag.F)
    @printf("  p(↓↓) = %.6f,  spin2 Bloch = (%+.4f, %+.4f, %+.4f)\n",
            init_diag.p_dd, init_diag.bloch2...)

    @printf("\nRunning GRAPE (iter_stop = %d, F_threshold = %.4f)…\n",
            iter_stop, F_threshold)
    res = optimize(pd.problem; method=GRAPE, iter_stop=iter_stop)

    @printf("\n--- Optimisation summary ---\n")
    @printf("Iterations: %d   Final J_T: %.6e   ⇒ F ≈ %.6f\n",
            res.iter, res.J_T, 1 - res.J_T)

    diag = spin_diagnostics_carrier(pd, res.optimized_controls)
    @printf("\n--- Re-propagation diagnostics (GRAPE pulse) ---\n")
    @printf("F:                  %.8f\n", diag.F)
    @printf("p(↓↓):              %.6f\n", diag.p_dd)
    @printf("2-spin purity:      %.6f\n", diag.purity_spins)
    @printf("spin2 Bloch:        (%+.4f, %+.4f, %+.4f)\n", diag.bloch2...)

    F_big = validate_truncation_carrier(pd, res.optimized_controls)
    @printf("F re-propagated at nmax=30:  %.8f  (truncation check)\n", F_big)

    return (; problem_data=pd, opt_result=res, diagnostics=diag,
              init_diagnostics=init_diag, F_big)
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    out = main_grape_carrier()
    plot_pulses_carrier(out.problem_data, out.opt_result)

    save_path = "results/data/ion_GRAPE_2spin_carrier_controls.jld2"
    jldsave(save_path;
            controls = out.opt_result.optimized_controls,
            tlist    = out.problem_data.tlist,
            T        = out.problem_data.T,
            t_strobo = out.problem_data.t_strobo,
            t_free   = out.problem_data.t_free,
            t_rot    = out.problem_data.t_rot,
            ζ        = out.problem_data.ζ,
            α        = out.problem_data.α_disp,
            N        = length(out.problem_data.sb.b_spin1) - 1,
            nmax     = length(out.problem_data.sb.b_fock) - 1,
            F        = out.diagnostics.F)
    println("\nOptimised controls saved to: $save_path")
end
