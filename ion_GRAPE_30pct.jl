# ion_GRAPE_30pct.jl
# Single-shot GRAPE at T = 0.30 · T_ref for N=1, z=1 (T_ref = analytic P=4 tf).
# Initial guess: the analytic stroboscopic protocol time-compressed onto the
# shortened grid (i.e. evaluate `protocol_amplitudes(t · T_ref/T, …)` so that
# all four ε's complete the full P=4 cycle within T). Stops as soon as F>0.99.
#
# Usage: julia --project=. ion_GRAPE_30pct.jl

include("ion_GRAPE.jl")

function run_30pct(; N::Int=1, nmax::Int=20, z_target::Float64=1.0,
                    P::Int=4, ℓ::Int=1,
                    nt::Int=300, iter_stop::Int=100,
                    T_factor::Float64=0.30, F_threshold::Float64=0.99,
                    ϕ1::Float64=Float64(π), ϕ2::Float64=0.0)

    sb = build_spinboson(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ, tf) = pp
    T_ref = tf
    T     = T_factor * T_ref

    @printf("=== GRAPE @ T = %.2f · T_ref  (early-stop F > %.3f) ===\n",
            T_factor, F_threshold)
    @printf("N = %d, z = %.3f, P_init = %d, ℓ = %d  (two trajectories: |↑⟩, |↓⟩)\n",
            N, z_target, P, ℓ)
    @printf("T_ref = %.6f ms,  T = %.6f ms,  ζ = %.4f\n", T_ref, T, ζ)
    flush(stdout)

    # Two trajectories: |0⟩_b ⊗ |↑⟩  and  |0⟩_b ⊗ |↓⟩, targets = U_target · ψ₀.
    # In QuantumOptics SpinBasis ordering, idx 1 ↔ |+J⟩ (=|↑⟩), idx end ↔ |−J⟩ (=|↓⟩).
    vac = fockstate(sb.b_fock, 0)
    dim_s = N + 1
    ψ_up   = vac ⊗ basisstate(sb.b_spin, 1)
    ψ_down = vac ⊗ basisstate(sb.b_spin, dim_s)
    U_target_op = build_target_unitary(ζ, N, sb)
    ψ_up_tgt   = U_target_op * ψ_up
    ψ_down_tgt = U_target_op * ψ_down

    init_states   = [Vector{ComplexF64}(ψ_up.data),     Vector{ComplexF64}(ψ_down.data)]
    target_states = [Vector{ComplexF64}(ψ_up_tgt.data), Vector{ComplexF64}(ψ_down_tgt.data)]
    d_V = 2

    J_T_fn, chi_fn = make_state_transfer_functionals(target_states)
    H1, H2, H3, H4 = control_operators(sb)

    # Time-compressed analytic protocol guess: stretch the P=4 protocol onto
    # the [0,T] window so the four ε's still complete one full cycle.
    s = T_ref / T   # speedup factor (>1)
    ε1(t) = protocol_amplitudes(t * s, Δ_abs, ϕ1, ϕ2, g0, τ)[1]
    ε2(t) = protocol_amplitudes(t * s, Δ_abs, ϕ1, ϕ2, g0, τ)[2]
    ε3(t) = protocol_amplitudes(t * s, Δ_abs, ϕ1, ϕ2, g0, τ)[3]
    ε4(t) = protocol_amplitudes(t * s, Δ_abs, ϕ1, ϕ2, g0, τ)[4]

    H = hamiltonian((H1, ε1), (H2, ε2), (H3, ε3), (H4, ε4))
    tlist = collect(range(0.0, T, length=nt))
    trajectories = [Trajectory(init_states[k], H;
                               target_state=target_states[k], prop_method=ExpProp)
                    for k in 1:d_V]

    # Early-stop: stop as soon as J_T < 1 - F_threshold.
    J_stop = 1.0 - F_threshold
    function check_convergence!(res)
        if res.J_T < J_stop
            res.converged = true
            res.message = @sprintf("F > %.4f reached at iteration %d (J_T = %.3e)",
                                   F_threshold, res.iter, res.J_T)
        end
        return res
    end

    problem = ControlProblem(
        trajectories, tlist;
        J_T = J_T_fn, chi = chi_fn,
        iter_stop = iter_stop,
        check_convergence! = check_convergence!,
    )

    pd = (; problem, sb, ζ, T, tf, tlist, target_states, init_states,
            d_V, g0, Δ_abs, τ, ϕ1, ϕ2)

    init_diag = evaluate_fidelity(pd, initial_controls(pd))
    @printf("Initial-guess (compressed analytic protocol):  F = %.6f\n",
            init_diag.F)
    flush(stdout)

    @printf("\nRunning GRAPE…\n"); flush(stdout)
    res = optimize(problem; method=GRAPE)
    diag = evaluate_fidelity(pd, res.optimized_controls)

    @printf("\n--- Optimisation summary ---\n")
    @printf("Iterations:  %d\n", res.iter)
    @printf("Final J_T:   %.6e   ⇒ F ≈ %.6f\n", res.J_T, 1 - res.J_T)
    @printf("Converged:   %s\n", res.converged ? "yes" : "no")
    res.converged && @printf("Message:     %s\n", res.message)
    @printf("\n--- Re-propagation diagnostics ---\n")
    @printf("F (avg over %d traj): %.8f\n", d_V, diag.F)
    for k in 1:d_V
        @printf("  trajectory %d: |⟨φ|ψ(T)⟩| = %.6f,  arg = %+.4f rad\n",
                k, abs(diag.overlaps[k]), angle(diag.overlaps[k]))
    end
    @printf("Max |1 − ‖ψ(T)‖²|:  %.2e\n", diag.norm_dev)
    flush(stdout)

    plot_pulses(pd, res; save_path="ion_GRAPE_30pct_pulses.png")

    save_path = "ion_GRAPE_30pct_controls.jld2"
    jldsave(save_path;
            ε1 = res.optimized_controls[1],
            ε2 = res.optimized_controls[2],
            ε3 = res.optimized_controls[3],
            ε4 = res.optimized_controls[4],
            tlist = pd.tlist,
            T     = pd.T,
            T_ref = T_ref,
            ζ     = pd.ζ,
            N     = N,
            nmax  = nmax,
            F     = diag.F)
    println("\nControls saved to: $save_path")

    return (; problem_data=pd, opt_result=res, diagnostics=diag,
              T_ref, T)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_30pct()
end
