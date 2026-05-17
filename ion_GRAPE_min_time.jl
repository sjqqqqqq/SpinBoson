# ion_GRAPE_min_time.jl
# Time-optimal search for N=1, z=1: starting from the analytic stroboscopic
# protocol with P=4 as the initial guess, run GRAPE, then sequentially shrink
# the horizon T and warm-start each shorter-T optimisation by linearly
# resampling the previous optimum onto the new time grid. The shortest T at
# which GRAPE still reaches F ≥ F_threshold is reported.
#
# Usage: julia --project=. ion_GRAPE_min_time.jl

include("ion_GRAPE.jl")

"""Linearly resample a control vector defined on `t_old` onto `t_new`."""
function resample(c::Vector{Float64}, t_old::AbstractVector,
                  t_new::AbstractVector)
    out = similar(t_new, Float64)
    n = length(t_old)
    @inbounds for (i, t) in enumerate(t_new)
        if t <= t_old[1]
            out[i] = c[1]
        elseif t >= t_old[end]
            out[i] = c[end]
        else
            j = searchsortedlast(t_old, t)
            j = clamp(j, 1, n - 1)
            α = (t - t_old[j]) / (t_old[j+1] - t_old[j])
            out[i] = c[j] * (1 - α) + c[j+1] * α
        end
    end
    return out
end

"""Build the problem at horizon T using a *given* initial control set
(vector-of-vectors on `tlist` of length nt). The protocol params (Δ_abs, τ,
g0) are still derived from (N, z_target, P, ℓ) so that physical scales /
labelling stay consistent — only the initial guess and T change."""
function build_problem_with_guess(guess::Vector{Vector{Float64}};
                                  N::Int, nmax::Int, z_target::Float64,
                                  P::Int, ℓ::Int, T::Float64, nt::Int,
                                  init::Symbol, iter_stop::Int,
                                  ϕ1::Float64=Float64(π), ϕ2::Float64=0.0)
    sb = build_spinboson(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ, tf) = pp

    U_target_op = build_target_unitary(ζ, N, sb)
    ψ0_ket      = build_initial(N, sb; init)
    ψtarget_ket = build_target(ζ, N, sb; init)
    init_states   = [Vector{ComplexF64}(ψ0_ket.data)]
    target_states = [Vector{ComplexF64}(ψtarget_ket.data)]
    d_V = 1

    J_T_fn, chi_fn = make_state_transfer_functionals(target_states)
    H1, H2, H3, H4 = control_operators(sb)

    @assert length(guess) == 4
    @assert all(length(g) == nt for g in guess)
    tlist = collect(range(0.0, T, length=nt))

    # Discretised piecewise-linear callable controls from the guess samples.
    function make_callable(arr::Vector{Float64})
        return function (t)
            if t <= tlist[1]
                return arr[1]
            elseif t >= tlist[end]
                return arr[end]
            end
            j = searchsortedlast(tlist, t)
            j = clamp(j, 1, nt - 1)
            α = (t - tlist[j]) / (tlist[j+1] - tlist[j])
            return arr[j] * (1 - α) + arr[j+1] * α
        end
    end
    ε1 = make_callable(guess[1])
    ε2 = make_callable(guess[2])
    ε3 = make_callable(guess[3])
    ε4 = make_callable(guess[4])

    H = hamiltonian((H1, ε1), (H2, ε2), (H3, ε3), (H4, ε4))

    trajectories = Trajectory[]
    for k in 1:d_V
        push!(trajectories,
              Trajectory(init_states[k], H;
                         target_state=target_states[k], prop_method=ExpProp))
    end

    problem = ControlProblem(
        trajectories, tlist;
        J_T = J_T_fn, chi = chi_fn, iter_stop = iter_stop,
    )

    return (; problem, sb, ζ, T, tf, tlist, U_target_op, target_states,
              init_states, control_ops=(H1, H2, H3, H4),
              d_V, init, g0, Δ_abs, τ, ϕ1, ϕ2)
end

# ===== MIN-TIME SEARCH =====

function min_time_search(; N::Int=1, nmax::Int=30, z_target::Float64=1.0,
                          P_init::Int=4, ℓ::Int=1, init::Symbol=:GHZ,
                          nt::Int=400, iter_stop::Int=400,
                          F_threshold::Float64=0.99,
                          T_factors=[1.0, 0.85, 0.75, 0.65, 0.55, 0.50,
                                     0.45, 0.40, 0.35, 0.30])

    pp_ref = protocol_params(N, z_target, P_init, ℓ)
    T_ref  = pp_ref.tf
    @printf("=== Min-time GRAPE search ===\n")
    @printf("N = %d, z = %.3f, F_threshold = %.3f\n", N, z_target, F_threshold)
    @printf("Reference (analytic, P = %d): T_ref = %.6f ms, ζ = %.4f\n",
            P_init, T_ref, pp_ref.ζ)

    # --- Step 0: warm-start at T = T_ref using the analytic guess ---
    pd = build_problem(; N, nmax, z_target, P=P_init, ℓ, T=T_ref, nt,
                        init, iter_stop)
    init_diag = evaluate_fidelity(pd, initial_controls(pd))
    @printf("\n[T = %.6f ms]  initial-guess F = %.6f → GRAPE…\n",
            T_ref, init_diag.F)
    res = optimize(pd.problem; method=GRAPE, iter_stop=iter_stop)
    diag = evaluate_fidelity(pd, res.optimized_controls)
    @printf("  GRAPE: iters = %d,  F = %.6f\n", res.iter, diag.F)

    history = Vector{NamedTuple}()
    push!(history, (; T=T_ref, F_init=init_diag.F, F_grape=diag.F,
                      iters=res.iter,
                      controls=copy.(res.optimized_controls),
                      tlist=copy(pd.tlist),
                      pd, res, diag))

    best = (T=T_ref, F=diag.F, controls=copy.(res.optimized_controls),
            tlist=copy(pd.tlist), pd, res, diag)

    prev_controls = copy.(res.optimized_controls)
    prev_tlist    = copy(pd.tlist)

    # --- Steps 1..: shrink T, warm-start from previous optimum (resampled) ---
    for f in T_factors
        f == 1.0 && continue
        T = f * T_ref
        new_tlist = collect(range(0.0, T, length=nt))
        # Resample previous optimum from prev_tlist[0..T_prev] onto [0..T]:
        # contract by mapping new_tlist linearly to [0..T_prev].
        scale = prev_tlist[end] / T
        sample_t = new_tlist .* scale
        guess = [resample(prev_controls[k], prev_tlist, sample_t) for k in 1:4]

        pd_f = build_problem_with_guess(guess; N, nmax, z_target, P=P_init,
                                         ℓ, T, nt, init, iter_stop)
        init_diag = evaluate_fidelity(pd_f, initial_controls(pd_f))
        @printf("\n[T = %.6f ms  (%.0f%% of T_ref)]  warm-start F = %.6f → GRAPE…\n",
                T, 100f, init_diag.F)
        res_f = optimize(pd_f.problem; method=GRAPE, iter_stop=iter_stop)
        diag_f = evaluate_fidelity(pd_f, res_f.optimized_controls)
        @printf("  GRAPE: iters = %d,  F = %.6f\n", res_f.iter, diag_f.F)

        push!(history, (; T, F_init=init_diag.F, F_grape=diag_f.F,
                          iters=res_f.iter,
                          controls=copy.(res_f.optimized_controls),
                          tlist=copy(pd_f.tlist),
                          pd=pd_f, res=res_f, diag=diag_f))

        if diag_f.F >= F_threshold
            best = (T=T, F=diag_f.F,
                    controls=copy.(res_f.optimized_controls),
                    tlist=copy(pd_f.tlist),
                    pd=pd_f, res=res_f, diag=diag_f)
            prev_controls = copy.(res_f.optimized_controls)
            prev_tlist    = copy(pd_f.tlist)
        else
            @printf("  → below threshold; stopping shrink.\n")
            break
        end
    end

    @printf("\n=== Summary ===\n")
    @printf("%12s  %12s  %10s  %8s\n", "T (ms)", "T/T_ref", "F", "iters")
    for h in history
        @printf("%12.6f  %12.4f  %10.6f  %8d\n",
                h.T, h.T / T_ref, h.F_grape, h.iters)
    end
    @printf("\nBest (shortest with F ≥ %.3f): T = %.6f ms (%.1f%% of T_ref), F = %.6f\n",
            F_threshold, best.T, 100 * best.T / T_ref, best.F)

    return (; history, best, T_ref, ζ=pp_ref.ζ)
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    out = min_time_search(N=1, nmax=20, z_target=1.0, P_init=4, ℓ=1,
                          init=:GHZ, nt=300, iter_stop=200,
                          F_threshold=0.99,
                          T_factors=[1.0, 0.6, 0.4, 0.3, 0.25,
                                     0.20, 0.17, 0.14, 0.12, 0.10])

    # Plot pulses + Fig.4(c)-style for the best (shortest) result
    plot_pulses(out.best.pd, out.best.res;
                save_path="ion_GRAPE_min_time_pulses.png")
    plot_fig4c_style(out.best.pd, out.best.res;
                     save_path="ion_GRAPE_min_time_fig4c.png")

    save_path = "ion_GRAPE_min_time_controls.jld2"
    jldsave(save_path;
            ε1    = out.best.controls[1],
            ε2    = out.best.controls[2],
            ε3    = out.best.controls[3],
            ε4    = out.best.controls[4],
            tlist = out.best.tlist,
            T     = out.best.T,
            T_ref = out.T_ref,
            ζ     = out.ζ,
            N     = 1,
            nmax  = length(out.best.pd.sb.b_fock) - 1,
            F     = out.best.F)
    println("\nBest controls saved to: $save_path")
end
