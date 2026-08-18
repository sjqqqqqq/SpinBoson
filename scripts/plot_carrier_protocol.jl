# plot_carrier_protocol.jl
# Dynamics of the ANALYTIC protocol with the π/2 carrier sandwich (the initial
# guess of ion_GRAPE_2spin_carrier.jl, no GRAPE):
#
#   strobe squeeze on spin1  →  π/2 carrier (|↓⟩₂ → |+⟩_{x+y})  →
#   displacement on spin2    →  inverse π/2 carrier (|+⟩_{x+y} → |↓⟩₂)
#
# Four panels: (1) the pulse sequence, (2) spin dynamics — spin1 stays in |↓⟩
# throughout, spin2 is parked in the (Jx+Jy) eigenstate during the displacement
# and returned to |↓⟩, (3) boson ⟨n⟩, (4) final boson Wigner (spins traced out)
# with the fidelity to the product target D(α)S(−ζ/2)|0⟩⊗|↓↓⟩.
#
# Usage: julia --project=. scripts/plot_carrier_protocol.jl

include(joinpath(@__DIR__, "ion_GRAPE_2spin_carrier.jl"))

"""Propagate the analytic-sandwich guess and return the state at every tlist
   point (columns of a dim × nt matrix)."""
function propagate_guess_storage(pd)
    gen0 = pd.problem.trajectories[1].generator
    return propagate(pd.init_state, gen0, pd.tlist; method=ExpProp, storage=true)
end

function main_carrier_protocol(; nmax::Int=30, nt::Int=2001,
                                 xrange=range(-6.0, 6.0, length=201),
                                 prange=range(-6.0, 6.0, length=201),
                                 save_path::String="results/figures/carrier_protocol_analytic.png")
    pd = build_problem_carrier(; nmax=nmax, nt=nt)
    sb = pd.sb
    t1 = pd.t_strobo + pd.t_rot                 # end of first carrier pulse
    t2 = pd.t_strobo + pd.t_free - pd.t_rot     # start of inverse carrier pulse

    Ψs = propagate_guess_storage(pd)
    ψf_vec = Vector{ComplexF64}(Ψs[:, end])
    F = abs2(dot(pd.target_state, ψf_vec))

    # Observables on the full space (fock ⊗ spin1 ⊗ spin2).
    Ib  = one(sb.b_fock)
    Sz1 = asmat(Ib ⊗ sigmaz(sb.b_spin1) ⊗ sb.Is2)
    Sz2 = asmat(Ib ⊗ sb.Is1 ⊗ sigmaz(sb.b_spin2))
    Sxy2 = asmat(Ib ⊗ sb.Is1 ⊗ ((sigmax(sb.b_spin2) + sigmay(sb.b_spin2)) / sqrt(2)))
    dn   = spindown(sb.b_spin1) ⊗ spindown(sb.b_spin2)
    Pdd  = asmat(Ib ⊗ projector(dn))
    Nop  = asmat((sb.ad * sb.a) ⊗ sb.Is1 ⊗ sb.Is2)

    ev(M) = [real(dot(ψ, M * ψ)) for ψ in eachcol(Ψs)]
    sz1, sz2, sxy2, pdd, nbar = ev(Sz1), ev(Sz2), ev(Sxy2), ev(Pdd), ev(Nop)

    @printf("Analytic protocol + π/2 sandwich:  F = %.6f\n", F)
    @printf("final: p(↓↓) = %.6f, ⟨σz1⟩ = %+.4f, ⟨σz2⟩ = %+.4f, ⟨n⟩ = %.4f\n",
            pdd[end], sz1[end], sz2[end], nbar[end])

    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=9,
            tickfontsize=7, legendfontsize=7, linewidth=1.4, dpi=200,
            gridalpha=0.25)

    stage_lines(plt) = vline!(plt, [pd.t_strobo, t1, t2];
                              color=:black, linestyle=:dot, alpha=0.4, label="")
    carrier_shade(plt) = vspan!(plt, [pd.t_strobo, t1, t2, pd.T];
                                color=:orange, alpha=0.10, label="")

    # --- Panel 1: pulses ---
    init = initial_controls2(pd)
    p_pulse = plot(; xlabel="t (ms)", ylabel="amplitude /(2π) [kHz]",
                   title="analytic pulses + π/2 carrier sandwich",
                   legend=:topleft)
    carrier_shade(p_pulse)
    plot!(p_pulse, pd.tlist, init[1] ./ (2π);
          label="ε1 : X⊗Jx1 (strobe)", color=:steelblue, alpha=0.55, linewidth=1.0)
    plot!(p_pulse, pd.tlist, init[6] ./ (2π);
          label="ε6 = ε8 : P⊗(Jx2,Jy2) (displace)", color=:crimson)
    plot!(p_pulse, pd.tlist, init[9] ./ (2π);
          label="ε9 : Jx2 carrier", color=:black)
    plot!(p_pulse, pd.tlist, init[10] ./ (2π);
          label="ε10 : Jy2 carrier", color=:magenta)
    stage_lines(p_pulse)

    # --- Panel 2: spin dynamics ---
    p_spin = plot(; xlabel="t (ms)", ylabel="expectation value",
                  title="spin dynamics", legend=:left, ylims=(-1.12, 1.12))
    carrier_shade(p_spin)
    plot!(p_spin, pd.tlist, sz1;  label="⟨σz⟩ spin1", color=:blue)
    plot!(p_spin, pd.tlist, sz2;  label="⟨σz⟩ spin2", color=:crimson)
    plot!(p_spin, pd.tlist, sxy2; label="⟨(σx+σy)/√2⟩ spin2", color=:teal)
    plot!(p_spin, pd.tlist, pdd;  label="p(↓↓)", color=:black, linestyle=:dash)
    stage_lines(p_spin)

    # --- Panel 3: boson excitation ---
    n_tgt = pd.α_disp^2 + sinh(pd.ζ / 2)^2
    p_n = plot(; xlabel="t (ms)", ylabel="⟨n⟩",
               title="boson excitation", legend=:topleft)
    carrier_shade(p_n)
    plot!(p_n, pd.tlist, nbar; label="⟨n⟩", color=:purple)
    hline!(p_n, [n_tgt]; label=@sprintf("target %.2f", n_tgt),
           color=:gray, linestyle=:dash)
    stage_lines(p_n)

    # --- Panel 4: final boson Wigner ---
    ψf = Ket(sb.b_full, ψf_vec)
    ρb = ptrace(ψf ⊗ dagger(ψf), [2, 3])
    xvec = collect(Float64, xrange)
    pvec = collect(Float64, prange)
    W = wigner(ρb, xvec, pvec)
    cmax = maximum(abs, W)
    p_W = heatmap(xvec, pvec, W';
                  c=:RdBu, clims=(-cmax, cmax),
                  xlabel="x", ylabel="p",
                  title=@sprintf("final boson Wigner   F = %.4f, ⟨n⟩ = %.2f",
                                 F, nbar[end]),
                  aspect_ratio=:equal, colorbar=true,
                  xlims=(xvec[1], xvec[end]), ylims=(pvec[1], pvec[end]))

    fig = plot(p_pulse, p_spin, p_n, p_W; layout=(2, 2), size=(1300, 950),
               plot_title=@sprintf("Analytic protocol + π/2 carrier sandwich:  |0,↓↓⟩ → D(%.2f)S(%+.2f)|0⟩⊗|↓↓⟩   (T = %.4f ms)",
                                    pd.α_disp, -pd.ζ / 2, pd.T),
               plot_titlefontsize=12, margin=5Plots.mm)
    savefig(fig, save_path)
    println("Saved: ", save_path)

    # Companion figure: the 5×2 per-channel pulse layout of
    # ion_GRAPE_2spin_carrier_pulses.png, analytic sandwich pulses only.
    plot_pulses_carrier(pd;
        controls=init, show_init=false,
        controls_label="analytic + π/2 sandwich",
        title=@sprintf("Analytic protocol + π/2 carrier sandwich   T=%.4f ms   target D(%.2f)S(%+.2f)|0>⊗|dd>   F=%.4f",
                       pd.T, pd.α_disp, -pd.ζ / 2, F),
        save_path="results/figures/ion_GRAPE_2spin_carrier_pulses_analytic.png")

    return (; fig, pd, F, sz1, sz2, sxy2, pdd, nbar)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_carrier_protocol()
end
