# project_2spin_down.jl
# What does the boson look like if we PROJECT the spins onto |↓↓⟩ after the
# plain analytic protocol (strobe squeeze on spin1 + conditional displacement
# on spin2, NO carrier sandwich)?
#
# Prediction: during the displacement stage H = g0·P̂·(Jx₂+Jy₂) = (g0/√2)·P̂·σᵤ
# with u = (x+y)/√2, and |↓⟩₂ = (|+u⟩⟨+u| + |−u⟩⟨−u|)|↓⟩₂ splits equally over
# the two branches, so the final state is
#
#   |↓⟩₁ ⊗ [⟨+u|↓⟩·D(+α)|sq⟩⊗|+u⟩ + ⟨−u|↓⟩·D(−α)|sq⟩⊗|−u⟩],  |sq⟩ = S(−ζ/2)|0⟩.
#
# Projecting spin2 onto |↓⟩ resums the branches with weights ⟨↓|±u⟩⟨±u|↓⟩ = ½:
#
#   |↓↓⟩ outcome  →  ½(D(α) + D(−α))|sq⟩   — EVEN squeezed cat,  p ≈ ½
#   |↓↑⟩ outcome  →  odd squeezed cat,                            p ≈ ½
#
# This is the measurement-based (heralded) alternative to the carrier sandwich
# of ion_GRAPE_2spin_carrier.jl: instead of making the displacement
# unconditional, measure spin2 and keep/flip by outcome.
#
# Usage: julia --project=. scripts/project_2spin_down.jl

include(joinpath(@__DIR__, "SpinBoson_test.jl"))   # build_spinboson2, stages, protocol_params

"""Run the two analytic stages (squeeze on spin1, then conditional displacement
   on spin2, no boson rotation, no carrier) from |0⟩⊗|↓↓⟩."""
function run_analytic_protocol(sb, pp, P::Int; ϕ1::Float64=Float64(π),
                               ϕ2::Float64=0.0, t_free_frac::Float64=1.0)
    (; g0, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free   = t_free_frac * t_strobo
    tf       = t_strobo + t_free

    ψ0 = build_initial2(sb; spins=:down)

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

    Hf_d = make_H_disp(sb, g0)
    _, ψf = timeevolution.schroedinger_dynamic(
        [t_strobo, tf], ψs[end], Hf_d;
        alg=Tsit5(), abstol=1e-12, reltol=1e-12,
        tstops=[t_strobo, tf], maxiters=10_000_000)

    α_disp = g0 * t_free / sqrt(2)
    return (; ψf=ψf[end], α_disp, t_strobo, t_free, tf)
end

"""Unnormalized boson ket ⟨s1 s2|ψ⟩ for spin outcomes s1, s2 ∈ {:down, :up}."""
function project_spins(sb, ψ, s1::Symbol, s2::Symbol)
    χ1 = s1 === :down ? spindown(sb.b_spin1) : spinup(sb.b_spin1)
    χ2 = s2 === :down ? spindown(sb.b_spin2) : spinup(sb.b_spin2)
    nmax = length(sb.b_fock) - 1
    data = ComplexF64[dagger(fockstate(sb.b_fock, n) ⊗ χ1 ⊗ χ2) * ψ
                      for n in 0:nmax]
    return Ket(sb.b_fock, data)
end

"""Decompose ψb on the (non-orthogonal) pair A = D(α)|sq⟩, B = D(−α)|sq⟩ via
   the Gram matrix; returns coefficients and the residual norm²."""
function cat_decomposition(ψb, A, B)
    G = [dagger(A)*A dagger(A)*B; dagger(B)*A dagger(B)*B]
    v = [dagger(A)*ψb; dagger(B)*ψb]
    c = G \ v
    resid = norm(ψb.data)^2 - real(dot(v, c))
    return (; c, resid)
end

function main_projection(; N::Int=1, nmax::Int=40, z_target::Float64=0.5,
                           P::Int=1, ℓ::Int=1,
                           xrange=range(-6.0, 6.0, length=201),
                           prange=range(-6.0, 6.0, length=201),
                           save_path::String="results/figures/project_2spin_down.png")
    sb = build_spinboson2(N, nmax)
    pp = protocol_params(N, z_target, P, ℓ)
    ζ = pp.ζ

    @printf("=== Spin projection after the analytic protocol (no carrier) ===\n")
    @printf("N=%d, nmax=%d, z=%.2f, P=%d, ℓ=%d, ζ=%.4f\n",
            N, nmax, z_target, P, ℓ, ζ)

    run = run_analytic_protocol(sb, pp, P)
    ψf, α = run.ψf, run.α_disp
    @printf("t_strobo=%.4f ms, tf=%.4f ms, α=%.4f, ⟨n⟩_final=%.4f\n",
            run.t_strobo, run.tf, α, real(expect(sb.n_op, ψf)))

    # --- projections onto the four spin outcomes ---
    outcomes = [(:down, :down), (:down, :up), (:up, :down), (:up, :up)]
    kets  = Dict(o => project_spins(sb, ψf, o...) for o in outcomes)
    probs = Dict(o => norm(kets[o].data)^2 for o in outcomes)
    @printf("\nOutcome probabilities:  p(↓↓)=%.4f  p(↓↑)=%.4f  p(↑↓)=%.4f  p(↑↑)=%.4f\n",
            probs[(:down, :down)], probs[(:down, :up)],
            probs[(:up, :down)], probs[(:up, :up)])

    ψ_dd = normalize(kets[(:down, :down)])
    ψ_du = normalize(kets[(:down, :up)])

    # --- ideal squeezed-cat references ---
    sq  = squeeze(sb.b_fock, -ζ / 2) * fockstate(sb.b_fock, 0)
    A   = displace(sb.b_fock, α) * sq
    B   = displace(sb.b_fock, -α) * sq
    cat_even = normalize(A + B)
    cat_odd  = normalize(A - B)

    F_dd_even = abs2(dagger(cat_even) * ψ_dd)
    F_dd_odd  = abs2(dagger(cat_odd)  * ψ_dd)
    F_du_even = abs2(dagger(cat_even) * ψ_du)
    F_du_odd  = abs2(dagger(cat_odd)  * ψ_du)
    @printf("\n|↓↓⟩ branch:  F(even cat)=%.6f  F(odd cat)=%.6f\n", F_dd_even, F_dd_odd)
    @printf("|↓↑⟩ branch:  F(even cat)=%.6f  F(odd cat)=%.6f\n", F_du_even, F_du_odd)

    dec = cat_decomposition(ψ_dd, A, B)
    @printf("\n|↓↓⟩ decomposition on {D(+α)|sq⟩, D(−α)|sq⟩}:\n")
    @printf("  c₊=%.4f∠%+.3fπ  c₋=%.4f∠%+.3fπ  rel. phase=%+.3fπ  resid=%.2e\n",
            abs(dec.c[1]), angle(dec.c[1]) / π, abs(dec.c[2]), angle(dec.c[2]) / π,
            angle(dec.c[2] / dec.c[1]) / π, dec.resid)

    # --- figure ---
    xvec = collect(Float64, xrange)
    pvec = collect(Float64, prange)
    default(fontfamily="Computer Modern", titlefontsize=11, guidefontsize=9,
            tickfontsize=7, legendfontsize=7, linewidth=1.4, dpi=200)

    wig(ψ) = wigner(ψ ⊗ dagger(ψ), xvec, pvec)
    Ws = [wig(ψ_dd), wig(ψ_du), wig(cat_even)]
    cmax = maximum(maximum.(abs, Ws))
    titles = [@sprintf("project |↓↓⟩:  p=%.3f,  F(even cat)=%.4f",
                       probs[(:down, :down)], F_dd_even),
              @sprintf("project |↓↑⟩:  p=%.3f,  F(odd cat)=%.4f",
                       probs[(:down, :up)], F_du_odd),
              @sprintf("ideal even cat  (D(%.2f)+D(−%.2f))S(%+.2f)|0⟩",
                       α, α, -ζ / 2)]
    p_W = [heatmap(xvec, pvec, W';
                   c=:RdBu, clims=(-cmax, cmax),
                   xlabel="x", ylabel="p", title=t,
                   aspect_ratio=:equal, colorbar=(k == 3),
                   xlims=(xvec[1], xvec[end]), ylims=(pvec[1], pvec[end]))
           for (k, (W, t)) in enumerate(zip(Ws, titles))]

    Pn(ψ) = abs2.(ψ.data)
    p_fock = plot(; xlabel="n", ylabel="P(n)", title="Fock distribution",
                  legend=:topright)
    plot!(p_fock, 0:nmax, Pn(ψ_dd); label="|↓↓⟩ branch",
          seriestype=:sticks, color=:blue, linewidth=2.5)
    plot!(p_fock, 0:nmax, Pn(cat_even); label="ideal even cat",
          color=:black, linestyle=:dash, marker=:circle, markersize=2.5)

    fig = plot(p_W..., p_fock; layout=(2, 2), size=(1200, 950),
               plot_title=@sprintf("Spin projection after analytic protocol (no carrier):  heralded squeezed cats   (ζ=%.2f, α=%.2f)",
                                    ζ, α),
               plot_titlefontsize=12, margin=5Plots.mm)
    savefig(fig, save_path)
    println("Saved: ", save_path)

    return (; sb, ψf, ψ_dd, ψ_du, probs, cat_even, cat_odd,
              F_dd_even, F_du_odd, dec, α, ζ, fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_projection()
end
