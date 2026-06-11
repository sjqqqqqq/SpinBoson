# test_dephasing_xdisp_polarized.jl
# Master-equation robustness test for the xdisp-target GRAPE pulses
# (T_frac ∈ {1, 0.9, 0.75, 0.5, 1/3}) and the underlying 3-stage analytic
# protocol (strobo squeeze → R(π/2) → free x-displacement), starting from
# |0⟩_b ⊗ |+J⟩ and decohering through bosonic dephasing
#
#     dρ/dt = −i[H(t), ρ] + γ · (n̂ ρ n̂ − ½ {n̂², ρ}),   n̂ = a†a.
#
# Reports F = ⟨ψ_target | ρ(T) | ψ_target⟩ vs γ, where ψ_target is the
# noiseless 3-stage final state.
#
# Usage: julia --project=. test_dephasing_xdisp_polarized.jl

using QuantumOptics
using OrdinaryDiffEq
using LinearAlgebra
using JLD2
using Printf
using Plots

include(joinpath(@__DIR__, "test_dephasing.jl"))    # make_dephasing_dynamic, fidelity_vs_gamma_pulses
                                # (also pulls in ion_GRAPE_displace.jl helpers)

# ----- 3-stage protocol under dephasing -----

function fidelity_vs_gamma_protocol_xdisp(sb, Δ_abs, ϕ1, ϕ2, g0, τ,
                                          t_strobo, tf, P,
                                          ψ0, ψ_target, γ_list)
    Hf_strobo = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=Inf)
    Hf_free   = make_H_dynamic_xdisp(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo)
    R_full    = exp(dense(-1im * (π/2) * (sb.ad * sb.a))) ⊗ one(sb.b_spin)
    Rdag_full = dagger(R_full)

    # Segment boundaries inside the strobo stage.
    tstops_s = Float64[]
    for p in 0:(P-1)
        t0 = 4p * τ
        push!(tstops_s, t0, t0 + τ, t0 + 2τ, t0 + 3τ)
    end
    push!(tstops_s, t_strobo)
    unique!(sort!(tstops_s))

    F = Float64[]
    for γ in γ_list
        if γ == 0.0
            _, ψs = timeevolution.schroedinger_dynamic(
                [0.0, t_strobo], ψ0, Hf_strobo;
                alg=Tsit5(), abstol=1e-10, reltol=1e-10,
                tstops=tstops_s, maxiters=10_000_000)
            ψ_rot = R_full * ψs[end]
            _, ψf = timeevolution.schroedinger_dynamic(
                [t_strobo, tf], ψ_rot, Hf_free;
                alg=Tsit5(), abstol=1e-10, reltol=1e-10,
                tstops=[t_strobo, tf], maxiters=10_000_000)
            push!(F, abs2(dagger(ψ_target) * ψf[end]))
        else
            f_strobo = make_dephasing_dynamic(sb, Hf_strobo, γ)
            _, ρs = timeevolution.master_dynamic(
                [0.0, t_strobo], ψ0, f_strobo;
                alg=Tsit5(), abstol=1e-10, reltol=1e-10,
                tstops=tstops_s, maxiters=10_000_000)
            ρ_rot = R_full * ρs[end] * Rdag_full
            f_free = make_dephasing_dynamic(sb, Hf_free, γ)
            _, ρf = timeevolution.master_dynamic(
                [t_strobo, tf], ρ_rot, f_free;
                alg=Tsit5(), abstol=1e-10, reltol=1e-10,
                tstops=[t_strobo, tf], maxiters=10_000_000)
            push!(F, real(dagger(ψ_target) * (ρf[end] * ψ_target)))
        end
    end
    return F
end

# ----- main -----

function main(; γ_list = [0.0; 10.0 .^ range(-3, 1, length=11) .* (2π)],
                save_prefix::String = "test_dephasing_xdisp_polarized")
    # Same parameters as the xdisp sweep: N=1, z=0.5, P=1, t_free=4τ.
    N, nmax, z_target, P, ℓ = 1, 20, 0.5, 1, 1
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free   = t_strobo
    tf_full  = t_strobo + t_free
    ϕ1, ϕ2   = Float64(π), 0.0

    sb       = build_spinboson(N, nmax)
    ψ0_ket   = build_initial(N, sb; init=:polarized)
    ψtgt_ket = build_target_ext_xdisp(sb, ψ0_ket, Δ_abs, ϕ1, ϕ2, g0, τ,
                                      t_strobo, t_free, P)
    @printf("init=polarized, Target ⟨n⟩=%.4f, ‖ψ_target‖²=%.10f\n",
            real(expect(sb.n_op, ψtgt_ket)),
            real(dagger(ψtgt_ket) * ψtgt_ket))

    pulses = [
        ("protocol",          nothing,                                        1.0,   :black),
        ("GRAPE T_frac=1",    "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac1000.jld2",  1.0,   :purple),
        ("GRAPE T_frac=0.9",  "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac900.jld2",   0.9,   :blue),
        ("GRAPE T_frac=0.75", "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac750.jld2",   0.75,  :green),
        ("GRAPE T_frac=0.5",  "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac500.jld2",   0.5,   :orange),
        ("GRAPE T_frac=1/3",  "results/data/ion_GRAPE_xdisp_pol_controls_Tfrac333.jld2",   1/3,   :red),
    ]

    @printf("\nγ_list/(2π) [kHz] = %s\n",
            string([@sprintf("%.3g", g/(2π)) for g in γ_list]))

    results = NamedTuple[]
    for (label, path, Tfrac, color) in pulses
        @printf("\n--- %s ---\n", label)
        if path === nothing
            F = fidelity_vs_gamma_protocol_xdisp(sb, Δ_abs, ϕ1, ϕ2, g0, τ,
                                                  t_strobo, tf_full, P,
                                                  ψ0_ket, ψtgt_ket, γ_list)
        else
            data = load(path)
            ε1, ε2, ε3, ε4 = data["ε1"], data["ε2"], data["ε3"], data["ε4"]
            tlist = collect(Float64, data["tlist"])
            F = fidelity_vs_gamma_pulses(sb, ε1, ε2, ε3, ε4, tlist,
                                          ψ0_ket, ψtgt_ket, γ_list)
        end
        for (γ, f) in zip(γ_list, F)
            @printf("  γ/(2π) = %8.4g kHz   F = %.6f\n", γ/(2π), f)
        end
        push!(results, (; label, Tfrac, color, F))
    end

    default(fontfamily="Computer Modern", titlefontsize=13, guidefontsize=11,
            tickfontsize=9, legendfontsize=9, linewidth=1.8, dpi=200)

    γ_plot = max.(γ_list, 1e-6) ./ (2π)
    fig = plot(xscale=:log10, xlabel="γ /(2π)  [kHz]", ylabel="Fidelity",
               title="Fidelity vs bosonic dephasing rate (xdisp target, polarized)",
               legend=:bottomleft, ylims=(0.0, 1.02))
    for r in results
        plot!(fig, γ_plot, r.F; label=r.label, color=r.color, marker=:circle)
    end
    hline!(fig, [0.99]; linestyle=:dot, color=:gray, alpha=0.5, label="")
    png_path = "results/figures/$(save_prefix).png"
    jld_path = "results/data/$(save_prefix).jld2"
    savefig(fig, png_path)
    println("\nSaved: ", png_path)

    jldsave(jld_path;
            γ_list = γ_list,
            labels = [r.label for r in results],
            Tfracs = [r.Tfrac for r in results],
            F_data = [r.F   for r in results],
            init   = "polarized",
            target = "xdisp")
    println("Saved: ", jld_path)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
