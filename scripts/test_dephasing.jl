# test_dephasing.jl
# Master-equation robustness test for the GRAPE pulses (T_frac ∈ {1, 0.9, 0.75,
# 0.5, 1/3}) and the original analytic squeeze+displacement protocol, all
# evolving |0⟩_b ⊗ |GHZ⟩ to the squeeze+displacement target under bosonic
# dephasing
#
#     dρ/dt = −i[H(t), ρ] + γ · (n̂ ρ n̂ − ½ {n̂², ρ}),   n̂ = a†a.
#
# Reports F = ⟨ψ_target | ρ(T) | ψ_target⟩ for each pulse and γ in a log
# sweep. The target is computed once, noiselessly, from the full analytic
# extended protocol (P=1 strobo + t_free=4τ free segment).
#
# Usage: julia --project=. test_dephasing.jl

using QuantumOptics
using OrdinaryDiffEq
using LinearAlgebra
using JLD2
using Printf
using Plots

include(joinpath(@__DIR__, "..", "src", "ion_GRAPE_displace.jl"))   # SpinBoson_sim helpers + protocol_amplitudes_ext

# ----- noise generator -----

"""Build a `(t, rho) -> (H, Js, Jdaggers)` closure with H from `Hf` and a single
   bosonic dephasing channel √γ·n̂."""
function make_dephasing_dynamic(sb, Hf, γ::Float64)
    n_op  = sb.n_op
    J     = sqrt(γ) * n_op
    Jdag  = dagger(J)
    return function (t, _)
        H = Hf(t, nothing)
        return H, [J], [Jdag]
    end
end

# ----- per-pulse propagation under dephasing -----

function fidelity_vs_gamma_pulses(sb, ε1, ε2, ε3, ε4, tlist,
                                  ψ0, ψ_target, γ_list)
    Hf = make_H_dynamic_from_pulses(sb, ε1, ε2, ε3, ε4, tlist)
    F  = Float64[]
    for γ in γ_list
        if γ == 0.0
            _, ψt = timeevolution.schroedinger_dynamic(
                [0.0, tlist[end]], ψ0, Hf;
                alg=Tsit5(), abstol=1e-10, reltol=1e-10, maxiters=10_000_000)
            ψf = ψt[end]
            push!(F, abs2(dagger(ψ_target) * ψf))
        else
            f_master = make_dephasing_dynamic(sb, Hf, γ)
            _, ρt = timeevolution.master_dynamic(
                [0.0, tlist[end]], ψ0, f_master;
                alg=Tsit5(), abstol=1e-10, reltol=1e-10, maxiters=10_000_000)
            ρf = ρt[end]
            push!(F, real(dagger(ψ_target) * (ρf * ψ_target)))
        end
    end
    return F
end

function fidelity_vs_gamma_protocol(sb, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo, tf,
                                    ψ0, ψ_target, γ_list)
    Hf = make_H_dynamic(sb, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo)
    F  = Float64[]
    for γ in γ_list
        if γ == 0.0
            _, ψt = timeevolution.schroedinger_dynamic(
                [0.0, tf], ψ0, Hf;
                alg=Tsit5(), abstol=1e-10, reltol=1e-10, maxiters=10_000_000)
            push!(F, abs2(dagger(ψ_target) * ψt[end]))
        else
            f_master = make_dephasing_dynamic(sb, Hf, γ)
            _, ρt = timeevolution.master_dynamic(
                [0.0, tf], ψ0, f_master;
                alg=Tsit5(), abstol=1e-10, reltol=1e-10, maxiters=10_000_000)
            push!(F, real(dagger(ψ_target) * (ρt[end] * ψ_target)))
        end
    end
    return F
end

# ----- main -----

function main(; γ_list = [0.0; 10.0 .^ range(-3, 1, length=11) .* (2π)],
                init::Symbol = :GHZ,
                pulses::Union{Nothing,Vector} = nothing,
                save_prefix::String = "test_dephasing")
    # Setup matching the sweep: N=1, z=0.5, P=1, t_free=4τ.
    N, nmax, z_target, P, ℓ = 1, 20, 0.5, 1, 1
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free   = t_strobo            # t_free_frac = 1.0
    tf_full  = t_strobo + t_free
    ϕ1, ϕ2   = Float64(π), 0.0

    sb        = build_spinboson(N, nmax)
    ψ0_ket    = build_initial(N, sb; init=init)
    ψtgt_ket  = build_target_ext(sb, ψ0_ket, Δ_abs, ϕ1, ϕ2, g0, τ,
                                 t_strobo, t_free, P)
    @printf("init = %s, Target ⟨n⟩ = %.4f, ‖ψ_target‖² = %.10f\n",
            init,
            real(expect(sb.n_op, ψtgt_ket)),
            real(dagger(ψtgt_ket) * ψtgt_ket))

    if pulses === nothing
        pulses = [
            ("protocol",         nothing,                                 1.0,   :black),
            ("GRAPE T_frac=1",   "results/data/ion_GRAPE_displace_controls.jld2",      1.0,   :purple),
            ("GRAPE T_frac=0.9", "results/data/ion_GRAPE_displace_controls_Tfrac900.jld2", 0.9,   :blue),
            ("GRAPE T_frac=0.75","results/data/ion_GRAPE_displace_controls_Tfrac750.jld2", 0.75,  :green),
            ("GRAPE T_frac=0.5", "results/data/ion_GRAPE_displace_controls_Tfrac500.jld2", 0.5,   :orange),
            ("GRAPE T_frac=1/3", "results/data/ion_GRAPE_displace_controls_Tfrac333.jld2", 1/3,   :red),
        ]
    end

    @printf("\nγ_list/(2π) [kHz] = %s\n",
            string([@sprintf("%.3g", g/(2π)) for g in γ_list]))

    results = NamedTuple[]
    for (label, path, Tfrac, color) in pulses
        @printf("\n--- %s ---\n", label)
        if path === nothing
            F = fidelity_vs_gamma_protocol(sb, Δ_abs, ϕ1, ϕ2, g0, τ, t_strobo,
                                           tf_full, ψ0_ket, ψtgt_ket, γ_list)
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

    # ----- plot -----
    default(fontfamily="Computer Modern", titlefontsize=13, guidefontsize=11,
            tickfontsize=9, legendfontsize=9, linewidth=1.8, dpi=200)

    γ_plot = max.(γ_list, 1e-6) ./ (2π)   # log axis: replace 0 with tiny value
    fig = plot(xscale=:log10, xlabel="γ /(2π)  [kHz]", ylabel="Fidelity",
               title=@sprintf("Fidelity vs bosonic dephasing rate (init=%s)", init),
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
            init   = String(init))
    println("Saved: ", jld_path)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
