# project_grape_2spin.jl
# Spin projection onto |↓↓⟩ applied to EXISTING GRAPE-optimized pulses (the
# eight bilinear controls only — NO carrier, NO π/2 sandwich), at the two saved
# horizons T = 0.5·Tref and T = Tref  (Tref = t_strobo + t_free, the analytic
# protocol's horizon).
#
# Pulses are loaded from results/data/ion_GRAPE_2spin_down_controls_Tfrac{05,10}.jld2
# (produced by ion_GRAPE_2spin.jl, init_spins=:down, rotate=false: GRAPE targets
# the FIXED entangled state D₂(cond)·S₁(cond)|0⟩|↓↓⟩).  Here we DON'T re-optimize;
# we re-propagate the stored pulses at a larger Fock cutoff, PROJECT the spins
# onto |↓↓⟩ (and |↓↑⟩), and score the heralded squeezed cats:
#
#   |↓↓⟩ outcome  →  even squeezed cat  (D(α)+D(−α))S(−ζ/2)|0⟩,  success p_dd
#   |↓↑⟩ outcome  →  odd  squeezed cat  (D(α)−D(−α))S(−ζ/2)|0⟩,  success p_du
#
# Compare to project_2spin_down.jl (the analytic protocol = the un-optimized,
# T=Tref point).
#
# Usage: julia --project=. scripts/project_grape_2spin.jl

include(joinpath(@__DIR__, "ion_GRAPE_2spin.jl"))   # build_spinboson2, control_operators2, protocol_params

# ===== PROJECTION HELPERS (same convention as project_2spin_down.jl) =====

"""Unnormalized boson ket ⟨s1 s2|ψ⟩ for spin outcomes s1, s2 ∈ {:down, :up}."""
function project_spins(sb, ψ, s1::Symbol, s2::Symbol)
    χ1 = s1 === :down ? spindown(sb.b_spin1) : spinup(sb.b_spin1)
    χ2 = s2 === :down ? spindown(sb.b_spin2) : spinup(sb.b_spin2)
    nmax = length(sb.b_fock) - 1
    data = ComplexF64[dagger(fockstate(sb.b_fock, n) ⊗ χ1 ⊗ χ2) * ψ
                      for n in 0:nmax]
    return Ket(sb.b_fock, data)
end

"""Ideal even/odd squeezed-cat references for parameters (ζ, α)."""
function cat_references(b_fock, ζ::Float64, α::Float64)
    sq = squeeze(b_fock, -ζ / 2) * fockstate(b_fock, 0)
    A  = displace(b_fock, α) * sq
    B  = displace(b_fock, -α) * sq
    return (; even=normalize(A + B), odd=normalize(A - B))
end

# ===== LOAD A SAVED PULSE SET, RE-PROPAGATE, PROJECT =====

"""Load the stored 8-channel pulses from `file`, re-propagate |0,↓↓⟩ under them
   at Fock cutoff `nmax`, then project the final state onto |↓↓⟩ / |↓↑⟩ and
   score the heralded cats.  `nmax` may exceed the optimization cutoff so the
   projection isn't distorted by truncation."""
function load_and_project(file::String; N::Int=1, nmax::Int=40,
                          z_target::Float64=0.5, P::Int=1, ℓ::Int=1)
    d        = load(file)
    controls = d["controls"]::Vector
    tlist    = d["tlist"]
    T        = d["T"]
    t_free   = d["t_free"]
    ζ        = d["ζ"]

    sb  = build_spinboson2(N, nmax)
    ψ0  = build_initial2(sb; spins=:down)
    Hc  = control_operators2(sb)
    H   = hamiltonian(collect(zip(Hc, controls))...)
    ψf_vec = propagate(Vector{ComplexF64}(ψ0.data), H, tlist; method=ExpProp)
    ψf  = Ket(sb.b_full, ψf_vec)

    # α from the physical protocol (fixed, independent of the horizon T).
    g0  = protocol_params(N, z_target, P, ℓ).g0
    α   = g0 * t_free / sqrt(2)

    ψ_dd = project_spins(sb, ψf, :down, :down)
    ψ_du = project_spins(sb, ψf, :down, :up)
    p_dd = norm(ψ_dd.data)^2
    p_du = norm(ψ_du.data)^2

    cats  = cat_references(sb.b_fock, ζ, α)
    ψ_ddn = normalize(ψ_dd)
    ψ_dun = normalize(ψ_du)
    F_dd_even = abs2(dagger(cats.even) * ψ_ddn)
    F_du_odd  = abs2(dagger(cats.odd)  * ψ_dun)
    nbar_dd   = real(dagger(ψ_ddn) * (number(sb.b_fock) * ψ_ddn))

    return (; T, t_free, ζ, α, F_stored=d["F"], p_dd, p_du,
              F_dd_even, F_du_odd, nbar_dd, ψ_dd=ψ_ddn, ψ_du=ψ_dun, cats, sb)
end

# ===== SCAN OVER THE SAVED HORIZONS =====

function main_scan(; N::Int=1, nmax::Int=40, z_target::Float64=0.5,
                     P::Int=1, ℓ::Int=1,
                     files=["results/data/ion_GRAPE_2spin_down_controls_Tfrac05.jld2",
                            "results/data/ion_GRAPE_2spin_down_controls_Tfrac10.jld2"],
                     xrange=range(-6.0, 6.0, length=201),
                     prange=range(-6.0, 6.0, length=201),
                     save_path::String="results/figures/project_grape_2spin_scan.png")
    @printf("=== |↓↓⟩ projection of EXISTING GRAPE pulses (8 bilinear, no carrier) ===\n")
    @printf("N=%d, projection nmax=%d, z=%.2f, P=%d, ℓ=%d\n", N, nmax, z_target, P, ℓ)

    pts = [load_and_project(f; N, nmax, z_target, P, ℓ) for f in files]
    Tref = maximum(p.T for p in pts)

    @printf("\n%-10s %-9s %-10s %-13s %-9s %-13s %-8s\n",
            "T/Tref", "F_stored", "p(↓↓)", "F(even|↓↓)", "p(↓↑)", "F(odd|↓↑)", "⟨n⟩_↓↓")
    for p in pts
        @printf("%-10.3f %-9.4f %-10.4f %-13.6f %-9.4f %-13.6f %-8.3f\n",
                p.T / Tref, p.F_stored, p.p_dd, p.F_dd_even,
                p.p_du, p.F_du_odd, p.nbar_dd)
    end
    α, ζ = pts[end].α, pts[end].ζ

    # ===== FIGURE: |↓↓⟩ and |↓↑⟩ Wigners at each horizon =====
    xvec = collect(Float64, xrange)
    pvec = collect(Float64, prange)
    wig(ψ) = wigner(ψ ⊗ dagger(ψ), xvec, pvec)

    default(fontfamily="Computer Modern", titlefontsize=10, guidefontsize=9,
            tickfontsize=7, legendfontsize=7, linewidth=1.4, dpi=200,
            gridalpha=0.25)

    Ws = [(wig(p.ψ_dd), wig(p.ψ_du)) for p in pts]
    cmax = maximum(maximum(max(maximum(abs, w1), maximum(abs, w2)))
                   for (w1, w2) in Ws)

    plts = Plots.Plot[]
    for (p, (Wdd, Wdu)) in zip(pts, Ws)
        push!(plts, heatmap(xvec, pvec, Wdd'; c=:RdBu, clims=(-cmax, cmax),
              xlabel="x", ylabel="p", aspect_ratio=:equal, colorbar=false,
              xlims=(xvec[1], xvec[end]), ylims=(pvec[1], pvec[end]),
              title=@sprintf("|↓↓⟩  T=%.2fTref  p=%.3f  F=%.4f",
                             p.T / Tref, p.p_dd, p.F_dd_even)))
        push!(plts, heatmap(xvec, pvec, Wdu'; c=:RdBu, clims=(-cmax, cmax),
              xlabel="x", ylabel="p", aspect_ratio=:equal, colorbar=true,
              xlims=(xvec[1], xvec[end]), ylims=(pvec[1], pvec[end]),
              title=@sprintf("|↓↑⟩  T=%.2fTref  p=%.3f  F=%.4f",
                             p.T / Tref, p.p_du, p.F_du_odd)))
    end

    fig = plot(plts...; layout=(length(pts), 2), size=(1150, 520 * length(pts)),
               plot_title=@sprintf("GRAPE (8 bilinear, no carrier) + spin projection:  heralded even/odd cats   (ζ=%.2f, α=%.2f)",
                                    ζ, α),
               plot_titlefontsize=12, margin=5Plots.mm)
    savefig(fig, save_path)
    println("\nSaved: ", save_path)
    return (; pts, Tref, α, ζ, fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_scan()
end
