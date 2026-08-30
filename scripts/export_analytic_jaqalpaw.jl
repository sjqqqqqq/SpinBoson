# export_analytic_jaqalpaw.jl
# Export the *analytic* stroboscopic protocol (Fig.4(c) of arXiv:2510.25870,
# `pulse_params` in src/SpinBoson_sim.jl) as JaqalPaw drive parameters.
#
# This is the reference case for the GRAPE exporter in export_jaqalpaw.jl: the
# pulse is known in closed form, so every step of the conversion can be checked
# analytically and numerically.
#
# ===== WHAT THE ANALYTIC PULSE LOOKS LIKE IN TONE SPACE =====
#
# Eq.(23) reads H = g(t)·a·[Jx e^{−iΔt} + Jy e^{+iΔt}e^{−iϕ}] + h.c.  Rewriting
# with σ₊ = Jx + iJy gives H = A(t)·a·σ₊ + B(t)·a†·σ₊ + h.c. with
#
#     A(t) = (g/2)[e^{−iΔt} − i·e^{+i(Δt−ϕ)}]        (red sideband)
#     B(t) = (g/2)[e^{+iΔt} − i·e^{−i(Δt−ϕ)}]        (blue sideband)
#
# — each sideband carries *two* frequency components, at ±Δ about resonance.
# So the analytic protocol is a four-tone drive on the fixed frequency grid
#
#     {ω_red − |Δ|, ω_red + |Δ|, ω_blue − |Δ|, ω_blue + |Δ|},
#
# all four at constant amplitude |g₀|/2, with only their *phases* stepping
# between the 4P stroboscopic segments.  That is the compact description
# exported under "four_tone".
#
# Two caveats, both checked by this script:
#
#  * The four tone phases satisfy no 2×2 product relation: with individual-beam
#    tones carrying {red, blue} and global-beam tones carrying {−Δ, +Δ}, a beat
#    note's phase is φ_individual − φ_global, which forces
#        (φ_blue,+Δ − φ_blue,−Δ) = (φ_red,+Δ − φ_red,−Δ).
#    The analytic pulse violates this by exactly π, independent of ϕ and of the
#    segment.  So the protocol is *not* realizable as a 2-tone global × 2-tone
#    individual product — it needs three tones on one beam.
#
#  * It *is* exactly realizable as two on-resonance sideband tones that are
#    amplitude- and phase-modulated at 2|Δ| (the "tones" block, same schema as
#    export_jaqalpaw.jl).  That is what a QSCOUT individual-addressing channel
#    can play today, and what python/spinboson_pulses.py emits.
#
# Usage:
#   julia --project=. scripts/export_analytic_jaqalpaw.jl
#   julia --project=. -i -e 'include("scripts/export_analytic_jaqalpaw.jl")'
#     export_analytic(N=1, z_target=0.5, P=5)
#     verify_analytic_export(N=1, z_target=0.5, P=5)

include(joinpath(@__DIR__, "..", "src", "SpinBoson_sim.jl"))  # protocol_params, pulse_params
include(joinpath(@__DIR__, "export_jaqalpaw.jl"))             # sideband_tones, unit constants

using JSON
using Printf

# ===== ANALYTIC CONTROLS =====

"""The four quadrature/spin amplitudes (X̂Jx, P̂Jx, X̂Jy, P̂Jy, rad/ms) of the
   analytic protocol at time `t`. Same expression as `protocol_amplitudes` in
   src/ion_GRAPE.jl, repeated here so this script doesn't drag in GRAPE."""
@inline function analytic_amplitudes(t::Float64, Δ_abs::Float64,
                                     ϕ1::Float64, ϕ2::Float64,
                                     g0::Float64, τ::Float64;
                                     t_strobo::Float64=Inf)
    Δ_eff, ϕ_eff, g_eff = pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ;
                                       t_strobo=t_strobo)
    θ = Δ_eff * t
    return ( g_eff * cos(θ),
             g_eff * sin(θ),
             g_eff * cos(θ - ϕ_eff),
            -g_eff * sin(θ - ϕ_eff) )
end

"""Bin-midpoint sample grid for the whole pulse.

A JaqalPaw modulation list of `nt` entries is `nt` equal-duration steps across
the pulse, so the value that belongs in entry i is the pulse evaluated at the
*middle* of bin i — not at its edge. Sampling on the edges would land exactly
on the segment discontinuities of `pulse_params`, where the value is ambiguous.

The bin width is fixed at τ/`per_seg` so no bin ever straddles a stroboscopic
segment boundary; the grid then runs on unchanged through the free-evolution
stage, whose length must be a whole number of bins."""
function sample_grid(t_strobo::Float64, τ::Float64, per_seg::Int;
                     t_free::Float64=0.0)
    dt = τ / per_seg
    n_free = round(Int, t_free / dt)
    abs(n_free * dt - t_free) > 1e-9 * max(dt, t_free) &&
        error("t_free = $t_free ms is not a whole number of $(dt) ms bins.")
    nt = round(Int, t_strobo / dt) + n_free
    return (collect((0.5:1:(nt - 0.5)) .* dt), dt, nt)
end

# ===== FOUR-TONE (FREQUENCY-DOMAIN) DESCRIPTION =====

"""Per-segment phases of the four fixed drive tones.

Returns a vector of dictionaries, one per stroboscopic segment, each giving the
segment's time window and the phase (degrees) of the tone at each of the four
offsets `ω_{red,blue} ± |Δ|`. Amplitudes are |g₀|/2 on every tone at all times;
a negative `g_eff` shows up as a 180° phase, not a negative amplitude."""
function four_tone_phases(Δ_abs::Float64, ϕ1::Float64, ϕ2::Float64,
                          g0::Float64, τ::Float64, n_seg::Int)
    segs = Dict{String,Any}[]
    for s in 1:n_seg
        t_mid = (s - 0.5) * τ
        Δ_eff, ϕ_eff, g_eff = pulse_params(t_mid, Δ_abs, ϕ1, ϕ2, g0, τ)
        sgn = g_eff >= 0 ? 0.0 : 180.0
        ϕ_deg = rad2deg(ϕ_eff)

        # A's e^{−iΔ_eff t} component sits at ω_red + Δ_eff; its partner at
        # ω_red − Δ_eff. Likewise B's e^{+iΔ_eff t} lands at ω_blue − Δ_eff.
        # Δ_eff flips sign per segment, so resolve which side each lands on.
        hi = Δ_eff >= 0     # true ⇒ the "unshifted-phase" red tone is at +|Δ|
        red_hi_phase  = hi ? sgn : (-90.0 - ϕ_deg + sgn)
        red_lo_phase  = hi ? (-90.0 - ϕ_deg + sgn) : sgn
        blue_hi_phase = hi ? (-90.0 + ϕ_deg + sgn) : sgn
        blue_lo_phase = hi ? sgn : (-90.0 + ϕ_deg + sgn)

        push!(segs, Dict(
            "segment"        => s,
            "t_start_s"      => (s - 1) * τ * MS_TO_S,
            "t_end_s"        => s * τ * MS_TO_S,
            "detuning_hz"    => Δ_eff * RAD_PER_MS_TO_HZ,
            "phi_deg"        => ϕ_deg,
            "g_hz"           => g_eff * RAD_PER_MS_TO_HZ,
            "red_minus_phase_deg"  => mod(red_lo_phase,  360.0),
            "red_plus_phase_deg"   => mod(red_hi_phase,  360.0),
            "blue_minus_phase_deg" => mod(blue_lo_phase, 360.0),
            "blue_plus_phase_deg"  => mod(blue_hi_phase, 360.0),
        ))
    end
    return segs
end

"""Check whether the four tone phases factor as (individual ⊗ global) beat
   notes, i.e. whether the pulse fits on 2 global + 2 individual tones.

A beat note's phase is φ_individual − φ_global, so with the individual beam
carrying {red, blue} and the global beam carrying {−|Δ|, +|Δ|} the four phases
must satisfy (blue₊ − blue₋) − (red₊ − red₋) ≡ 0 (mod 2π). Returns the residual
in degrees for each segment."""
function tone_factorization_residual(segs::Vector{Dict{String,Any}})
    return [mod((s["blue_plus_phase_deg"] - s["blue_minus_phase_deg"]) -
                (s["red_plus_phase_deg"]  - s["red_minus_phase_deg"]) + 180.0,
                360.0) - 180.0
            for s in segs]
end

# ===== EXPORT =====

"""Export the analytic protocol as JaqalPaw drive parameters.

Defaults reproduce the protocol shown on slide 6 of `docs/SpinBoson.pptx`:
P = 1 stroboscopic cycle followed by an equally long free-evolution stage
(`t_free_frac = 1`) during which `pulse_params` holds Δ = 0, ϕ = 0, g = +g₀ —
the displacement that turns S|0⟩ into D·R·S|0⟩.

`per_cycle` is the number of samples per 2|Δ| modulation period in the two-tone
representation. Returns the dictionary that was written."""
function export_analytic(; N::Int=1, z_target::Float64=0.5, P::Int=1, ℓ::Int=1,
                          ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                          t_free_frac::Float64=1.0,
                          per_cycle::Int=32,
                          out_path::String="results/data/analytic_jaqalpaw.json",
                          verbose::Bool=true)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, ζ, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free = t_free_frac * t_strobo
    tf = t_strobo + t_free
    n_seg = 4P

    # Each segment holds ℓ periods of e^{iΔt}, hence 2ℓ periods of the 2|Δ|
    # amplitude modulation that the two-tone representation has to track.
    per_seg = per_cycle * 2ℓ
    tlist, dt, nt = sample_grid(t_strobo, τ, per_seg; t_free=t_free)

    ε = ntuple(k -> Float64[analytic_amplitudes(t, Δ_abs, ϕ1, ϕ2, g0, τ;
                                                t_strobo=t_strobo)[k]
                            for t in tlist], 4)
    r_red, φ_red, r_blue, φ_blue, resid = sideband_tones(ε...)

    segs = four_tone_phases(Δ_abs, ϕ1, ϕ2, g0, τ, n_seg)
    fac_resid = tone_factorization_residual(segs)

    # During the free stage the drive is a plain resonant two-tone pulse:
    # Δ = 0 collapses both ±|Δ| components onto their sideband.
    free_stage = t_free <= 0 ? nothing : Dict(
        "t_start_s"   => t_strobo * MS_TO_S,
        "t_end_s"     => tf * MS_TO_S,
        "rate_hz"     => 2 * abs(g0) / sqrt(2) * RAD_PER_MS_TO_HZ,
        "phase_deg"   => 45.0,
        "description" => "displacement drive: Δ=0, ϕ=0, g=+g0 ⇒ both tones " *
                         "constant and equal",
    )

    out = Dict{String,Any}(
        "source"      => "analytic stroboscopic protocol (pulse_params)",
        "N"           => N, "z_target" => z_target, "P" => P, "l" => ℓ,
        "zeta"        => ζ,
        "phi1_deg"    => rad2deg(ϕ1), "phi2_deg" => rad2deg(ϕ2),
        "g0_hz"       => g0 * RAD_PER_MS_TO_HZ,
        "detuning_hz" => Δ_abs * RAD_PER_MS_TO_HZ,
        "tau_s"       => τ * MS_TO_S,
        "t_strobo_s"  => t_strobo * MS_TO_S,
        "t_free_s"    => t_free * MS_TO_S,
        "duration_s"  => tf * MS_TO_S,
        "n_samples"   => nt,
        "sample_dt_s" => dt * MS_TO_S,
        "times_s"     => tlist .* MS_TO_S,
        "free_stage"  => free_stage,
        "ions" => [Dict(
            "index"        => 1,
            "red_rate_hz"  => r_red,  "red_phase_deg"  => φ_red,
            "blue_rate_hz" => r_blue, "blue_phase_deg" => φ_blue,
        )],
        "carrier" => nothing,
        "four_tone" => Dict(
            "tone_rate_hz"          => abs(g0) * RAD_PER_MS_TO_HZ,
            "offsets_hz"            => [-Δ_abs, Δ_abs] .* RAD_PER_MS_TO_HZ,
            "segments"              => segs,
            "factorizable"          => all(<(1e-6), abs.(fac_resid)),
            "factorization_residual_deg" => fac_resid,
        ),
        "convention" => "H/hbar = 2pi*(r_red/2)*exp(-i*phi_red)*a*sigma_plus " *
                        "+ 2pi*(r_blue/2)*exp(-i*phi_blue)*adag*sigma_plus + h.c. " *
                        "Rates in Hz, phases in degrees, both sideband tones on resonance.",
    )

    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, out)
    end

    if verbose
        @printf("=== Analytic protocol → JaqalPaw ===\n")
        @printf("N = %d, z = %.3f, P = %d, ℓ = %d, ϕ₁ = %.1f°, ϕ₂ = %.1f°\n",
                N, z_target, P, ℓ, rad2deg(ϕ1), rad2deg(ϕ2))
        @printf("g₀ = %.3f kHz, |Δ| = %.3f kHz, τ = %.4f µs\n",
                g0 * RAD_PER_MS_TO_HZ / 1e3, Δ_abs * RAD_PER_MS_TO_HZ / 1e3,
                τ * 1e3)
        @printf("t_strobo = %.4f µs + t_free = %.4f µs  ⇒  T = %.4f µs\n",
                t_strobo * 1e3, t_free * 1e3, tf * 1e3)
        @printf("\ntwo-tone (AM/PM) form: %d samples, dt = %.2f ns, %d per segment\n",
                nt, dt * 1e6, per_seg)
        @printf("  sideband rate range: red %.3f–%.3f kHz, blue %.3f–%.3f kHz\n",
                minimum(r_red)/1e3, maximum(r_red)/1e3,
                minimum(r_blue)/1e3, maximum(r_blue)/1e3)
        @printf("  round-trip residual: %.2e rad/ms\n", resid)
        min_step_ns = dt * 1e6
        @printf("  min step %.2f ns vs JaqalPaw floor 10 ns: %s\n",
                min_step_ns, min_step_ns >= 10 ? "ok ✓" : "TOO FAST ✗")

        @printf("\nfour-tone (fixed-frequency) form: %d segments, all tones at %.3f kHz\n",
                n_seg, abs(g0) * RAD_PER_MS_TO_HZ / 1e3 / 2)
        @printf("  offsets: ω_red ± %.3f kHz, ω_blue ± %.3f kHz\n",
                Δ_abs * RAD_PER_MS_TO_HZ / 1e3, Δ_abs * RAD_PER_MS_TO_HZ / 1e3)
        @printf("  2×2 beat-note factorization: %s (residual %+.1f° on every segment)\n",
                all(<(1e-6), abs.(fac_resid)) ? "ok ✓" : "IMPOSSIBLE ✗",
                fac_resid[1])
        if !all(<(1e-6), abs.(fac_resid))
            @printf("  ⇒ needs 3 tones on one beam; use the two-tone AM/PM form instead.\n")
        end
        if free_stage !== nothing
            @printf("\nfree-evolution stage: Δ=0, both tones constant at %.3f kHz, phase %+.0f°\n",
                    free_stage["rate_hz"] / 1e3, free_stage["phase_deg"])
        end
        @printf("\nwrote %s\n", out_path)
    end
    return out
end

# ===== VERIFICATION =====

"""Rebuild the ε's from the exported tone amplitudes and phases, and check
   both the algebra and the four-tone phase table against direct evaluation."""
function verify_analytic_export(; N::Int=1, z_target::Float64=0.5, P::Int=1,
                                 ℓ::Int=1, ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                                 t_free_frac::Float64=1.0, per_cycle::Int=32)
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free = t_free_frac * t_strobo
    tlist, _, _ = sample_grid(t_strobo, τ, per_cycle * 2ℓ; t_free=t_free)
    ε = ntuple(k -> Float64[analytic_amplitudes(t, Δ_abs, ϕ1, ϕ2, g0, τ;
                                                t_strobo=t_strobo)[k]
                            for t in tlist], 4)
    r_red, φ_red, r_blue, φ_blue, _ = sideband_tones(ε...)

    # tones → complex coefficients → ε, i.e. the inverse of sideband_tones
    A = @. (r_red  / 2 / RAD_PER_MS_TO_HZ) * cis(-deg2rad(φ_red))
    B = @. (r_blue / 2 / RAD_PER_MS_TO_HZ) * cis(-deg2rad(φ_blue))
    ε_back = (real.(A) .+ real.(B),
              imag.(B) .- imag.(A),
              .-(imag.(A) .+ imag.(B)),
              real.(B) .- real.(A))
    err_eps = maximum(maximum(abs, ε[k] .- ε_back[k]) for k in 1:4)

    # four-tone table → A(t), B(t), compared against the direct values
    segs = four_tone_phases(Δ_abs, ϕ1, ϕ2, g0, τ, 4P)
    amp = abs(g0) / 2
    ω = Δ_abs
    err_tone = 0.0
    for (i, t) in enumerate(tlist)
        t >= t_strobo && continue        # free stage is not a four-tone segment
        s = segs[min(Int(floor(t / τ)) + 1, length(segs))]
        A_rec = amp * (cis(-ω * t) * cis(deg2rad(s["red_plus_phase_deg"])) +
                       cis(+ω * t) * cis(deg2rad(s["red_minus_phase_deg"])))
        B_rec = amp * (cis(+ω * t) * cis(deg2rad(s["blue_minus_phase_deg"])) +
                       cis(-ω * t) * cis(deg2rad(s["blue_plus_phase_deg"])))
        err_tone = max(err_tone, abs(A[i] - A_rec), abs(B[i] - B_rec))
    end

    @printf("=== Verification ===\n")
    @printf("tone round trip (ε → tones → ε):  max err %.2e rad/ms\n", err_eps)
    @printf("four-tone table → A(t), B(t):     max err %.2e rad/ms  (scale g₀ = %.2f)\n",
            err_tone, g0)
    return (; err_eps, err_tone)
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    export_analytic()
    println()
    verify_analytic_export()
end
