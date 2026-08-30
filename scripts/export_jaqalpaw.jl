# export_jaqalpaw.jl
# Convert GRAPE control pulses (results/data/*.jld2) into the physical
# two-tone Raman drive parameters that JaqalPaw needs, and dump them as JSON
# for `python/spinboson_pulses.py`.
#
# ===== THE MAP =====
#
# The simulation Hamiltonian is written in the quadrature/spin-axis basis
#
#     H_ion = ε1·X̂Jx + ε2·P̂Jx + ε3·X̂Jy + ε4·P̂Jy,     X̂ = a+a†, P̂ = i(a†−a)
#
# with the ε's in angular frequency (rad/ms).  A resonant bichromatic Raman
# drive on one ion produces
#
#     H_ion = [A·a + B·a†]·σ₊ + h.c.,   A = (ηΩ_red/2)·e^{−iφ_red}
#                                       B = (ηΩ_blue/2)·e^{−iφ_blue}
#
# (σ₊ = Jx + iJy).  Expanding and matching coefficients gives an invertible
# linear map,
#
#     A = (ε1 − ε4)/2 − i(ε2 + ε3)/2
#     B = (ε1 + ε4)/2 + i(ε2 − ε3)/2
#
# so every GRAPE pulse is realizable as an amplitude- and phase-modulated
# red/blue sideband pair — no residual, and `sideband_tones` checks the round
# trip numerically.  Tone frequencies are static (both sidebands on resonance):
# the whole time dependence lives in the tone amplitudes and phases.
#
# The carrier controls of ion_GRAPE_2spin_carrier.jl,
#
#     H_car = ε9·Jx + ε10·Jy = Ω_c·(cosφ_c·Jx + sinφ_c·Jy)
#
# map onto a single co-propagating tone at the carrier frequency.
#
# ===== UNITS =====
# JLD2 holds angular frequencies in rad/ms and times in ms.  The JSON holds
# ordinary frequencies in Hz and times in s, matching JaqalPaw/QSCOUT.
#
# Exported rates are defined by
#     H/ħ = 2π·(r_red/2)·e^{−iφ_red}·a·σ₊ + 2π·(r_blue/2)·e^{−iφ_blue}·a†·σ₊ + h.c.
#     H/ħ = 2π·(r_car/2)·e^{−iφ_car}·σ₊ + h.c.
# i.e. r_red/blue are the usual sideband Rabi rates ηΩ and r_car the carrier
# Rabi rate (π-time = 1/(2·r_car)).  Phases are in degrees.
#
# Usage:
#   julia --project=. scripts/export_jaqalpaw.jl                      # all files
#   julia --project=. scripts/export_jaqalpaw.jl results/data/foo.jld2

using JLD2
using JSON
using Printf

const RAD_PER_MS_TO_HZ = 1000 / (2π)   # rad/ms → Hz
const MS_TO_S = 1e-3

# ===== TONE DECOMPOSITION =====

"""Decompose one ion's four bilinear controls (rad/ms, ordered
   X̂Jx, P̂Jx, X̂Jy, P̂Jy) into red/blue sideband tones.

Returns `(r_red, φ_red, r_blue, φ_blue)` with rates in Hz and phases in
degrees, plus the max round-trip residual in rad/ms."""
function sideband_tones(ε1::Vector{Float64}, ε2::Vector{Float64},
                        ε3::Vector{Float64}, ε4::Vector{Float64})
    @assert length(ε1) == length(ε2) == length(ε3) == length(ε4)
    A = @. (ε1 - ε4) / 2 - im * (ε2 + ε3) / 2
    B = @. (ε1 + ε4) / 2 + im * (ε2 - ε3) / 2

    # Round trip: the map must be exact, so this is a pure sanity check.
    resid = max(maximum(abs, @. real(A) + real(B) - ε1),
                maximum(abs, @. imag(B) - imag(A) - ε2),
                maximum(abs, @. -(imag(A) + imag(B)) - ε3),
                maximum(abs, @. real(B) - real(A) - ε4))

    r_red   = @. 2 * abs(A) * RAD_PER_MS_TO_HZ
    r_blue  = @. 2 * abs(B) * RAD_PER_MS_TO_HZ
    φ_red   = @. -rad2deg(angle(A))
    φ_blue  = @. -rad2deg(angle(B))
    return (r_red, φ_red, r_blue, φ_blue, resid)
end

"""Carrier controls (ε_x·Jx + ε_y·Jy, rad/ms) → (rate in Hz, phase in deg)."""
function carrier_tone(εx::Vector{Float64}, εy::Vector{Float64})
    r = @. hypot(εx, εy) * RAD_PER_MS_TO_HZ
    φ = @. rad2deg(atan(εy, εx))
    return (r, φ)
end

# ===== JLD2 LOADING =====

"""Pull the control arrays out of a JLD2 file in either of the two layouts
used in results/data: a `controls` vector-of-vectors (8 or 10 entries, the
2-spin runs) or separate `ε1`…`ε4` keys (the single-ion runs).

Returns `(bilinear, carrier)` where `bilinear` is a vector of 4-tuples, one
per driven ion, and `carrier` is `nothing` or a 2-tuple."""
function load_controls(data::Dict)
    if haskey(data, "controls")
        c = data["controls"]
        n = length(c)
        n in (4, 8, 10) || error("Unexpected control count $n; expected 4, 8 or 10.")
        n_ion = n >= 8 ? 2 : 1
        bilinear = [(c[4i-3], c[4i-2], c[4i-1], c[4i]) for i in 1:n_ion]
        carrier = n == 10 ? (c[9], c[10]) : nothing
        return bilinear, carrier
    elseif haskey(data, "ε1")
        bilinear = [(data["ε1"], data["ε2"], data["ε3"], data["ε4"])]
        return bilinear, nothing
    else
        error("No `controls` or `ε1` key found — unrecognised control file.")
    end
end

# ===== EXPORT =====

"""Convert one JLD2 control file into the JaqalPaw-facing JSON description.

`carrier_ion` (1-based) says which ion the carrier controls act on; in
ion_GRAPE_2spin_carrier.jl that is ion 2.  Writes to `out_path` (default:
alongside the input, with a `_jaqalpaw.json` suffix) and returns the
dictionary that was written."""
function export_jaqalpaw(jld2_path::String;
                         out_path::Union{Nothing,String}=nothing,
                         carrier_ion::Int=2,
                         verbose::Bool=true)
    data = load(jld2_path)
    bilinear, carrier = load_controls(data)

    tlist = collect(Float64, data["tlist"])
    nt = length(tlist)
    T = Float64(get(data, "T", tlist[end]))

    # PulseData spreads a modulation list evenly over `dur`, so the nodes must
    # be uniform in time for the hardware waveform to match the simulated one.
    dts = diff(tlist)
    dt_spread = maximum(dts) - minimum(dts)
    dt_spread > 1e-9 * maximum(dts) &&
        error("tlist is not uniformly spaced (spread $(dt_spread) ms); " *
              "PulseData modulation lists assume uniform sampling.")

    ions = Dict{String,Any}[]
    max_resid = 0.0
    for (i, (ε1, ε2, ε3, ε4)) in enumerate(bilinear)
        r_red, φ_red, r_blue, φ_blue, resid = sideband_tones(ε1, ε2, ε3, ε4)
        max_resid = max(max_resid, resid)
        push!(ions, Dict(
            "index"       => i,
            "red_rate_hz" => r_red,   "red_phase_deg"  => φ_red,
            "blue_rate_hz" => r_blue, "blue_phase_deg" => φ_blue,
        ))
    end

    carrier_dict = nothing
    if carrier !== nothing
        r_car, φ_car = carrier_tone(carrier[1], carrier[2])
        carrier_dict = Dict(
            "ion"           => carrier_ion,
            "rate_hz"       => r_car,
            "phase_deg"     => φ_car,
        )
    end

    out = Dict{String,Any}(
        # `source` is the human-readable label both exporters agree on; the
        # analytic export writes it too, so jaqal/ can print either file.
        "source"        => "GRAPE controls ($(basename(jld2_path)))",
        "source_file"   => basename(jld2_path),
        "n_samples"     => nt,
        "duration_s"    => T * MS_TO_S,
        "sample_dt_s"   => (tlist[2] - tlist[1]) * MS_TO_S,
        "times_s"       => tlist .* MS_TO_S,
        "ions"          => ions,
        "carrier"       => carrier_dict,
        "convention"    => "H/hbar = 2pi*(r_red/2)*exp(-i*phi_red)*a*sigma_plus " *
                           "+ 2pi*(r_blue/2)*exp(-i*phi_blue)*adag*sigma_plus + h.c.; " *
                           "carrier H/hbar = 2pi*(r_car/2)*exp(-i*phi_car)*sigma_plus + h.c. " *
                           "Rates in Hz, phases in degrees, both sidebands on resonance.",
    )
    for key in ("F", "ζ", "α", "N", "nmax", "t_strobo", "t_free", "t_rot")
        haskey(data, key) || continue
        v = data[key]
        out[key] = v isa Real && key in ("t_strobo", "t_free", "t_rot") ? v * MS_TO_S : v
    end

    out_path = something(out_path,
                         replace(jld2_path, r"\.jld2$" => "_jaqalpaw.json"))
    open(out_path, "w") do io
        JSON.print(io, out)
    end

    if verbose
        @printf("%s → %s\n", basename(jld2_path), basename(out_path))
        @printf("  %d ion(s), %d samples, T = %.4f ms, dt = %.4f µs\n",
                length(ions), nt, T, (tlist[2] - tlist[1]) * 1e3)
        @printf("  round-trip residual: %.2e rad/ms (should be ~1e-13)\n", max_resid)
        for (i, ion) in enumerate(ions)
            @printf("  ion %d: max red %.3f kHz, max blue %.3f kHz\n",
                    i, maximum(ion["red_rate_hz"]) / 1e3,
                    maximum(ion["blue_rate_hz"]) / 1e3)
        end
        if carrier_dict !== nothing
            @printf("  carrier (ion %d): max %.3f kHz\n",
                    carrier_ion, maximum(carrier_dict["rate_hz"]) / 1e3)
            report_tone_conflicts(ions[carrier_ion], carrier_dict, tlist)
        end
    end
    return out
end

"""Warn about samples where the carrier and the sidebands are both on.

A QSCOUT individual-addressing channel carries two tones.  Both are spoken for
by the red/blue sideband pair, so the carrier can only be driven in windows
where the sidebands are idle — `python/spinboson_pulses.py` puts it on tone 1.
Overlapping samples cannot be played as written."""
function report_tone_conflicts(ion::Dict, carrier::Dict, tlist::Vector{Float64};
                               rel_tol::Float64=0.01)
    sb = max.(ion["red_rate_hz"], ion["blue_rate_hz"])
    car = carrier["rate_hz"]
    sb_thr = rel_tol * maximum(sb)
    car_thr = rel_tol * maximum(car)
    idx = findall(i -> sb[i] > sb_thr && car[i] > car_thr, eachindex(tlist))
    if isempty(idx)
        @printf("  tone check: carrier and sidebands never overlap — 2 tones suffice ✓\n")
    else
        @printf("  tone check: ⚠ %d/%d samples drive carrier AND sidebands together\n",
                length(idx), length(tlist))
        @printf("    (t = %.4f–%.4f ms; needs a 3rd tone, or re-run GRAPE with the\n",
                tlist[idx[1]], tlist[idx[end]])
        @printf("     carrier constrained to the sideband-idle windows)\n")
    end
    return idx
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    files = isempty(ARGS) ?
            filter(f -> endswith(f, ".jld2") && !startswith(basename(f), "test_"),
                   readdir("results/data", join=true)) :
            ARGS
    for f in files
        try
            export_jaqalpaw(f)
        catch e
            @printf("%s → skipped (%s)\n", basename(f), sprint(showerror, e))
        end
        println()
    end
end
