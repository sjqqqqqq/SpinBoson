# export_jaqalpaw.jl
# Step 3: turn the GRAPE controls of spinboson_grape.jl into the physical
# two-tone Raman drive parameters JaqalPaw needs, as JSON.
#
# ===== THE MAP =====
#
# The simulation writes the drive in the quadrature/spin-axis basis, per ion,
#
#     H_ion = e1*X*Jx + e2*P*Jx + e3*X*Jy + e4*P*Jy,   X = a+a†, P = i(a†−a)
#
# with the e's in angular frequency (rad/ms). A resonant bichromatic Raman
# drive on one ion produces
#
#     H_ion = [A*a + B*a†]*sigma_plus + h.c.,   A = (eta*Omega_red /2)*exp(-i*phi_red)
#                                               B = (eta*Omega_blue/2)*exp(-i*phi_blue)
#
# (sigma_plus = Jx + i*Jy). Matching coefficients gives an INVERTIBLE linear map
#
#     A = (e1 − e4)/2 − i(e2 + e3)/2      red sideband tone
#     B = (e1 + e4)/2 + i(e2 − e3)/2      blue sideband tone
#
# so every GRAPE pulse is exactly realizable as an amplitude- and
# phase-modulated red/blue sideband pair — no residual. `sideband_tones` checks
# the round trip numerically on every export (expect ~1e-14 rad/ms).
#
# Both tones sit on their sideband resonances, so the tone frequencies are
# static and all the time dependence lives in the amplitudes and phases.
#
# ===== UNITS =====
# The JLD2 holds angular frequencies in rad/ms and times in ms. The JSON holds
# ordinary frequencies in Hz and times in s, matching JaqalPaw/QSCOUT:
#
#     H/hbar = 2pi*(r_red /2)*exp(-i*phi_red )*a   *sigma_plus
#            + 2pi*(r_blue/2)*exp(-i*phi_blue)*adag*sigma_plus + h.c.
#
# i.e. r_red/r_blue are the usual sideband Rabi rates (eta*Omega), in Hz, and
# the phases are in degrees.
#
# Usage:
#   julia --project=. export_jaqalpaw.jl                  # both horizons
#   julia --project=. export_jaqalpaw.jl results/foo.jld2

using JLD2
using JSON
using Printf

const RAD_PER_MS_TO_HZ = 1000 / (2π)   # rad/ms -> Hz
const MS_TO_S = 1e-3

# JaqalPaw refuses modulation steps shorter than 4 clock cycles at 409.6 MHz.
const MIN_STEP_S = 10e-9

# ===== TONE DECOMPOSITION =====

"""Decompose one ion's four quadrature controls (rad/ms, ordered X*Jx, P*Jx,
X*Jy, P*Jy) into red/blue sideband tones.

Returns `(r_red, phi_red, r_blue, phi_blue, residual)` with rates in Hz, phases
in degrees, and the max round-trip error in rad/ms."""
function sideband_tones(ε1::Vector{Float64}, ε2::Vector{Float64},
                        ε3::Vector{Float64}, ε4::Vector{Float64})
    @assert length(ε1) == length(ε2) == length(ε3) == length(ε4)
    A = @. (ε1 - ε4) / 2 - im * (ε2 + ε3) / 2
    B = @. (ε1 + ε4) / 2 + im * (ε2 - ε3) / 2

    # The map is exact, so this is a pure sanity check on the algebra.
    resid = max(maximum(abs, @. real(A) + real(B) - ε1),
                maximum(abs, @. imag(B) - imag(A) - ε2),
                maximum(abs, @. -(imag(A) + imag(B)) - ε3),
                maximum(abs, @. real(B) - real(A) - ε4))

    r_red  = @. 2 * abs(A) * RAD_PER_MS_TO_HZ
    r_blue = @. 2 * abs(B) * RAD_PER_MS_TO_HZ
    φ_red  = @. -rad2deg(angle(A))
    φ_blue = @. -rad2deg(angle(B))
    return (r_red, φ_red, r_blue, φ_blue, resid)
end

# ===== EXPORT =====

"""Convert one GRAPE control file into the JaqalPaw-facing JSON description.

Writes alongside the input with a `_jaqalpaw.json` suffix and returns the
dictionary written."""
function export_jaqalpaw(jld2_path::String;
                         out_path::Union{Nothing,String}=nothing,
                         verbose::Bool=true)
    data = load(jld2_path)
    c = data["controls"]
    length(c) == 8 || error("Expected 8 controls (2 ions x 4 quadratures), got $(length(c)).")

    tlist = collect(Float64, data["tlist"])
    nt = length(tlist)
    T = Float64(get(data, "T", tlist[end]))

    # PulseData spreads a modulation list evenly over the pulse, so the sample
    # grid must be uniform for the hardware waveform to match the simulated one.
    dts = diff(tlist)
    spread = maximum(dts) - minimum(dts)
    spread > 1e-9 * maximum(dts) &&
        error("tlist is not uniformly spaced (spread $spread ms); " *
              "PulseData modulation lists assume uniform sampling.")

    step_s = (T / nt) * MS_TO_S
    step_s < MIN_STEP_S &&
        error(@sprintf("%d samples over %.1f us is %.2f ns per step, below the %.0f ns floor.",
                       nt, T * 1e3, step_s * 1e9, MIN_STEP_S * 1e9))

    ions = Dict{String,Any}[]
    max_resid = 0.0
    for i in 1:2
        ε1, ε2, ε3, ε4 = c[4i-3], c[4i-2], c[4i-1], c[4i]
        r_red, φ_red, r_blue, φ_blue, resid = sideband_tones(ε1, ε2, ε3, ε4)
        max_resid = max(max_resid, resid)
        push!(ions, Dict(
            "index"        => i,
            "red_rate_hz"  => r_red,  "red_phase_deg"  => φ_red,
            "blue_rate_hz" => r_blue, "blue_phase_deg" => φ_blue,
        ))
    end

    out = Dict{String,Any}(
        "source"      => "GRAPE controls ($(basename(jld2_path)))",
        "n_samples"   => nt,
        "duration_s"  => T * MS_TO_S,
        "sample_dt_s" => (tlist[2] - tlist[1]) * MS_TO_S,
        "times_s"     => tlist .* MS_TO_S,
        "ions"        => ions,
        "convention"  => "H/hbar = 2pi*(r_red/2)*exp(-i*phi_red)*a*sigma_plus " *
                         "+ 2pi*(r_blue/2)*exp(-i*phi_blue)*adag*sigma_plus + h.c. " *
                         "Rates in Hz, phases in degrees, both sidebands on resonance.",
    )
    for key in ("F", "F_guess", "T_frac", "ζ", "N", "nmax")
        haskey(data, key) && (out[key] = data[key])
    end

    out_path = something(out_path, replace(jld2_path, r"\.jld2$" => "_jaqalpaw.json"))
    open(out_path, "w") do io
        JSON.print(io, out)
    end

    if verbose
        @printf("%s -> %s\n", basename(jld2_path), basename(out_path))
        @printf("  %d samples, T = %.4f ms, dt = %.4f us\n",
                nt, T, (tlist[2] - tlist[1]) * 1e3)
        @printf("  round-trip residual: %.2e rad/ms (should be ~1e-14)\n", max_resid)
        for (i, ion) in enumerate(ions)
            @printf("  ion %d: max red %.3f kHz, max blue %.3f kHz\n",
                    i, maximum(ion["red_rate_hz"]) / 1e3,
                    maximum(ion["blue_rate_hz"]) / 1e3)
        end
    end
    return out
end

# ===== VISUALISATION =====

using Plots

"""Plot the exported drive: sideband rate and phase per ion, versus time.

This is the hardware-facing view of the pulse — what the two individual-
addressing channels actually play. Phases are wrapped to (-180, 180]."""
function plot_tones(out; save_path::String="results/jaqalpaw_tones.png")
    t_us = out["times_s"] .* 1e6

    default(fontfamily="Computer Modern", titlefontsize=10, guidefontsize=9,
            tickfontsize=7, legendfontsize=8, linewidth=1.4, dpi=200)

    rate_max = maximum(maximum(max.(ion["red_rate_hz"], ion["blue_rate_hz"]))
                       for ion in out["ions"]) / 1e3

    panels = Plots.Plot[]
    for row in 1:2, ion in out["ions"]
        i = ion["index"]
        if row == 1
            p = plot(; ylabel="sideband rate [kHz]", title="ion $i",
                       ylims=(-0.02 * rate_max, 1.08 * rate_max),
                       legend=(i == 1 ? :topright : false))
            plot!(p, t_us, ion["blue_rate_hz"] ./ 1e3; color=:royalblue, label="blue")
            plot!(p, t_us, ion["red_rate_hz"] ./ 1e3; color=:crimson, label="red")
        else
            wrap(φ) = @. (φ + 180.0) % 360.0 - 180.0
            p = plot(; xlabel="t (us)", ylabel="phase [deg]",
                       ylims=(-190, 190), yticks=-180:90:180, legend=false)
            plot!(p, t_us, wrap(ion["blue_phase_deg"]); color=:royalblue,
                  seriestype=:scatter, markersize=1.1, markerstrokewidth=0)
            plot!(p, t_us, wrap(ion["red_phase_deg"]); color=:crimson,
                  seriestype=:scatter, markersize=1.1, markerstrokewidth=0)
        end
        push!(panels, p)
    end

    ttl = @sprintf("JaqalPaw drive: %s   T = %.1f us, %d samples",
                   out["source"], out["duration_s"] * 1e6, out["n_samples"])
    fig = plot(panels...; layout=grid(2, 2), size=(1250, 700),
               plot_title=ttl, plot_titlefontsize=11,
               leftmargin=7Plots.mm, bottommargin=4Plots.mm)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("Saved: ", save_path)
    return fig
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    # Controls live in results/; ARGS overrides for a single file.
    results_dir = joinpath(@__DIR__, "results")
    files = isempty(ARGS) ?
            filter(f -> endswith(f, ".jld2"),
                   isdir(results_dir) ? readdir(results_dir, join=true) : String[]) : ARGS
    isempty(files) && error("No .jld2 control files found — run spinboson_grape.jl first.")
    for f in files
        out = export_jaqalpaw(f)
        # spinboson_{grape,analytic}_controls_Tfrac50.jld2 -> "grape_Tfrac50"
        tag = replace(basename(f), "spinboson_" => "", "_controls" => "",
                                   ".jld2" => "")
        plot_tones(out; save_path="results/jaqalpaw_tones_$tag.png")
        println()
    end
end
