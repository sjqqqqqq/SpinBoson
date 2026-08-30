# plot_jaqalpaw_export.jl
# Slide figures for the analytic-protocol → JaqalPaw conversion.
#
# Produces three PNGs in results/figures/:
#   jaqalpaw_tone_map.png    the ε's and the sideband tones they map to
#   jaqalpaw_four_tone.png   the fixed four-tone spectrum and its phase table
#   jaqalpaw_hardware.png    emulated RFSoC waveform vs the intended one
#
# The third needs the emulator trace:
#   .venv/bin/python jaqal/verify_waveform.py --dump results/data/emulated_waveform.json
#
# Usage: julia --project=. scripts/plot_jaqalpaw_export.jl

include(joinpath(@__DIR__, "export_analytic_jaqalpaw.jl"))

using Plots
using JSON
using Printf

const FIGDIR = "results/figures"

function slide_defaults()
    default(fontfamily="Computer Modern", titlefontsize=13, guidefontsize=11,
            tickfontsize=9, legendfontsize=9, linewidth=2, dpi=200,
            background_color=:white, grid=true, gridalpha=0.15)
end

# ===== FIGURE 1: ε's → sideband tones =====

"""The conversion, in one figure.

Top row repeats the protocol exactly as slide 6 states it — the piecewise
constant Δ(t), ϕ(t), g(t). The two rows below are the same pulse after the
tone map: the red and blue sideband Rabi rates and phases that a JaqalPaw
`PulseData` actually carries."""
function plot_tone_map(; N::Int=1, z_target::Float64=0.5, P::Int=1, ℓ::Int=1,
                        ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                        t_free_frac::Float64=1.0, per_cycle::Int=32,
                        save_path::String=joinpath(FIGDIR, "jaqalpaw_tone_map.png"))
    slide_defaults()
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, Δ_abs, τ) = pp
    t_strobo = pp.tf
    t_free = t_free_frac * t_strobo
    tf = t_strobo + t_free

    tlist, _, _ = sample_grid(t_strobo, τ, per_cycle * 2ℓ; t_free=t_free)
    ε = ntuple(k -> Float64[analytic_amplitudes(t, Δ_abs, ϕ1, ϕ2, g0, τ;
                                                t_strobo=t_strobo)[k]
                            for t in tlist], 4)
    r_red, φ_red, r_blue, φ_blue, _ = sideband_tones(ε...)
    tμ = tlist .* 1e3                                    # ms → µs
    kHz = RAD_PER_MS_TO_HZ / 1e3

    # --- top row: the protocol as slide 6 draws it ---
    tp = collect(range(0.0, tf, length=4000))
    prot = [pulse_params(t, Δ_abs, ϕ1, ϕ2, g0, τ; t_strobo=t_strobo) for t in tp]
    tpμ = tp .* 1e3
    top = Plots.Plot[]
    for (idx, (lab, ylab, col)) in enumerate(
            (("Δ(t)", "Δ / 2π  [kHz]", :orange),
             ("ϕ(t)", "ϕ  [rad]",      :green),
             ("g(t)", "g / 2π  [kHz]", :purple)))
        y = [idx == 2 ? p[2] : p[idx == 1 ? 1 : 3] * kHz for p in prot]
        pl = plot(tpμ, y, xlabel="t  [µs]", ylabel=ylab, title=lab,
                  color=col, legend=false)
        vline!(pl, [t_strobo * 1e3], color=:black, linestyle=:dash, alpha=0.6)
        push!(top, pl)
    end

    # --- bottom rows: the same pulse as two sideband tones ---
    # Red and blue coincide exactly through the displacement stage, so draw the
    # blue trace dashed on top of the red one instead of hiding it.
    p_rate = plot(xlabel="", ylabel="Rabi rate  [kHz]",
                  title="sideband tone amplitudes:  rate = 2|A|, 2|B|",
                  legend=:outerright, ylims=(0, 13.6))
    plot!(p_rate, tμ, r_red ./ 1e3, label="red  |A|", color=:crimson, linewidth=2.5)
    plot!(p_rate, tμ, r_blue ./ 1e3, label="blue |B|", color=:royalblue,
          linewidth=2, linestyle=:dash)

    wrap(x) = mod.(x .+ 180.0, 360.0) .- 180.0
    p_ph = plot(xlabel="t  [µs]", ylabel="phase  [deg]",
                title="sideband tone phases:  φ = −arg A, −arg B",
                legend=:outerright, ylims=(-200, 200))
    plot!(p_ph, tμ, wrap(φ_red), label="φ_red", color=:crimson,
          seriestype=:steppost, linewidth=2.5)
    plot!(p_ph, tμ, wrap(φ_blue), label="φ_blue", color=:royalblue,
          seriestype=:steppost, linewidth=2, linestyle=:dash)

    for p in (p_rate, p_ph)
        vline!(p, [t_strobo * 1e3], color=:black, linestyle=:dash,
               alpha=0.6, label="")
        for s in 1:(4P - 1)
            vline!(p, [s * τ * 1e3], color=:gray, linestyle=:dot,
                   alpha=0.5, label="")
        end
    end
    annotate!(p_rate, t_strobo * 1e3 / 2, 12.3,
              text("squeeze: 4 segments, both tones modulated at 2|Δ|",
                   11, :center, :black))
    annotate!(p_rate, t_strobo * 1e3 + t_free * 1e3 / 2, 12.3,
              text("displace: Δ = 0, both tones constant and equal",
                   11, :center, :black))

    lay = @layout [grid(1, 3); b; c]
    fig = plot(top[1], top[2], top[3], p_rate, p_ph, layout=lay,
               size=(2000, 1150),
               plot_title=@sprintf("Same pulse, two descriptions   (N=%d, z=%.2f, ζ=%.3f, P=%d;  |Δ|/2π = %.1f kHz,  T = %.1f µs)",
                                   N, z_target, pp.ζ, P,
                                   Δ_abs * kHz, tf * 1e3),
               plot_titlefontsize=15, margin=6Plots.mm, left_margin=12Plots.mm)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("wrote $save_path")
    return fig
end

# ===== FIGURE 2: four-tone spectrum =====

"""Stick spectrum of the four fixed drive tones plus the per-segment phase
   table, and the 2×2 beat-note factorization residual that rules the compact
   form out."""
function plot_four_tone(; N::Int=1, z_target::Float64=0.5, P::Int=1, ℓ::Int=1,
                         ϕ1::Float64=Float64(π), ϕ2::Float64=0.0,
                         save_path::String=joinpath(FIGDIR, "jaqalpaw_four_tone.png"))
    slide_defaults()
    pp = protocol_params(N, z_target, P, ℓ)
    (; g0, Δ_abs, τ) = pp
    Δ_khz = Δ_abs * RAD_PER_MS_TO_HZ / 1e3
    amp_khz = abs(g0) * RAD_PER_MS_TO_HZ / 1e3 / 2

    segs = four_tone_phases(Δ_abs, ϕ1, ϕ2, g0, τ, 4P)
    resid = tone_factorization_residual(segs)

    # Panel 1 — the fixed frequency comb. The red and blue sidebands sit a full
    # mode frequency apart, so give each its own zoomed axis rather than
    # stacking both pairs on one offset scale.
    combs = Plots.Plot[]
    for (name, col, sym) in (("red", :crimson, "ω_red"), ("blue", :royalblue, "ω_blue"))
        pc = plot(xlabel="offset from $sym  [kHz]", ylabel="amplitude  [kHz]",
                  title="$name sideband: two tones at ± |Δ|",
                  legend=false, xlims=(-1.8Δ_khz, 1.8Δ_khz),
                  ylims=(0, 1.5amp_khz))
        for xv in (-Δ_khz, +Δ_khz)
            plot!(pc, [xv, xv], [0, amp_khz], color=col, linewidth=7)
            scatter!(pc, [xv], [amp_khz], color=col, markersize=8)
            annotate!(pc, xv, 1.14amp_khz,
                      text(@sprintf("%+.0f kHz", xv), 10, :center, col))
        end
        vline!(pc, [0.0], color=:gray, linestyle=:dash, alpha=0.7)
        annotate!(pc, 0.0, 1.40amp_khz,
                  text(@sprintf("|g0|/2 = %.2f kHz each", amp_khz), 10, :center))
        push!(combs, pc)
    end

    # Panel 2 — phases of the four tones, segment by segment. All four take
    # values in {0, 90, 180, 270}, so nudge the series apart horizontally or
    # they hide behind one another.
    seg_idx = collect(1:length(segs))
    p2 = plot(xlabel="stroboscopic segment", ylabel="tone phase  [deg]",
              title="only the phases change, in steps of 90° across 4P = $(4P) segments",
              legend=:outerright, xlims=(0.4, length(segs) + 0.6),
              ylims=(-40, 320), yticks=0:90:270)
    for (k, (key, lab, col, mk)) in enumerate(
            (("red_minus_phase_deg",  "red  −|Δ|", :crimson,   :circle),
             ("red_plus_phase_deg",   "red  +|Δ|", :darkorange, :diamond),
             ("blue_minus_phase_deg", "blue −|Δ|", :royalblue, :utriangle),
             ("blue_plus_phase_deg",  "blue +|Δ|", :navy,      :square)))
        dx = (k - 2.5) * 0.12
        scatter!(p2, seg_idx .+ dx, [s[key] for s in segs], label=lab,
                 color=col, marker=mk, markersize=6, markerstrokewidth=0.5)
    end

    # Panel 3 — the constraint a 2-tone × 2-tone product would have to satisfy.
    p3 = plot(xlabel="stroboscopic segment",
              ylabel="residual  [deg]",
              title="(φ_blue,+Δ − φ_blue,−Δ) − (φ_red,+Δ − φ_red,−Δ):  must be 0 for a 2×2 product",
              legend=:topright, xlims=(0.5, length(segs) + 0.5), ylims=(-240, 120))
    hline!(p3, [0.0], color=:seagreen, linestyle=:dash,
           label="realizable on 2 global × 2 individual tones")
    plot!(p3, seg_idx, resid, color=:darkorange, marker=:square, markersize=5,
          label="analytic protocol")
    annotate!(p3, length(segs) / 2 + 0.5, -205,
              text("off by exactly −180° on every segment ⇒ needs a third tone on one beam",
                   11, :center, :darkorange))

    lay = @layout [grid(1, 2); b; c]
    fig = plot(combs[1], combs[2], p2, p3, layout=lay, size=(1900, 1150),
               plot_title="The same pulse as a fixed four-tone drive — compact, but not 2×2 factorizable",
               plot_titlefontsize=15, margin=6Plots.mm, left_margin=12Plots.mm)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("wrote $save_path")
    return fig
end

# ===== FIGURE 3: emulated hardware waveform =====

"""Intended vs emulated RFSoC output for both sideband tones."""
function plot_hardware(; trace_path::String="results/data/emulated_waveform.json",
                        save_path::String=joinpath(FIGDIR, "jaqalpaw_hardware.png"))
    slide_defaults()
    isfile(trace_path) || error("Missing $trace_path — run:\n" *
        "  .venv/bin/python jaqal/verify_waveform.py --dump $trace_path")
    d = JSON.parsefile(trace_path)

    t_zoom = 56.4   # µs — the first two stroboscopic segments

    plts = Plots.Plot[]
    for (side, col, tone) in (("blue", :royalblue, 0), ("red", :crimson, 1))
        s = d[side]
        tμ = Float64.(s["t_s"]) .* 1e6
        amp_i = Float64.(s["amp_intended"]); amp_e = Float64.(s["amp_emulated"])
        ph_i  = Float64.(s["phase_intended"]); ph_e = Float64.(s["phase_emulated"])
        z = findall(<=(t_zoom), tμ)

        pa = plot(xlabel="t  [µs]", ylabel="amplitude  [/100]",
                  title=@sprintf("tone %d — %s sideband @ %.3f MHz, full pulse",
                                 tone, side, s["freq_hz"][1] / 1e6),
                  legend=:topright)
        plot!(pa, tμ, amp_i, label="intended", color=col, linewidth=1.5)
        plot!(pa, tμ, amp_e, label="RFSoC emulator", color=:black,
              linestyle=:dash, linewidth=0.9)
        vspan!(pa, [0.0, t_zoom], color=:gray, alpha=0.15, label="")

        pz = plot(xlabel="t  [µs]", ylabel="amplitude  [/100]",
                  title=@sprintf("zoom 0–%.0f µs   (max err %.4f)", t_zoom, s["amp_err"]),
                  legend=:topright)
        plot!(pz, tμ[z], amp_i[z], label="intended", color=col, linewidth=3)
        plot!(pz, tμ[z], amp_e[z], label="RFSoC emulator", color=:black,
              linestyle=:dash, linewidth=1.6)

        pp_ = plot(xlabel="t  [µs]", ylabel="phase  [deg]",
                   title=@sprintf("phase, zoom   (max err %.4f deg)", s["phase_err"]),
                   legend=:topright)
        plot!(pp_, tμ[z], ph_i[z], label="intended", color=col, linewidth=3,
              seriestype=:steppost)
        plot!(pp_, tμ[z], ph_e[z], label="RFSoC emulator", color=:black,
              linestyle=:dash, linewidth=1.6, seriestype=:steppost)
        push!(plts, pa, pz, pp_)
    end

    fig = plot(plts...; layout=(2, 3), size=(1650, 900),
               plot_title=@sprintf("Compiled to RFSoC bytecode (%d × 256-bit words) and replayed through JaqalPaw's firmware emulator",
                                   d["bytecode_words"]),
               plot_titlefontsize=15, margin=6Plots.mm, left_margin=12Plots.mm)
    mkpath(dirname(save_path))
    savefig(fig, save_path)
    println("wrote $save_path")
    return fig
end

# ===== RUN =====
if abspath(PROGRAM_FILE) == @__FILE__
    plot_tone_map()
    plot_four_tone()
    plot_hardware()
end
