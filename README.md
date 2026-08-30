# Spin-dependent squeezing → QSCOUT pulses

Julia simulation of the spin-dependent squeezing protocol for trapped ions from
[arXiv:2510.25870](https://arxiv.org/abs/2510.25870) (Bond *et al.*) — Hamiltonian
Eq. (23), pulse protocol Fig. 4(c), parameters from Fig. 5 — plus the export path that
turns those pulses into QSCOUT hardware waveforms via
[JaqalPaw](https://github.com/sandialabs/JaqalPaw). `ref1.pdf` is the reference paper.

This branch carries three things and the code needed to reproduce them:

| | what | where it comes from |
|---|---|---|
| 1 | **Analytic pulse** — the closed-form stroboscopic protocol | `scripts/export_analytic_jaqalpaw.jl` |
| 2 | **GRAPE pulse** — numerically optimized two-spin controls | `scripts/ion_GRAPE_2spin.jl` → `results/data/*.jld2` |
| 3 | **Jaqal conversion** — either pulse compiled to RFSoC bytecode and checked against the firmware emulator | `scripts/export_jaqalpaw.jl` + `jaqal/` |

The GRAPE control files are committed (they take ~10 min each to regenerate), so you can
go straight from a clone to hardware waveforms without running an optimization.

> **Always run from the repo root.** `include` paths use `@__DIR__` and work from
> anywhere, but data and figure paths are relative to the repo root.

---

## Setup

Julia 1.11 and Python 3.12 were used; `Manifest.toml` pins the exact Julia package versions.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # Julia deps
bash scripts/setup_jaqalpaw.sh                        # clones JaqalPaw into third_party/, builds .venv/
```

`setup_jaqalpaw.sh` creates `third_party/JaqalPaw` and `.venv` (both git-ignored) and
installs `jaqalpaw[emulator]` — currently 1.3.0a0 — in editable mode.

---

## 1. The analytic pulse

Eq. (23) reads `H = g(t)·a·[Jx e^{−iΔt} + Jy e^{+iΔt}e^{−iϕ}] + h.c.`, driven as a
stroboscopic 4-segment cycle (`τ = 2πℓ/|Δ|`; the signs of Δ and g flip on a
`(+,+), (−,+), (+,−), (−,−)` schedule and the phase alternates between ϕ₁ and ϕ₂).
`pulse_params` in `src/SpinBoson_sim.jl` is the single source of truth for that cycle.

Simulate it (dense-matrix reference implementation, self-contained):

```bash
julia --project=. scripts/ion_test.jl        # simulate + 4-panel figures for z=1.0, z=0.3
```

or interactively, which is preferable since operator construction and ODE compilation are
expensive on the first call:

```julia
julia --project=. -i -e 'include("scripts/ion_test.jl")'

res = simulate(N=1, nmax=20, z_target=1.0, P=5)   # main entry point
plot_results(res; save_path="out.png")
sweep_P(N=1, z_target=1.0, P_range=1:20)          # find the min P with F ≥ 0.99
```

Export it as a drive description:

```bash
julia --project=. scripts/export_analytic_jaqalpaw.jl   # → results/data/analytic_jaqalpaw.json
```

Defaults are `N=1, z=0.5, P=1, ℓ=1` with `t_free_frac=1`, giving `g₀ = 5 kHz`,
`|Δ|/2π = 35.449 kHz`, `τ = 28.21 µs`, `t_strobo = 112.84 µs` and `T = 225.68 µs`.
Override them in the REPL: `export_analytic(N=1, z_target=0.5, P=5)`.

## 2. The GRAPE pulse

`scripts/ion_GRAPE_2spin.jl` runs state-to-state GRAPE for the two-spin system,

```
ψ₀    = |0⟩_b ⊗ |s⟩₁ ⊗ |s⟩₂
ψ_tgt = D₂(cond) · [R] · S₁(cond) · ψ₀
```

over all **eight** bilinear controls (`X̂⊗Jx, P̂⊗Jx, X̂⊗Jy, P̂⊗Jy` on each spin, with
`X̂ = a + a†` and `P̂ = i(a† − a)`), starting from the analytic sequence as the initial
guess. Running it takes roughly 50 s per iteration and ~10 min in total:

```bash
julia --project=. scripts/ion_GRAPE_2spin.jl   # writes results/data/ion_GRAPE_2spin_down_controls.jld2
```

You don't need to. Four optimized runs are committed, all at `N=1, nmax=20, z=0.5,
ζ=1.0, P=1`, 250 time steps, 8 controls:

| file | T | F | notes |
|---|---|---|---|
| `ion_GRAPE_2spin_controls.jld2` | 225.68 µs | 0.9907 | **the featured run** — `init_spins=:up`, `rotate=true`, i.e. the target includes the π/2 boson rotation |
| `ion_GRAPE_2spin_down_controls_Tfrac10.jld2` | 225.68 µs | 0.9999 | the script's current default (`init_spins=:down`, `rotate=false`): with spin 1 in \|↓⟩ the conditional squeeze acts as S(−ζ/2), so no boson rotation is needed and the analytic guess is exactly representable |
| `ion_GRAPE_2spin_controls_Tfrac05.jld2` | 112.84 µs | 0.9919 | half-duration, `:up` |
| `ion_GRAPE_2spin_down_controls_Tfrac05.jld2` | 112.84 µs | 0.9925 | half-duration, `:down` |

Re-render the pulse figures from the saved controls without re-optimizing:

```bash
julia --project=. scripts/replot_2spin_pulses.jl   # 3 PNGs in results/figures/
```

## 3. Converting to Jaqal

### The physical map

The simulation writes the drive in the quadrature/spin-axis basis,
`H = ε1·X̂Jx + ε2·P̂Jx + ε3·X̂Jy + ε4·P̂Jy`. A resonant bichromatic Raman drive gives
`H = A·a·σ₊ + B·a†·σ₊ + h.c.`, and matching coefficients yields an **invertible** linear map

```
A = (ε1 − ε4)/2 − i(ε2 + ε3)/2      red sideband tone
B = (ε1 + ε4)/2 + i(ε2 − ε3)/2      blue sideband tone
```

so any pulse — analytic or GRAPE — is exactly realizable as two on-resonance sideband
tones with amplitude and phase modulation, no residual. `sideband_tones` in
`scripts/export_jaqalpaw.jl` checks the round trip numerically on every export (it should
come back ~1e-14 rad/ms). Both tones sit on their sideband resonances, so all of the time
dependence lives in the amplitudes and phases.

Exported rates are Rabi rates in Hz and phases in degrees, under the convention string
carried in the JSON:

```
H/ħ = 2π·(r_red/2)·e^{−iφ_red}·a·σ₊ + 2π·(r_blue/2)·e^{−iφ_blue}·a†·σ₊ + h.c.
```

### The pipeline

```bash
# analytic protocol → JSON
julia --project=. scripts/export_analytic_jaqalpaw.jl

# every GRAPE .jld2 → *_jaqalpaw.json  (or pass one file as an argument)
julia --project=. scripts/export_jaqalpaw.jl

# inspect the resulting PulseData and the calibration headroom
.venv/bin/python jaqal/spinboson_pulses.py
.venv/bin/python jaqal/spinboson_pulses.py results/data/ion_GRAPE_2spin_controls_jaqalpaw.json

# compile → emulate → compare against the JSON (should PASS)
.venv/bin/python jaqal/verify_waveform.py
.venv/bin/python jaqal/verify_waveform.py \
    --pulse-file results/data/ion_GRAPE_2spin_controls_jaqalpaw.json

# plot the emulated waveform
.venv/bin/jaqalpaw-emulate jaqal/spinboson_analytic.jaqal
```

`verify_waveform.py` is the real end-to-end test: it compiles the gate to RFSoC bytecode,
replays it through JaqalPaw's firmware emulator, and compares the waveform the hardware
would produce against the exported JSON — covering the rate-to-amplitude calibration, the
discrete modulation semantics, phase wrapping and DDS word quantization. Both the analytic
and the two-spin GRAPE drives currently pass with amplitude error ≤ 0.003/100 and phase
error 0.000°.

To regenerate the export figures (`jaqalpaw_tone_map.png`, `jaqalpaw_four_tone.png`,
`jaqalpaw_hardware.png`):

```bash
.venv/bin/python jaqal/verify_waveform.py --dump results/data/emulated_waveform.json
julia --project=. scripts/plot_jaqalpaw_export.jl
```

### The gates

`jaqal/spinboson_pulses.py` defines two gates on top of `QSCOUTBuiltins`. The global beam
runs as a constant square pulse (the lower Raman leg); each individual beam carries the two
modulated sideband tones — tone 0 blue, tone 1 red.

* `SBSqueeze q[i]` — one ion, one channel. The analytic protocol and any single-ion GRAPE
  run. Used by `jaqal/spinboson_analytic.jaqal`.
* `SBSqueeze2 q[i] q[j]` — two ions on two channels under a single global pulse, so the two
  individual beams stay phase coherent for the whole gate. The two-spin GRAPE runs. Used by
  `jaqal/spinboson_grape.jaqal`.

Both take an optional `tone_mask` (`0b01` blue only, `0b10` red only, `0b11` both). A single
sideband is not the protocol, but it is the natural sideband-calibration diagnostic and it
is how `verify_waveform.py` reads the two modulation streams apart.

A Jaqal program can't carry a file path, so the gates pick their drive file from — in order
— the `pulse_file` argument, the `sb_pulse_file` calibration parameter,
`$SPINBOSON_PULSE_FILE`, then the analytic export:

```bash
SPINBOSON_PULSE_FILE=results/data/ion_GRAPE_2spin_controls_jaqalpaw.json \
    .venv/bin/jaqalpaw-emulate -s jaqal/spinboson_grape.jaqal
```

### Two sign conventions to check against the apparatus

Both are inert for a resonant Mølmer–Sørensen gate but **not** for this protocol, which is
sensitive to the relative phase of the two sidebands. They are module constants at the top
of `jaqal/spinboson_pulses.py`:

* `PHASE_SIGN` flips the exported phases as a group. The export defines them through
  `exp(−iφ)`; whether that matches the AOM's sideband sense depends on which Raman leg is
  the upper one.
* `SIDEBAND_SIGN` swaps which tone is red and which is blue, i.e. whether
  `ia_center_frequency + mode` addresses the blue sideband.

The calibration numbers in `SpinBosonPulses` (`sb_lamb_dicke = 0.1`, `sb_mode_index = 0`)
and everything inherited from `QSCOUTBuiltins` are placeholders that the control software
overwrites with calibrated values. With the placeholders, full scale is a 41.7 kHz sideband
rate and the pulses above ask for ≤ 29/100 amplitude.

---

## Things that bit us — don't re-derive them

* A JaqalPaw modulation **list** is N equal-duration steps across the pulse (a **tuple** is
  a spline instead), so sample at bin **midpoints**, not edges — edges land exactly on the
  `pulse_params` segment discontinuities. Minimum step is 4 clock cycles (10 ns) at
  `CLKFREQ = 409.6 MHz`; `gate_SBSqueeze` raises rather than silently violating it.
* The analytic protocol is really a **four-tone** drive: each sideband carries components at
  ±Δ, on the fixed grid `{ω_red ± |Δ|, ω_blue ± |Δ|}`, at constant amplitude |g₀|/2 with only
  per-segment phase steps. That form is exported under `four_tone` — but it is **not**
  realizable as 2 global × 2 individual tones: the beat-note phases would have to satisfy
  `(φ_blue,₊ − φ_blue,₋) = (φ_red,₊ − φ_red,₋)`, and the protocol violates it by exactly π on
  every segment, independent of ϕ. Use the two-tone AM/PM form — that is what the gates emit.
* A QSCOUT individual-addressing channel has only **two tones**, both spoken for by the
  sideband pair. A drive file that also carries carrier controls therefore can't be played as
  written; `report_tone_conflicts` in the exporter flags the overlapping samples and
  `gate_SBSqueeze` refuses such a file.
* JaqalPaw's firmware **emulator** (not its compiler) misattributes updates when both tones
  of a channel step on the same clock cycle — each tone's record comes back holding a merge of
  the two streams. Stagger the grids and both are exact, so the bytecode is right and only the
  emulator's bookkeeping is confused. `verify_waveform.py` works around it by checking one
  sideband per compile via the gate's `tone_mask`.
* The compiler caches imported pulse modules per process, so rewriting a pulse `.py` and
  recompiling in the same process silently reuses the old one. One config per process.

---

## Layout

```
src/SpinBoson_sim.jl              QuantumOptics simulation; pulse_params / protocol_params
scripts/
  ion_test.jl                     self-contained dense-matrix analytic simulation
  SpinBoson_test.jl               two-spin builders (build_spinboson2, make_H_squeeze, …)
  ion_GRAPE_2spin.jl              two-spin GRAPE optimization
  replot_2spin_pulses.jl          re-render pulse figures from saved controls
  export_analytic_jaqalpaw.jl     analytic protocol → JSON
  export_jaqalpaw.jl              GRAPE .jld2 → JSON  (the ε → sideband-tone map)
  plot_jaqalpaw_export.jl         export / emulator figures
  setup_jaqalpaw.sh               clone JaqalPaw + build .venv
jaqal/
  spinboson_pulses.py             SBSqueeze / SBSqueeze2 gate definitions
  spinboson_analytic.jaqal        one-ion circuit
  spinboson_grape.jaqal           two-ion circuit
  verify_waveform.py              compile → emulate → compare
results/data/                     committed GRAPE controls (.jld2); exported .json is regenerated
results/figures/                  generated PNGs (git-ignored)
```

Hilbert space convention in `scripts/ion_test.jl`: the state vector has dimension
`(nmax+1)·(N+1)`, indexed `idx = (idx_b − 1)·dim_s + idx_s` with the boson Fock index
`idx_b ∈ 1:nmax+1` and the Dicke index `idx_s ∈ 1:N+1` running `|−J⟩ … |+J⟩`. This matches
`kron(boson_op, spin_op)`, and anything that builds states or reads out marginals depends on
it. `nmax` is the Fock cutoff — `sinh²(ζJ)` grows quickly with `z_target` and `N`, so bump it
if the `P(n)` panel hits the boundary or `⟨n⟩` saturates.

Generated PNGs, `.txt`, `.log` and `.pptx` files are git-ignored, as are the exported
`results/data/*.json` (regenerate them with the commands above).
