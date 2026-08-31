# Spin-dependent squeezing → QSCOUT pulses

Simulation of the spin-dependent squeezing protocol for trapped ions from
[arXiv:2510.25870](https://arxiv.org/abs/2510.25870) (Bond *et al.*) — Hamiltonian
Eq. (23), pulse sequence Fig. 4(c) — optimized with GRAPE and converted into QSCOUT
hardware waveforms via [JaqalPaw](https://github.com/sandialabs/JaqalPaw).
`ref1.pdf` is the reference paper.

Three steps, one file each:

| | | |
|---|---|---|
| **1** | `spinboson_protocol.jl` | simulate the analytic protocol on \|0⟩\|↓↓⟩, show the Wigner function |
| **2** | `spinboson_grape.jl` | GRAPE, starting from the analytic pulse as the initial guess |
| **3** | `export_jaqalpaw.jl` + `spinboson_pulses.py` + `verify_waveform.py` | convert to sideband tones, compile, and verify against the firmware emulator |

Everything the collaborator needs is committed, including the GRAPE controls
(~40 min each to regenerate), so you can go straight from a clone to verified
hardware waveforms without running an optimization.

> **Run everything from the repo root.** Output paths are relative to it.

---

## Setup

Julia 1.11 and Python 3.12; `Manifest.toml` pins the exact Julia package versions.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # Julia deps
bash setup_jaqalpaw.sh                                # clones JaqalPaw, builds .venv/
```

`setup_jaqalpaw.sh` creates `third_party/JaqalPaw` and `.venv` (both git-ignored) and
installs `jaqalpaw[emulator]` — currently 1.3.0a0 — in editable mode.

---

## 1. The analytic protocol

Hilbert space is `fock ⊗ spin1 ⊗ spin2`, starting from `|0⟩ ⊗ |↓⟩₁ ⊗ |↓⟩₂`, in two stages:

* **Squeeze** (boson ↔ spin 1), stroboscopic:
  `H(t) = g(t)·a·[Jx₁ e^{−iΔ(t)t} + Jy₁ e^{+iΔ(t)t} e^{−iϕ(t)}] + h.c.`
  Four segments per cycle of duration `τ = 2πℓ/|Δ|`; the signs of Δ and g flip on a
  `(+,+), (−,+), (−,−), (+,−)` schedule and the phase alternates between ϕ₁ and ϕ₂.
  Over `t_strobo` this generates the spin-1 conditional squeeze `S(ζ·Jz₁)`.
* **Displacement** (boson ↔ spin 2), free: `H = g₀·P̂·(Jx₂ + Jy₂)`, where `pulse_params`
  holds `(Δ=0, ϕ=0, g=+g₀)`. This displaces the boson conditionally on spin 2.

Starting from `|↓↓⟩` the conditional squeeze is `S(−ζ/2)`, so the squeeze axis already
comes out at the orientation the displacement stage wants and **no π/2 boson rotation is
needed between the stages** (`rotate=true` inserts one for the `|↑↑⟩` case).

```bash
julia --project=. spinboson_protocol.jl     # → results/spinboson_wigner.png
```

```
=== Analytic protocol on |0>|dd> ===
N = 1, nmax = 60, z = 0.500, P = 1, l = 1, dim(H) = 244
g0 = 2pi x 5.000 kHz, |D| = 2pi x 35.449 kHz, tau = 0.0282 ms, zeta = 1.0000
t_strobo = 0.1128 ms, t_free = 0.1128 ms, T = 0.2257 ms

<n>: t=0 -> 0.0000, after squeeze -> 0.2394, final -> 6.5226
     ideal squeeze <n> = sinh^2(zeta/2) = 0.2715  (P=1 undershoot: 11.8%)
norm: 1.000000000001, population in top 2 Fock levels: 2.71e-14  (cutoff ok)
```

The figure shows the pulse sequence over Wigner snapshots at `t = 0`, after the squeeze,
and at `T`: vacuum → squeezed along p → two displaced lobes.

**Two things worth knowing.** `nmax = 60` is not conservative padding — the displacement
drives ⟨n⟩ to 6.5, and at `nmax = 20` nearly 5 % of the population piles into the top
Fock levels and the truncated dynamics are simply wrong. ⟨n⟩ converges from `nmax ≈ 45`.
The run prints that check every time. And the final Wigner has **no interference
fringes**, correctly: tracing out spin 2 leaves a classical mixture of the two
conditional displacements, not a cat. The fringes appear only if you project spin 2.

## 2. GRAPE

Eight bilinear controls, four per spin, in the quadrature/spin-axis basis:

```
H(t) = Σ_{s=1,2} [ ε1ˢ(t)·X̂Jxˢ + ε2ˢ(t)·P̂Jxˢ + ε3ˢ(t)·X̂Jyˢ + ε4ˢ(t)·P̂Jyˢ ],
X̂ = a + a†,   P̂ = i(a† − a)
```

The analytic protocol *is* a point in this control space — using
`c·a + c*·a† = Re(c)·X̂ + Im(c)·P̂`, stage 1 becomes `(ε1,ε2) = g·(cos θ, sin θ)` and
`(ε3,ε4) = g·(cos(θ−ϕ), −sin(θ−ϕ))` with `θ = Δ(t)·t`, and stage 2 becomes
`(ε5..ε8) = (0, g₀, 0, g₀)`. That is the initial guess, so GRAPE starts on the analytic
solution and refines it. The target is exactly the state step 1 produces, so the two
steps cannot drift apart.

The convergence threshold is **F > 0.99** throughout.

```bash
julia --project=. spinboson_grape.jl        # ~40 min; the outputs are committed
```

| | T_frac = 1.0 | T_frac = 0.5 |
|---|---|---|
| duration | 225.7 µs | 112.8 µs |
| F, analytic guess | 0.999903 | 0.415919 |
| F, GRAPE | 0.999903 | 0.992380 |
| F re-propagated at nmax = 60 | 0.999903 | 0.992384 |
| iterations | 1 | 12 |
| max \|GRAPE − guess\| | **0 % of g₀** | **66 % of g₀** |

**At full duration the GRAPE pulse *is* the analytic pulse.** The guess already clears
F > 0.99 at iteration 0, so the optimizer returns it untouched — all eight channels show
0.00 % deviation. That is the correct answer, not a failure: the analytic protocol
already solves the problem at `T_frac = 1`.

**At half duration GRAPE finds a genuinely different pulse.** Spin 1 keeps the
stroboscopic oscillation *shape* but raises the amplitude from 5 to ~6.5 kHz peak, buying
back the area lost to the halved duration; spin 2 raises its drive from 5 to ~7.5 kHz and
turns it on *during* the squeeze stage instead of waiting — GRAPE overlaps the two stages,
which is exactly the freedom the sequential analytic construction gives up.

GRAPE runs at `nmax = 30` for speed (dense `expm` per time step, ~160 s/iteration) and
every result is re-propagated against a freshly built `nmax = 60` target, so a pulse that
quietly exploited the truncation would be caught. Both agree to 6 decimal places.

`plot_pulses` puts all eight panels on **one shared y-scale**, with each panel's deviation
in its title. Per-panel auto-scaling is actively misleading here: channels ε1/ε3 on spin 2
are identically zero in the analytic protocol, so auto-scaling blows a 0.3 % correction up
to fill the frame and the pulse reads as completely different when it is within 1 %.

`replot()` regenerates every figure from the saved controls without re-optimizing.

## 3. Converting to JaqalPaw

### The map

Writing the drive per ion as `H = ε1·X̂Jx + ε2·P̂Jx + ε3·X̂Jy + ε4·P̂Jy` and a resonant
bichromatic Raman drive as `H = A·a·σ₊ + B·a†·σ₊ + h.c.`, matching coefficients gives an
**invertible** linear map:

```
A = (ε1 − ε4)/2 − i(ε2 + ε3)/2      red sideband tone
B = (ε1 + ε4)/2 + i(ε2 − ε3)/2      blue sideband tone
```

So any pulse — analytic or GRAPE — is realizable *exactly* as an amplitude- and
phase-modulated red/blue sideband pair, with no residual. `sideband_tones` checks the
round trip on every export; it comes back at ~3.6e-15 rad/ms. Both tones sit on their
sideband resonances, so the tone frequencies are static and all the time dependence lives
in the amplitudes and phases. Exported rates are Rabi rates in Hz, phases in degrees:

```
H/ħ = 2π·(r_red/2)·e^{−iφ_red}·a·σ₊ + 2π·(r_blue/2)·e^{−iφ_blue}·a†·σ₊ + h.c.
```

### Pipeline

```bash
julia --project=. export_jaqalpaw.jl              # controls → results/*_jaqalpaw.json + figures
.venv/bin/python spinboson_pulses.py             # PulseData summary + calibration headroom
.venv/bin/python verify_waveform.py              # compile → emulate → compare  (PASS)
.venv/bin/python verify_waveform.py \
    --pulse-file results/spinboson_grape_controls_Tfrac50_jaqalpaw.json
```

`verify_waveform.py` is the real end-to-end test: it compiles the gate to RFSoC bytecode,
replays it through JaqalPaw's firmware emulator, and compares the waveform the hardware
would produce against the exported JSON — covering the rate-to-amplitude calibration, the
discrete modulation semantics, phase wrapping and DDS word quantization.

| | T_frac = 1 | T_frac = 0.5 |
|---|---|---|
| peak amplitude | 23.9 / 17.0 of 100 | 32.9 / 30.3 of 100 |
| amplitude error | ≤ 0.003/100 | ≤ 0.003/100 |
| phase error | 0.0000° | 0.0000° |
| bytecode | 1520 words | 2264 words |
| | **PASS** | **PASS** |

### The gate

`spinboson_pulses.py` defines `SBSqueeze2 q[i] q[j]` on top of `QSCOUTBuiltins`: the
global beam runs as a constant square pulse (the lower Raman leg) and each individual beam
carries that ion's two modulated sideband tones — tone 0 blue, tone 1 red. Both individual
pulses sit under the *one* global pulse, which is what keeps them phase coherent with each
other. `spinboson_grape.jaqal` is the circuit.

A Jaqal program can't carry a file path, so the gate picks its drive file from — in order —
the `pulse_file` argument, the `sb_pulse_file` calibration parameter, then
`$SPINBOSON_PULSE_FILE`:

```bash
SPINBOSON_PULSE_FILE=results/spinboson_grape_controls_Tfrac50_jaqalpaw.json \
    .venv/bin/jaqalpaw-emulate -s spinboson_grape.jaqal
```

`tone_mask` (`0b01` blue, `0b10` red, `0b11` both) drives a single sideband — not the
protocol, but the natural sideband-calibration diagnostic, and how `verify_waveform.py`
reads the two modulation streams apart.

### Before trusting a run on hardware

The calibration values in `SpinBosonPulses` (`sb_lamb_dicke = 0.1`, `sb_mode_index = 0`)
and everything inherited from `QSCOUTBuiltins` are **placeholders** that the control
software overwrites with calibrated ones. With the placeholders, full scale is a 41.7 kHz
sideband rate and the pulses above ask for ≤ 33/100.

Two sign conventions need checking against the apparatus. Both are inert for a resonant
Mølmer–Sørensen gate but **not** for this protocol, which is sensitive to the relative
phase of the two sidebands:

* `PHASE_SIGN` flips the exported phases as a group. The export defines them through
  `exp(−iφ)`; whether that matches the AOM's sideband sense depends on which Raman leg is
  the upper one.
* `SIDEBAND_SIGN` swaps which tone is red and which is blue, i.e. whether
  `ia_center_frequency + mode` addresses the blue sideband.

---

## Things that bit us — don't re-derive them

* A JaqalPaw modulation **list** is N equal-duration steps across the pulse (a **tuple** is
  a spline instead), so sample at bin **midpoints**, not edges — edges land exactly on the
  `pulse_params` segment discontinuities. Minimum step is 4 clock cycles (10 ns) at
  `CLKFREQ = 409.6 MHz`; the exporter and the gate both raise rather than violate it.
* The analytic protocol is really a **four-tone** drive: each sideband carries components
  at ±Δ on the fixed grid `{ω_red ± |Δ|, ω_blue ± |Δ|}`. It is **not** realizable as
  2 global × 2 individual tones — the beat-note phases would have to satisfy
  `(φ_blue,₊ − φ_blue,₋) = (φ_red,₊ − φ_red,₋)`, and the protocol violates it by exactly π
  on every segment, independent of ϕ. The two-tone AM/PM form used here has no such
  problem, because the amplitude and phase modulation absorbs the ±Δ structure.
* A QSCOUT individual-addressing channel has only **two tones**, both spoken for by the
  sideband pair. A pulse that also wants a carrier term cannot be played as written; it
  needs a third tone, or a GRAPE re-run with the carrier confined to sideband-idle windows.
* JaqalPaw's firmware **emulator** (not its compiler) misattributes updates when both tones
  of a channel step on the same clock cycle — each tone's record comes back holding a merge
  of the two streams. Stagger the grids and both are exact, so the bytecode is right and
  only the emulator's bookkeeping is confused. `verify_waveform.py` works around it by
  checking one sideband per compile via `tone_mask`.
* The compiler caches imported pulse modules per process, so rewriting a pulse `.py` and
  recompiling in the same process silently reuses the old one. One config per process.

---

## Layout

```
spinboson_protocol.jl      step 1   simulation + Wigner
spinboson_grape.jl         step 2   GRAPE from the analytic guess
export_jaqalpaw.jl         step 3   eps -> sideband tones -> JSON + figures
spinboson_pulses.py        step 3   SBSqueeze2 gate definition
spinboson_grape.jaqal      step 3   two-ion circuit
verify_waveform.py         step 3   compile -> emulate -> compare
setup_jaqalpaw.sh                   clone JaqalPaw + build .venv
results/
  spinboson_grape_controls_Tfrac{100,50}.jld2    committed GRAPE controls
  *_jaqalpaw.json                                regenerated by export_jaqalpaw.jl
  *.png                                          figures from all three steps
```

Figure style follows the rest of the project: Julia `Plots`, `fontfamily="Computer
Modern"`, `dpi=200`. That font has no subscript, superscript or arrow glyphs — `ε₁`, `σ₊`,
`↓` render as ☒ — so plot text is written `e1`, `sigma+`, `dd`.
