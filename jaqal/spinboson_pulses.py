"""JaqalPaw gate pulses for the spin-dependent squeezing protocol.

Reads a JSON drive description written by `scripts/export_analytic_jaqalpaw.jl`
(analytic stroboscopic protocol) or `scripts/export_jaqalpaw.jl` (GRAPE
controls) and turns it into QSCOUT `PulseData`.

The JSON gives, per ion and per time sample, the red and blue sideband Rabi
rates and phases defined by

    H/hbar = 2*pi*(r_red /2) * exp(-i*phi_red ) * a    * sigma_plus
           + 2*pi*(r_blue/2) * exp(-i*phi_blue) * adag * sigma_plus + h.c.

with both tones sitting exactly on their sideband resonances, so all of the
time dependence lives in the amplitudes and phases. That maps one-to-one onto a
single individual-addressing channel: tone 0 drives the blue sideband, tone 1
the red one, both amplitude- and phase-modulated.

A drive file holds one entry per driven ion. `gate_SBSqueeze` plays one of them
on one channel (the analytic protocol, and any single-ion GRAPE run);
`gate_SBSqueeze2` plays two of them on two channels under a single global pulse,
which is what the two-spin GRAPE runs need.

Which file gets played is, in order of precedence: the `pulse_file` argument,
the `sb_pulse_file` calibration parameter, the `SPINBOSON_PULSE_FILE`
environment variable, then the analytic export. The environment variable is the
practical one, because a Jaqal program has no way to pass a path:

    SPINBOSON_PULSE_FILE=results/data/ion_GRAPE_2spin_controls_jaqalpaw.json \
        .venv/bin/jaqalpaw-emulate jaqal/spinboson_grape.jaqal

Sign conventions worth checking against the apparatus before trusting a run:

  * `PHASE_SIGN` flips the exported phases as a group. The export defines them
    through exp(-i*phi); whether that matches the AOM's sideband sense depends
    on which Raman leg is the upper one.
  * `SIDEBAND_SIGN` swaps which tone is red and which is blue, i.e. whether
    `ia_center_frequency + mode` addresses the blue sideband.

Both are inert for a resonant Molmer-Sorensen gate but not for this protocol,
which is sensitive to the relative phase of the two sidebands.

Run this file directly for a self-test:

    .venv/bin/python jaqal/spinboson_pulses.py
    .venv/bin/python jaqal/spinboson_pulses.py results/data/<other>_jaqalpaw.json
"""

import json
import os

import numpy as np

from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.utilities.helper_functions import discretize_frequency
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins, GLOBAL_BEAM

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYTIC_PULSE_FILE = os.path.join(
    HERE, os.pardir, "results", "data", "analytic_jaqalpaw.json"
)
## Environment override, read at gate-call time so one process can compile
## against a file chosen after this module was imported.
PULSE_FILE_ENV = "SPINBOSON_PULSE_FILE"

PHASE_SIGN = +1.0
SIDEBAND_SIGN = +1.0

# JaqalPaw refuses modulation steps shorter than 4 clock cycles (~10 ns).
MIN_STEP_S = 10e-9


def resolve_pulse_file(path=None):
    """Pick the drive file: explicit path, then $SPINBOSON_PULSE_FILE, then the
    analytic export."""
    return path or os.environ.get(PULSE_FILE_ENV) or ANALYTIC_PULSE_FILE


def load_drive(path=None):
    """Read an exported drive description.

    Returns the raw dictionary, with `source` filled in for files written by
    `export_jaqalpaw.jl`, which names the field `source_file` instead.
    """
    resolved = resolve_pulse_file(path)
    with open(resolved) as f:
        drive = json.load(f)
    drive.setdefault("source", drive.get("source_file", os.path.basename(resolved)))
    return drive


class SpinBosonPulses(QSCOUTBuiltins):
    """Spin-dependent squeezing pulses on top of the QSCOUT builtins.

    The inherited `CalibrationParameters` supply the frequencies and the
    amplitude scale; the three fields below are the extra knobs this protocol
    needs. As with the builtins, the values here are placeholders that the
    control software overwrites with calibrated ones.
    """

    ## Lamb-Dicke parameter for the addressed mode: sideband Rabi rate is
    ## sb_lamb_dicke times the carrier Rabi rate at the same beam amplitude.
    sb_lamb_dicke: float = 0.1
    ## Index into lower_motional_mode_frequencies for the squeezed mode.
    sb_mode_index: int = 0
    ## Exported drive file. Empty means "resolve at call time" — see
    ## `resolve_pulse_file`.
    sb_pulse_file: str = ""

    # ===== calibration-derived quantities =====

    @property
    def sb_mode_frequency(self):
        """Frequency of the motional mode being squeezed."""
        return self.lower_motional_mode_frequencies[self.sb_mode_index]

    def sb_carrier_rate_full_scale_on(self, channel):
        """Carrier Rabi rate (Hz) on `channel` at individual-beam amplitude 100,
        with the global beam at its calibrated counter-propagating amplitude."""
        rate_at_cal = 0.5 / self.counter_resonant_pi_time
        return rate_at_cal * 100.0 / self.amp1_counterprop_list[channel]

    def sb_rate_full_scale_on(self, channel):
        """Sideband Rabi rate (Hz) on `channel` at individual-beam amplitude 100."""
        return self.sb_lamb_dicke * self.sb_carrier_rate_full_scale_on(channel)

    @property
    def sb_carrier_rate_full_scale(self):
        """Carrier full-scale rate on the first addressed channel."""
        return self.sb_carrier_rate_full_scale_on(self.target0)

    @property
    def sb_rate_full_scale(self):
        """Sideband full-scale rate on the first addressed channel."""
        return self.sb_rate_full_scale_on(self.target0)

    def sb_rate_to_amp(self, rates, channel=None):
        """Sideband Rabi rates (Hz) -> individual-beam amplitudes (0-100).

        Each channel has its own calibrated counter-propagating amplitude, so
        the conversion is per-channel; `channel=None` uses the first addressed
        one. Assumes the two-photon Rabi rate is linear in the individual-beam
        amplitude, which is how the builtins' amplitude lists are calibrated.
        Raises if the pulse asks for more than full scale, since silently
        clipping would change the gate rather than degrade it.
        """
        channel = self.target0 if channel is None else channel
        full_scale = self.sb_rate_full_scale_on(channel)
        amps = np.asarray(rates, dtype=float) * 100.0 / full_scale
        peak = float(amps.max(initial=0.0))
        if peak > 100.0:
            raise ValueError(
                f"Pulse needs amplitude {peak:.1f} (>100) on channel {channel}: "
                f"peak sideband rate {float(np.max(rates)):.0f} Hz exceeds full "
                f"scale {full_scale:.0f} Hz. Lengthen the pulse, or recalibrate "
                "sb_lamb_dicke / counter_resonant_pi_time."
            )
        return amps

    # ===== pulse construction =====

    def _load_checked(self, pulse_file, n_ions_needed):
        """Load a drive file and check it can be played as written."""
        drive = load_drive(pulse_file or self.sb_pulse_file or None)
        dur = drive["duration_s"]

        step = dur / drive["n_samples"]
        if step < MIN_STEP_S:
            raise ValueError(
                f"{drive['n_samples']} samples over {dur * 1e6:.1f} us is "
                f"{step * 1e9:.2f} ns per step, below the {MIN_STEP_S * 1e9:.0f} ns "
                "floor. Re-export with fewer samples."
            )

        n_ions = len(drive["ions"])
        if n_ions < n_ions_needed:
            raise ValueError(
                f"{drive['source']} describes {n_ions} driven ion(s); this gate "
                f"needs {n_ions_needed}. Use gate_SBSqueeze for a single-ion "
                "drive and gate_SBSqueeze2 for a two-ion one."
            )
        if drive.get("carrier") is not None:
            raise ValueError(
                f"{drive['source']} carries carrier controls as well as "
                "sidebands. An individual-addressing channel has only two "
                "tones, both spoken for by the sideband pair, so this needs a "
                "third tone or a GRAPE re-run with the carrier confined to the "
                "sideband-idle windows (see report_tone_conflicts in "
                "scripts/export_jaqalpaw.jl)."
            )
        return drive, dur

    def _global_pulse(self, dur):
        """The constant square global pulse (the lower Raman leg)."""
        return PulseData(
            GLOBAL_BEAM,
            dur,
            freq0=self.global_center_frequency,
            amp0=self.amp0_counterprop,
            phase0=0,
            sync_mask=0b01,
            fb_enable_mask=0b01,
        )

    def _ia_pulse(self, channel, ion, dur, tone_mask):
        """One ion's modulated sideband pair on one individual-addressing channel."""
        if tone_mask & 0b01:
            blue_amp = list(self.sb_rate_to_amp(ion["blue_rate_hz"], channel))
            blue_phase = list(PHASE_SIGN * np.asarray(ion["blue_phase_deg"]))
        else:
            blue_amp, blue_phase = 0, 0
        if tone_mask & 0b10:
            red_amp = list(self.sb_rate_to_amp(ion["red_rate_hz"], channel))
            red_phase = list(PHASE_SIGN * np.asarray(ion["red_phase_deg"]))
        else:
            red_amp, red_phase = 0, 0

        mode = SIDEBAND_SIGN * self.sb_mode_frequency
        freq_blue = discretize_frequency(self.ia_center_frequency) + \
            discretize_frequency(mode)
        freq_red = discretize_frequency(self.ia_center_frequency) - \
            discretize_frequency(mode)

        # Lists (not tuples) so JaqalPaw treats them as discrete steps, matching
        # the piecewise-constant sampling the exporter used.
        return PulseData(
            channel,
            dur,
            freq0=freq_blue,
            freq1=freq_red,
            amp0=blue_amp,
            amp1=red_amp,
            phase0=blue_phase,
            phase1=red_phase,
            sync_mask=0b11,
            fb_enable_mask=0b00,
        )

    # ===== gates =====

    def gate_SBSqueeze(self, channel, tone_mask=0b11, pulse_file=None):
        """Play a single-ion spin-dependent squeezing pulse on `channel`.

        This is the analytic stroboscopic protocol and the single-ion GRAPE
        runs. The global beam runs as a constant square pulse (the lower Raman
        leg); the individual beam carries the two modulated sideband tones.

        `tone_mask` selects which sidebands are driven: 0b01 keeps only the blue
        tone (tone 0), 0b10 only the red one (tone 1), 0b11 both. Driving a
        single sideband is not the protocol, but it is the natural diagnostic
        for checking the sideband calibration and is what
        `verify_waveform.py` uses to read the two modulation streams apart.
        """
        drive, dur = self._load_checked(pulse_file, 1)
        return [
            self._global_pulse(dur),
            self._ia_pulse(channel, drive["ions"][0], dur, tone_mask),
        ]

    def gate_SBSqueeze2(self, channel_a, channel_b, tone_mask=0b11,
                        pulse_file=None):
        """Play a two-ion spin-dependent squeezing pulse (the two-spin GRAPE runs).

        Ion 1 of the drive file goes on `channel_a`, ion 2 on `channel_b`, both
        under the same global pulse, so the two individual beams stay phase
        coherent with each other for the whole gate. Same `tone_mask` semantics
        as `gate_SBSqueeze`, applied to both channels.
        """
        drive, dur = self._load_checked(pulse_file, 2)
        return [
            self._global_pulse(dur),
            self._ia_pulse(channel_a, drive["ions"][0], dur, tone_mask),
            self._ia_pulse(channel_b, drive["ions"][1], dur, tone_mask),
        ]


class jaqal_pulses:
    GatePulses = SpinBosonPulses


# ===== SELF-TEST =====

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    gp = SpinBosonPulses()
    drive = load_drive(path)
    ions = drive["ions"]
    channels = [gp.target0, gp.target1][: len(ions)]

    print("=== exported drive ===")
    print(f"source        : {drive['source']}")
    print(f"driven ions   : {len(ions)}")
    print(f"duration      : {drive['duration_s'] * 1e6:.3f} us")
    print(f"samples       : {drive['n_samples']}  "
          f"({drive['sample_dt_s'] * 1e9:.1f} ns per step)")
    if "detuning_hz" in drive:
        print(f"|Delta|       : {drive['detuning_hz'] / 1e3:.3f} kHz")
        print(f"g0            : {drive['g0_hz'] / 1e3:.3f} kHz")
    if "F" in drive:
        print(f"GRAPE fidelity: {drive['F']:.6f}")

    print("\n=== calibration ===")
    print(f"mode frequency: {gp.sb_mode_frequency / 1e6:.4f} MHz")
    for ch in channels:
        print(f"channel {ch}: carrier @ amp 100 "
              f"{gp.sb_carrier_rate_full_scale_on(ch) / 1e3:.1f} kHz, "
              f"sideband @ amp 100 {gp.sb_rate_full_scale_on(ch) / 1e3:.1f} kHz "
              f"(eta = {gp.sb_lamb_dicke})")

    for ion, ch in zip(ions, channels):
        peak = max(max(ion["blue_rate_hz"]), max(ion["red_rate_hz"]))
        amps = gp.sb_rate_to_amp(ion["blue_rate_hz"] + ion["red_rate_hz"], ch)
        print(f"ion {ion['index']} on channel {ch}: peak sideband rate "
              f"{peak / 1e3:.3f} kHz -> amplitude {amps.max():.1f}/100")

    print("\n=== PulseData ===")
    pulses = (gp.gate_SBSqueeze2(*channels, pulse_file=path) if len(ions) >= 2
              else gp.gate_SBSqueeze(channels[0], pulse_file=path))
    for pd in pulses:
        print(f"  channel {pd.channel}: {pd.dur} clock cycles "
              f"({pd.real_dur * 1e6:.3f} us)")
        for tone in (0, 1):
            f = getattr(pd, f"freq{tone}")
            a = getattr(pd, f"amp{tone}")
            p = getattr(pd, f"phase{tone}")
            n_a = len(a) if hasattr(a, "__len__") else 1
            n_p = len(p) if hasattr(p, "__len__") else 1
            f_txt = f"{f / 1e6:.6f} MHz" if not hasattr(f, "__len__") else "modulated"
            print(f"    tone {tone}: freq {f_txt}, "
                  f"amp {n_a} pt(s), phase {n_p} pt(s)")

    ft = drive.get("four_tone")
    if ft is not None:
        print("\n=== four-tone form (not used; see module docstring) ===")
        print(f"  {len(ft['segments'])} segments, all tones at "
              f"{ft['tone_rate_hz'] / 2e3:.3f} kHz")
        print(f"  realizable as 2 global x 2 individual tones: {ft['factorizable']} "
              f"(phase residual {ft['factorization_residual_deg'][0]:+.1f} deg)")
