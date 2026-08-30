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
"""

import json
import os

import numpy as np

from jaqalpaw.ir.pulse_data import PulseData
from jaqalpaw.utilities.helper_functions import discretize_frequency
from qscout.v1.QSCOUTBuiltins import QSCOUTBuiltins, GLOBAL_BEAM

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PULSE_FILE = os.path.join(
    HERE, os.pardir, "results", "data", "analytic_jaqalpaw.json"
)

PHASE_SIGN = +1.0
SIDEBAND_SIGN = +1.0

# JaqalPaw refuses modulation steps shorter than 4 clock cycles (~10 ns).
MIN_STEP_S = 10e-9


def load_drive(path=None):
    """Read an exported drive description. Returns the raw dictionary."""
    with open(path or DEFAULT_PULSE_FILE) as f:
        return json.load(f)


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
    ## Exported drive file (absolute path).
    sb_pulse_file: str = DEFAULT_PULSE_FILE

    # ===== calibration-derived quantities =====

    @property
    def sb_mode_frequency(self):
        """Frequency of the motional mode being squeezed."""
        return self.lower_motional_mode_frequencies[self.sb_mode_index]

    @property
    def sb_carrier_rate_full_scale(self):
        """Carrier Rabi rate (Hz) at individual-beam amplitude 100, with the
        global beam at its calibrated counter-propagating amplitude."""
        rate_at_cal = 0.5 / self.counter_resonant_pi_time
        return rate_at_cal * 100.0 / self.amp1_counterprop_list[self.target0]

    @property
    def sb_rate_full_scale(self):
        """Sideband Rabi rate (Hz) at individual-beam amplitude 100."""
        return self.sb_lamb_dicke * self.sb_carrier_rate_full_scale

    def sb_rate_to_amp(self, rates):
        """Sideband Rabi rates (Hz) -> individual-beam amplitudes (0-100).

        Assumes the two-photon Rabi rate is linear in the individual-beam
        amplitude, which is how the builtins' amplitude lists are calibrated.
        Raises if the pulse asks for more than full scale, since silently
        clipping would change the gate rather than degrade it.
        """
        amps = np.asarray(rates, dtype=float) * 100.0 / self.sb_rate_full_scale
        peak = float(amps.max(initial=0.0))
        if peak > 100.0:
            raise ValueError(
                f"Pulse needs amplitude {peak:.1f} (>100): peak sideband rate "
                f"{float(np.max(rates)):.0f} Hz exceeds full scale "
                f"{self.sb_rate_full_scale:.0f} Hz. Lengthen the pulse, or "
                f"recalibrate sb_lamb_dicke / counter_resonant_pi_time."
            )
        return amps

    # ===== gates =====

    def gate_SBSqueeze(self, channel, tone_mask=0b11, pulse_file=None):
        """Play one exported spin-dependent squeezing pulse on `channel`.

        The global beam runs as a constant square pulse (the lower Raman leg);
        the individual beam carries the two modulated sideband tones.

        `tone_mask` selects which sidebands are driven: 0b01 keeps only the blue
        tone (tone 0), 0b10 only the red one (tone 1), 0b11 both. Driving a
        single sideband is not the protocol, but it is the natural diagnostic
        for checking the sideband calibration and is what
        `verify_waveform.py` uses to read the two modulation streams apart.
        """
        drive = load_drive(pulse_file or self.sb_pulse_file)
        dur = drive["duration_s"]

        step = dur / drive["n_samples"]
        if step < MIN_STEP_S:
            raise ValueError(
                f"{drive['n_samples']} samples over {dur * 1e6:.1f} us is "
                f"{step * 1e9:.2f} ns per step, below the {MIN_STEP_S * 1e9:.0f} ns "
                "floor. Re-export with fewer samples."
            )

        ion = drive["ions"][0]
        if tone_mask & 0b01:
            blue_amp = list(self.sb_rate_to_amp(ion["blue_rate_hz"]))
            blue_phase = list(PHASE_SIGN * np.asarray(ion["blue_phase_deg"]))
        else:
            blue_amp, blue_phase = 0, 0
        if tone_mask & 0b10:
            red_amp = list(self.sb_rate_to_amp(ion["red_rate_hz"]))
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
        return [
            PulseData(
                GLOBAL_BEAM,
                dur,
                freq0=self.global_center_frequency,
                amp0=self.amp0_counterprop,
                phase0=0,
                sync_mask=0b01,
                fb_enable_mask=0b01,
            ),
            PulseData(
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
            ),
        ]


class jaqal_pulses:
    GatePulses = SpinBosonPulses


# ===== SELF-TEST =====

if __name__ == "__main__":
    gp = SpinBosonPulses()
    drive = load_drive()
    ion = drive["ions"][0]

    print("=== exported drive ===")
    print(f"source        : {drive['source']}")
    print(f"duration      : {drive['duration_s'] * 1e6:.3f} us")
    print(f"samples       : {drive['n_samples']}  "
          f"({drive['sample_dt_s'] * 1e9:.1f} ns per step)")
    print(f"|Delta|       : {drive['detuning_hz'] / 1e3:.3f} kHz")
    print(f"g0            : {drive['g0_hz'] / 1e3:.3f} kHz")

    print("\n=== calibration ===")
    print(f"mode frequency: {gp.sb_mode_frequency / 1e6:.4f} MHz")
    print(f"carrier @ amp 100 : {gp.sb_carrier_rate_full_scale / 1e3:.1f} kHz")
    print(f"sideband @ amp 100: {gp.sb_rate_full_scale / 1e3:.1f} kHz "
          f"(eta = {gp.sb_lamb_dicke})")

    peak = max(max(ion["blue_rate_hz"]), max(ion["red_rate_hz"]))
    amps = gp.sb_rate_to_amp(ion["blue_rate_hz"] + ion["red_rate_hz"])
    print(f"peak sideband rate: {peak / 1e3:.3f} kHz "
          f"-> amplitude {amps.max():.1f}/100")

    print("\n=== PulseData ===")
    for pd in gp.gate_SBSqueeze(gp.target0):
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
