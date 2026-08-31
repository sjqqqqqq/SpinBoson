"""End-to-end check of the exported two-spin GRAPE pulse.

Compiles the gate to RFSoC bytecode, replays it through JaqalPaw's firmware
emulator, and compares the waveform the hardware would produce against the
amplitudes and phases in the exported JSON. That covers everything between the
Julia exporter and the board: the rate-to-amplitude calibration, the discrete
modulation semantics, phase wrapping, and DDS word quantization.

One wrinkle. JaqalPaw's *emulator* (not its compiler) misattributes updates when
both tones of a channel step on the same clock cycle: each tone's record comes
back holding a merge of the two streams. Stagger the two grids and both come
back exact, so the bytecode is right and only the emulator's bookkeeping is
confused. Since the real pulse steps both tones together, this script checks one
sideband per compile via the gate's `tone_mask`, then confirms the full
four-tone gate compiles.

    .venv/bin/python verify_waveform.py
    .venv/bin/python verify_waveform.py \\
        --pulse-file results/spinboson_grape_controls_Tfrac50_jaqalpaw.json
"""

import asyncio
import contextlib
import json
import os
import sys
import tempfile

import numpy as np

from jaqalpaw.compiler.jaqal_compiler import CircuitCompiler
from jaqalpaw.emulator.firmware_emulator import chunk_data_direct, firmware_emulator
from jaqalpaw.utilities.parameters import CLKFREQ

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spinboson_pulses import (  # noqa: E402
    PULSE_FILE_ENV,
    SpinBosonPulses,
    load_drive,
    resolve_pulse_file,
)

JAQAL_FILE = os.path.join(HERE, "spinboson_grape.jaqal")

# Indices into the emulator's per-channel record (byte_decoding.mod_type_dict).
MOD = {"f0": 0, "a0": 1, "p0": 2, "f1": 3, "a1": 4, "p1": 5}

CIRCUIT = """from .spinboson_pulses usepulses *
let ion1 {channel_a}
let ion2 {channel_b}
let tones {tone_mask}
register q[8]
prepare_all
SBSqueeze2 q[ion1] q[ion2] tones
measure_all
"""


def compile_bytecode(jaqal_file):
    """Compile a Jaqal program to a flat byte string."""
    cc = CircuitCompiler(file=jaqal_file)
    cc.compile()
    return b"".join(w for ch in cc.bytecode(0xFF) for words in ch for w in words)


@contextlib.contextmanager
def temp_circuit(code):
    """Write a Jaqal program next to the pulse module and yield its path.

    `usepulses` resolves `.spinboson_pulses` relative to the Jaqal file, so the
    scratch file has to live in this directory rather than a temp dir.
    """
    fd, path = tempfile.mkstemp(suffix=".jaqal", prefix="_verify_", dir=HERE)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        yield path
    finally:
        os.unlink(path)


def emulate(code_bytes, num_channels=8):
    """Replay bytecode through the firmware emulator; return its per-channel record."""
    record = {
        c: {
            d: {
                "time": [0], "data": [0], "waittrig": [0], "enablemask": [0],
                "fwd_frame0_mask": [0], "inv_frame0_mask": [0],
                "fwd_frame1_mask": [0], "inv_frame1_mask": [0],
            }
            for d in range(8)
        }
        for c in range(8)
    }
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        firmware_emulator(
            chunk_data_direct(code_bytes, chunksize=1),
            num_channels=num_channels,
            master_data_record=record,
        )
    )
    loop.close()
    return record


def find_gate_updates(rec, n_expected, step_cycles):
    """Pick the gate's own updates out of the full emulator timeline.

    The program also contains prepare_all and measure_all, so the record holds a
    handful of extra events on either side. The gate is the only stretch that
    updates on a regular grid of `step_cycles`, so take the longest run with
    that spacing.
    """
    t = np.asarray(rec["time"], dtype=float)
    v = np.asarray(rec["data"], dtype=float)
    if len(t) < n_expected:
        raise RuntimeError(
            f"Only {len(t)} updates on this parameter; expected >= {n_expected}. "
            "The gate did not compile into the waveform."
        )

    # Rounding to clock cycles spreads each step over floor/ceil of the ideal.
    uniform = np.abs(np.diff(t) - step_cycles) <= 1.5
    best_len = best_end = run = 0
    for i, u in enumerate(uniform):
        run = run + 1 if u else 0
        if run > best_len:
            best_len, best_end = run, i + 1
    if best_len + 1 < n_expected:
        raise RuntimeError(
            f"Longest uniform run is {best_len + 1} updates, expected {n_expected}."
        )
    sl = slice(best_end - best_len, best_end - best_len + n_expected)
    return t[sl], v[sl]


def circular_error_deg(a, b):
    """Largest angular difference (degrees), accounting for wrapping."""
    d = (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
    return float(np.max(np.abs(d)))


def emulate_gate(channels, tone_mask):
    """Compile and emulate the gate with only `tone_mask` driven."""
    code = CIRCUIT.format(channel_a=channels[0], channel_b=channels[1],
                          tone_mask=tone_mask)
    with temp_circuit(code) as path:
        return emulate(compile_bytecode(path))


def check_tone(gp, drive, channels, ion_index, tone, record,
               amp_tol=0.05, phase_tol=0.5, dump=None):
    """Compare one ion's emulated sideband against the exported one."""
    ion = drive["ions"][ion_index]
    channel = channels[ion_index]
    n = drive["n_samples"]
    step_cycles = drive["duration_s"] * CLKFREQ / n
    side = "blue" if tone == 0 else "red"
    ch = record[channel]

    t_upd, amp = find_gate_updates(ch[MOD[f"a{tone}"]], n, step_cycles)
    _, phase = find_gate_updates(ch[MOD[f"p{tone}"]], n, step_cycles)

    want_amp = np.asarray(gp.sb_rate_to_amp(ion[f"{side}_rate_hz"], channel))
    want_phase = np.asarray(ion[f"{side}_phase_deg"])

    amp_err = float(np.max(np.abs(amp - want_amp)))
    phase_err = circular_error_deg(phase, want_phase)
    ok = amp_err <= amp_tol and phase_err <= phase_tol

    freqs = sorted({f for f in ch[MOD[f"f{tone}"]]["data"] if f})
    if dump is not None:
        dump[f"ion{ion_index + 1}_{side}"] = {
            "t_s": ((t_upd - t_upd[0]) / CLKFREQ).tolist(),
            "amp_emulated": amp.tolist(), "amp_intended": want_amp.tolist(),
            "phase_emulated": phase.tolist(),
            "phase_intended": ((want_phase + 180.0) % 360.0 - 180.0).tolist(),
            "freq_hz": freqs, "amp_err": amp_err, "phase_err": phase_err,
            "channel": channel,
        }
    print(f"  ion {ion_index + 1} ch {channel} tone {tone} ({side:4s}): "
          f"amp err {amp_err:7.4f} (tol {amp_tol}), "
          f"phase err {phase_err:7.4f} deg (tol {phase_tol})  "
          f"{'ok' if ok else 'FAIL'}")
    print(f"      freq {[f / 1e6 for f in freqs]} MHz, "
          f"amp range {want_amp.min():.2f}-{want_amp.max():.2f}/100")
    return ok


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dump_path = argv[argv.index("--dump") + 1] if "--dump" in argv else None
    pulse_file = (argv[argv.index("--pulse-file") + 1]
                  if "--pulse-file" in argv else None)

    # The compiled gate resolves its drive file independently of this process's
    # objects, so hand it over through the environment before compiling.
    pulse_file = os.path.abspath(resolve_pulse_file(pulse_file))
    os.environ[PULSE_FILE_ENV] = pulse_file

    gp = SpinBosonPulses()
    drive = load_drive(pulse_file)
    n = drive["n_samples"]
    if len(drive["ions"]) != 2:
        print(f"{drive['source']} drives {len(drive['ions'])} ion(s); expected 2.")
        return 1
    channels = [gp.target0, gp.target1]

    print("=== exported drive ===")
    print(f"  {drive['source']}")
    print(f"  {drive['duration_s'] * 1e6:.3f} us, {n} samples "
          f"({drive['duration_s'] * CLKFREQ / n:.2f} clock cycles per step)")
    print(f"  2 ions on channels {channels}")
    if "F" in drive:
        print(f"  GRAPE F = {drive['F']:.6f}  (T_frac = {drive.get('T_frac')})")

    print("\n=== per-sideband waveform check ===")
    dump = {} if dump_path else None
    ok = True
    for tone in (0, 1):
        # One compile per tone; both channels come back in the same record, and
        # only the two tones of ONE channel confuse the emulator.
        record = emulate_gate(channels, 1 << tone)
        for ion_index in (0, 1):
            ok &= check_tone(gp, drive, channels, ion_index, tone, record,
                             dump=dump)

    print("\n=== full four-tone gate ===")
    code = compile_bytecode(JAQAL_FILE)
    n_words = len(code) // 32
    print(f"  {os.path.basename(JAQAL_FILE)} compiles: "
          f"{len(code)} bytes ({n_words} 256-bit words)")

    if dump_path:
        dump["bytecode_words"] = n_words
        dump["duration_s"] = drive["duration_s"]
        dump["source"] = drive["source"]
        with open(dump_path, "w") as f:
            json.dump(dump, f)
        print(f"  emulated traces written to {dump_path}")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
