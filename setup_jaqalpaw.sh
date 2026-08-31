#!/usr/bin/env bash
# Fetch JaqalPaw and install it into a project-local virtualenv.
#
# Both third_party/ and .venv/ are git-ignored, so this is how you recreate
# them on a fresh clone. Run from the repo root:
#
#   bash setup_jaqalpaw.sh
#
# Afterwards:
#   julia --project=. export_jaqalpaw.jl        # controls -> drive JSON
#   .venv/bin/python spinboson_pulses.py        # inspect the PulseData
#   .venv/bin/python verify_waveform.py         # compile + emulate + compare
#   .venv/bin/jaqalpaw-emulate spinboson_grape.jaqal   # plot the waveform

set -euo pipefail

REPO="https://github.com/sandialabs/JaqalPaw"
DEST="third_party/JaqalPaw"

if [ ! -d "$DEST" ]; then
    echo "==> cloning $REPO into $DEST"
    git clone --depth 1 "$REPO" "$DEST"
else
    echo "==> $DEST already present, skipping clone"
fi

if [ ! -d .venv ]; then
    echo "==> creating .venv"
    python3 -m venv .venv
fi

echo "==> installing JaqalPaw (editable) and its dependencies"
.venv/bin/pip install --quiet --upgrade pip
# matplotlib comes from the [emulator] extra; jaqalpaq is pulled in as a dependency.
.venv/bin/pip install --quiet -e "${DEST}[emulator]"

echo "==> installed:"
.venv/bin/python -c "
import importlib.metadata as md
for p in ('jaqalpaw', 'jaqalpaq', 'numpy', 'scipy'):
    print(f'  {p} {md.version(p)}')
"
