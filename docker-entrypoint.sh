#!/bin/sh
set -e

# mediapipe imports sounddevice, which calls Pa_Initialize() at import time.
# PortAudio's PulseAudio host API fails hard when no server is reachable, which
# would make `import mediapipe` raise before any video is processed. Start a
# private server for this container if one is not already running.
if ! pactl info >/dev/null 2>&1; then
    pulseaudio --start --exit-idle-time=-1 >/dev/null 2>&1 || true
fi

exec "$@"
