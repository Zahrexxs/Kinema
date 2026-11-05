# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KINEMA_UPLOAD_DIR=/data/uploads \
    KINEMA_OUTPUT_DIR=/data/outputs

# mediapipe imports sounddevice, which initialises PortAudio on import, and
# PortAudio aborts if its PulseAudio host API cannot reach a server. OpenCV also
# links GL/EGL. These are the same packages CI installs for the test job.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
        libportaudio2 \
        pulseaudio \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 kinema \
    && mkdir -p /data/uploads /data/outputs /tmp/runtime-kinema \
    && chown -R kinema:kinema /data /tmp/runtime-kinema

WORKDIR /app

# Pinned snapshot first so the dependency layer is cached independently of source.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml README.md LICENSE ./
COPY kinema ./kinema
RUN pip install --no-cache-dir --no-deps .

# Bake the weights in so a running container needs no network access.
# Rebuild with --build-arg KINEMA_MODEL_VARIANTS="lite" for a smaller image.
ARG KINEMA_MODEL_VARIANTS="lite full heavy"
RUN python -m kinema.fetch_models ${KINEMA_MODEL_VARIANTS}

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER kinema
ENV XDG_RUNTIME_DIR=/tmp/runtime-kinema

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "kinema.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
