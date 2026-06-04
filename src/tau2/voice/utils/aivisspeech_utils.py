"""AivisSpeech TTS adapter shim, callable from synthesize.py.

Accepts the same `ElevenLabsTTSConfig` that the rest of the user simulator
uses so the call site doesn't need provider-specific config handling. Ignores
ElevenLabs-specific fields (voice_id, model_id, voice_settings) and reads
its own knobs from AIVISSPEECH_* env vars. Output is PCM_S16LE at whatever
sample rate the engine returns (44.1 kHz by default); downstream effect
processing resamples to whatever the channel needs.

Talks VOICEVOX-compatible HTTP to a local AivisSpeech engine. No API key
required — the engine runs entirely on the host machine.
"""

from __future__ import annotations

import io
import os
import wave

import httpx
from loguru import logger

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import ElevenLabsTTSConfig

_AIVISSPEECH_HOST = os.getenv("AIVISSPEECH_HOST", "http://127.0.0.1:10101")
# Default = コハク ノーマル. Override via AIVISSPEECH_STYLE_ID (see GET /speakers).
_AIVISSPEECH_STYLE_ID = int(os.getenv("AIVISSPEECH_STYLE_ID", "1878365376"))
# Request engine to resample to the tau2 user-sim PCM rate (DEFAULT_PCM_SAMPLE_RATE
# in tau2/config.py = 16000). The downstream StreamingTelephonyConverter is
# constructed with input_sample_rate=PCM_SAMPLE_RATE and ignores the actual
# AudioFormat.sample_rate of each chunk, so any provider that returns audio at
# a rate other than PCM_SAMPLE_RATE plays back time-stretched (44.1 kHz native
# AivisSpeech → ~2.76x slow without this resample).
_AIVISSPEECH_OUTPUT_SAMPLE_RATE = int(os.getenv("AIVISSPEECH_OUTPUT_SAMPLE_RATE", "16000"))
_AIVISSPEECH_SPEED_SCALE = float(os.getenv("AIVISSPEECH_SPEED_SCALE", "1.0"))
_AIVISSPEECH_PITCH_SCALE = float(os.getenv("AIVISSPEECH_PITCH_SCALE", "0.0"))
_AIVISSPEECH_INTONATION_SCALE = float(os.getenv("AIVISSPEECH_INTONATION_SCALE", "1.0"))
_AIVISSPEECH_VOLUME_SCALE = float(os.getenv("AIVISSPEECH_VOLUME_SCALE", "1.0"))
# Silence ms appended to each clip so maestra's server VAD reliably detects
# "user stopped speaking". Matches the OpenAI TTS shim's default (600 ms vs
# server_vad's 500 ms threshold).
_AIVISSPEECH_TRAILING_SILENCE_MS = int(os.getenv("AIVISSPEECH_TRAILING_SILENCE_MS", "600"))
_AIVISSPEECH_TIMEOUT_S = float(os.getenv("AIVISSPEECH_TIMEOUT_S", "60"))

_AUDIO_TAG_TOKENS = ("[cough]", "[sneeze]", "[sniffle]", "[pause]")


def _strip_audio_tags(text: str) -> str:
    """Remove ElevenLabs-style audio tags that AivisSpeech would read literally."""
    out = text
    for tok in _AUDIO_TAG_TOKENS:
        out = out.replace(tok, "")
    return " ".join(out.split())


def tts_aivisspeech(text: str, config: ElevenLabsTTSConfig) -> AudioData:
    """Synthesize `text` to PCM_S16LE via the local AivisSpeech engine."""
    safe_text = _strip_audio_tags(text)

    text_preview = safe_text[:50] + "…" if len(safe_text) > 50 else safe_text
    logger.debug(
        f"AivisSpeech TTS: synthesizing '{text_preview}' "
        f"(host={_AIVISSPEECH_HOST}, style_id={_AIVISSPEECH_STYLE_ID})"
    )

    base = _AIVISSPEECH_HOST.rstrip("/")
    with httpx.Client(timeout=_AIVISSPEECH_TIMEOUT_S) as client:
        q_resp = client.post(
            f"{base}/audio_query",
            params={"speaker": _AIVISSPEECH_STYLE_ID, "text": safe_text},
        )
        q_resp.raise_for_status()
        query = q_resp.json()
        query["speedScale"] = _AIVISSPEECH_SPEED_SCALE
        query["pitchScale"] = _AIVISSPEECH_PITCH_SCALE
        query["intonationScale"] = _AIVISSPEECH_INTONATION_SCALE
        query["volumeScale"] = _AIVISSPEECH_VOLUME_SCALE
        query["outputSamplingRate"] = _AIVISSPEECH_OUTPUT_SAMPLE_RATE

        s_resp = client.post(
            f"{base}/synthesis",
            params={"speaker": _AIVISSPEECH_STYLE_ID},
            json=query,
            headers={"Content-Type": "application/json"},
        )
        s_resp.raise_for_status()
        wav_bytes = s_resp.content

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        pcm_bytes = w.readframes(w.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"AivisSpeech returned sample_width={sample_width} bytes, expected 2 (PCM_S16LE)."
        )
    if n_channels != 1:
        raise ValueError(
            f"AivisSpeech returned channels={n_channels}, expected 1 (mono)."
        )

    if _AIVISSPEECH_TRAILING_SILENCE_MS > 0:
        silence_samples = int(sample_rate * _AIVISSPEECH_TRAILING_SILENCE_MS / 1000)
        pcm_bytes = pcm_bytes + (b"\x00\x00" * silence_samples)

    return AudioData(
        data=pcm_bytes,
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
            channels=1,
        ),
    )
