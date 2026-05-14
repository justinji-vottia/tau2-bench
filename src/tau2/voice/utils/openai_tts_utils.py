"""OpenAI TTS adapter shim, callable from synthesize.py.

Accepts the same `ElevenLabsTTSConfig` that the rest of the user simulator
uses so the call site doesn't need provider-specific config handling. Ignores
ElevenLabs-specific fields (voice_id, model_id, voice_settings) and uses a
fixed OpenAI voice. The output is PCM_S16LE @ 24 kHz mono; downstream effect
processing resamples to whatever the channel needs.

This shim is provided by the maestra-bench fork of tau2 to make development
without an ElevenLabs key possible. Real benchmark runs should use ElevenLabs.
"""

from __future__ import annotations

import os

from loguru import logger
from openai import OpenAI

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import ElevenLabsTTSConfig

# OpenAI TTS PCM output is fixed: 24 kHz, 16-bit signed little-endian, mono.
_OPENAI_TTS_SAMPLE_RATE = 24000
_OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
# nova/shimmer are the most natural female voices for a customer-on-phone vibe.
# alloy is more robotic; switch back if voice consistency matters more than realism.
_OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")
# 1.0 = normal pace. Tuned down from 1.3 because maestra's gpt-realtime
# server VAD was clipping the last few syllables (notably city names) when
# audio came in fast — slower gives it room to parse before VAD fires.
_OPENAI_TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.0"))
# Silence ms appended to each TTS clip so maestra's server VAD reliably
# detects "user stopped speaking". server_vad's default silence_duration_ms
# is 500 ms; padding 600 ms gives a safety margin.
_OPENAI_TTS_TRAILING_SILENCE_MS = int(os.getenv("OPENAI_TTS_TRAILING_SILENCE_MS", "600"))
# Style steering only supported by gpt-4o-mini-tts. The default is tuned for
# a Japanese caller-on-phone persona; override with OPENAI_TTS_INSTRUCTIONS.
_OPENAI_TTS_INSTRUCTIONS = os.getenv(
    "OPENAI_TTS_INSTRUCTIONS",
    "日本語のネイティブスピーカーとして、自然で穏やかな電話越しの会話の口調で話してください。"
    "わずかに丁寧で、機械的にならず人間らしいリズムで話してください。",
)


def tts_openai(text: str, config: ElevenLabsTTSConfig) -> AudioData:
    """Synthesize `text` to PCM_S16LE @ 24 kHz via OpenAI TTS."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # ElevenLabsTTSConfig fields are intentionally ignored — voice_id maps to
    # ElevenLabs voice strings that don't apply, and audio tags ([cough] etc.)
    # are unsupported by OpenAI TTS. Strip any audio tags so they don't read
    # out literally.
    safe_text = _strip_audio_tags(text)

    text_preview = safe_text[:50] + "…" if len(safe_text) > 50 else safe_text
    logger.debug(
        f"OpenAI TTS: synthesizing '{text_preview}' "
        f"(model={_OPENAI_TTS_MODEL}, voice={_OPENAI_TTS_VOICE})"
    )

    client = OpenAI(api_key=api_key)
    create_kwargs = {
        "model": _OPENAI_TTS_MODEL,
        "voice": _OPENAI_TTS_VOICE,
        "input": safe_text,
        "response_format": "pcm",  # headerless 24kHz mono 16-bit signed LE
        "speed": _OPENAI_TTS_SPEED,
    }
    # instructions is only accepted by gpt-4o-mini-tts; silently skip for tts-1*.
    if _OPENAI_TTS_INSTRUCTIONS and "mini" in _OPENAI_TTS_MODEL:
        create_kwargs["instructions"] = _OPENAI_TTS_INSTRUCTIONS
    response = client.audio.speech.create(**create_kwargs)
    audio_bytes = response.read()

    # Pad with PCM16 silence so the agent's server-side VAD has a clear
    # end-of-speech marker. PCM_S16LE silence = 0x00 bytes; 2 bytes/sample.
    if _OPENAI_TTS_TRAILING_SILENCE_MS > 0:
        silence_samples = int(
            _OPENAI_TTS_SAMPLE_RATE * _OPENAI_TTS_TRAILING_SILENCE_MS / 1000
        )
        audio_bytes = audio_bytes + (b"\x00\x00" * silence_samples)

    return AudioData(
        data=audio_bytes,
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=_OPENAI_TTS_SAMPLE_RATE,
            channels=1,
        ),
    )


_AUDIO_TAG_TOKENS = ("[cough]", "[sneeze]", "[sniffle]", "[pause]")


def _strip_audio_tags(text: str) -> str:
    """Remove ElevenLabs-style audio tags that OpenAI TTS would read literally."""
    out = text
    for tok in _AUDIO_TAG_TOKENS:
        out = out.replace(tok, "")
    return " ".join(out.split())
