"""Core voice synthesis (TTS) functions."""

import os

from dotenv import load_dotenv
from loguru import logger

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.utils.retry import tts_retry
from tau2.voice.utils.elevenlabs_utils import tts_elevenlabs

load_dotenv()

ProviderConfig = ElevenLabsTTSConfig


def _provider_with_fallback(provider: str) -> str:
    """Auto-fall back to OpenAI TTS when ElevenLabs key is missing.

    This is for development environments without an ElevenLabs key — the user
    sets OPENAI_API_KEY only and we transparently route to OpenAI's TTS.
    """
    if provider == "elevenlabs" and not os.getenv("ELEVENLABS_API_KEY"):
        if os.getenv("OPENAI_API_KEY"):
            logger.warning(
                "ELEVENLABS_API_KEY missing — falling back to OpenAI TTS for user "
                "simulator voice. Set ELEVENLABS_API_KEY to use ElevenLabs."
            )
            return "openai"
    return provider


@tts_retry
def synthesize_voice(
    text: str,
    provider: str,
    provider_config: ProviderConfig,
) -> AudioData:
    """Synthesize voice from text using the specified configuration."""
    provider = _provider_with_fallback(provider)

    if provider == "elevenlabs":
        audio_data = tts_elevenlabs(text=text, config=provider_config)
    elif provider == "openai":
        # Local import keeps openai-only deps out of the elevenlabs path.
        from tau2.voice.utils.openai_tts_utils import tts_openai

        audio_data = tts_openai(text=text, config=provider_config)
    else:
        raise ValueError(f"Unsupported synthesis provider: {provider}")

    if not audio_data.format.is_pcm16:
        raise ValueError(
            f"TTS must output PCM_S16LE, got {audio_data.format.encoding}. "
            "Configure the provider to use PCM output format."
        )

    return audio_data
