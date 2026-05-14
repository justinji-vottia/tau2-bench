import os
import re
from copy import deepcopy
from typing import Tuple

from elevenlabs import ElevenLabs
from loguru import logger

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import (
    DEFAULT_ELEVEN_LAB_AUDIO_FORMAT,
    ElevenLabsTTSConfig,
)


def make_elevenlabs_output_format(codec: str, sample_rate: int, bitrate: int) -> str:
    """
    Make an ElevenLabs output format string from codec, sample rate, and bitrate

    Args:
        codec: The codec to use
        sample_rate: The sample rate to use
        bitrate: The bitrate to use

    Returns:
        The ElevenLabs output format string
    """
    return f"{codec}_{sample_rate}_{bitrate}"


def parse_elevenlabs_output_format(output_format: str) -> Tuple[str, int, int]:
    """
    Parse an ElevenLabs output format string into codec, sample rate, and bitrate

    Args:
        output_format: The ElevenLabs output format string

    Returns:
        Tuple of (codec, sample rate, bitrate)
    """
    pat = re.compile(r"^(\w+)_(\d+)_(\d+)$")
    match = pat.match(output_format)
    if not match:
        raise ValueError(f"Invalid ElevenLabs output format: {output_format}")
    return match.group(1), int(match.group(2)), int(match.group(3))


AUDIO_TAG_PATTERN = re.compile(r"\[(cough|sneeze|sniffle)\]")
PAUSE_TAG_PATTERN = re.compile(r"\[pause\]", re.IGNORECASE)


def tts_elevenlabs(
    text: str,
    config: ElevenLabsTTSConfig,
) -> AudioData:
    """Text to speech using ElevenLabs API

    Args:
        text: The text to synthesize
        config: The configuration to use
    Returns:
        AudioData with the specified output format
    """
    api_key = config.api_key or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not found in config or environment")

    # Guard: ElevenLabs rejects empty/whitespace-only input with HTTP 400
    # `input_text_empty`. The user simulator occasionally produces a turn
    # that ElevenLabs sees as empty after THEIR speaker-tag + emoji strip
    # (e.g. backchannels like "[cough]" or pure emoji turns). We mirror their
    # strip locally and short-circuit to a silence placeholder so a single
    # weird turn doesn't tank the whole sim.
    def _looks_empty_to_elevenlabs(s: str) -> bool:
        stripped = s
        # ElevenLabs strips speaker tags `[name]:` and known audio tags
        stripped = re.sub(r"\[[^\]]{1,32}\]\s*:?", "", stripped)
        # …and emojis (anything outside BMP printable + JP/EN ASCII)
        stripped = re.sub(
            r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF]+",
            "",
            stripped,
        )
        return not stripped.strip()

    if _looks_empty_to_elevenlabs(text):
        logger.debug(
            "ElevenLabs TTS: skipping empty-after-strip input ({!r}) — returning silence",
            text[:40],
        )
        fmt = DEFAULT_ELEVEN_LAB_AUDIO_FORMAT
        silence_samples = int(fmt.sample_rate * 0.05)  # 50 ms PCM silence
        return AudioData(
            data=(b"\x00\x00" * silence_samples),
            format=deepcopy(fmt),
        )

    is_v3 = "v3" in config.model_id.lower()

    # Replace [pause] with ellipsis for non-v3 models (v3 supports [pause] natively)
    if not is_v3 and PAUSE_TAG_PATTERN.search(text):
        text = PAUSE_TAG_PATTERN.sub("...", text)

    # Warn if audio tags are present but model isn't v3 (tags only work with v3)
    if AUDIO_TAG_PATTERN.search(text) and not is_v3:
        logger.warning(
            f"Audio tags detected in text but model is {config.model_id}. "
            "Audio tags like [cough], [sneeze], [sniffle] only work with v3 models."
        )

    client = ElevenLabs(api_key=api_key)

    voice_id = config.voice_id

    # Log before making API call to help diagnose timeouts
    text_preview = text[:50] + "..." if len(text) > 50 else text
    logger.debug(
        f"ElevenLabs TTS: calling API for text '{text_preview}' "
        f"(voice_id={voice_id}, model={config.model_id})"
    )

    try:
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            voice_settings=config.voice_settings,
            model_id=config.model_id,
            output_format=config.output_format_name,
            seed=config.seed,
        )

        audio_bytes = b"".join(audio)
    except Exception as e:
        # Defensive: ElevenLabs occasionally returns 400 input_text_empty even
        # when our pre-filter passed (rare backchannels with characters we
        # didn't strip). Don't tank the whole sim on these — fall through to
        # the silence placeholder. Other errors keep their normal retry path.
        if "input_text_empty" in str(e) or "validation_error" in str(e).lower():
            logger.warning(
                f"ElevenLabs TTS rejected input as empty after strip "
                f"(text={text_preview!r}) — returning silence placeholder"
            )
            fmt = DEFAULT_ELEVEN_LAB_AUDIO_FORMAT
            silence_samples = int(fmt.sample_rate * 0.05)
            return AudioData(
                data=(b"\x00\x00" * silence_samples),
                format=deepcopy(fmt),
            )
        logger.debug(
            f"ElevenLabs TTS API call failed: {type(e).__name__}: {e} "
            f"(text='{text_preview}', voice_id={voice_id}, model={config.model_id})"
        )
        raise

    logger.debug(f"ElevenLabs TTS: received {len(audio_bytes)} bytes of audio")

    # Validate that we received audio data
    if len(audio_bytes) == 0:
        logger.error(f"ElevenLabs TTS returned empty audio for text: '{text}'")
        raise ValueError(f"ElevenLabs TTS returned empty audio for text: '{text}'")

    # Append a tail of PCM silence so downstream VAD (e.g. maestra's
    # gpt-realtime server_vad with silence_duration_ms≈500) reliably detects
    # end-of-utterance instead of waiting for tau2's idle silence frames.
    # Env override: TAU2_ELEVENLABS_TRAILING_SILENCE_MS (default 600).
    trailing_ms = int(os.getenv("TAU2_ELEVENLABS_TRAILING_SILENCE_MS", "600"))
    if trailing_ms > 0:
        fmt = config.output_audio_format
        silence_samples = int(fmt.sample_rate * trailing_ms / 1000)
        # PCM_S16LE silence = 0x00 bytes; 2 bytes per sample, mono.
        audio_bytes = audio_bytes + (b"\x00\x00" * silence_samples)

    audio_data = AudioData(
        data=audio_bytes,
        format=deepcopy(config.output_audio_format),
    )

    return audio_data
