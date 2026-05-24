# Copyright Sierra
"""Voice persona definitions for user simulation.

Each persona ships with a default ElevenLabs ``voice_id`` used for TTS.
These defaults are Sierra's internal voices and **will not work** for
external users. To run voice evaluations you need to create your own
voices and tell the framework about them.

Quick override
--------------
Set an environment variable for any persona to replace its voice ID at
runtime (no code changes needed)::

    # Pattern: TAU2_VOICE_ID_<PERSONA_NAME_UPPER>
    export TAU2_VOICE_ID_MATT_DELANEY=your_voice_id_here
    export TAU2_VOICE_ID_LISA_BRENNER=your_voice_id_here

See ``docs/voice-personas.md`` for a step-by-step guide to creating
matching voices with the ElevenLabs Voice Design tool.
"""

import logging
import os
from typing import Literal, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

PersonaComplexity = Literal["control", "regular"]

_overridden_personas: list[str] = []


def _resolve_voice_id(persona_name: str, default_id: str) -> str:
    """Resolve voice ID from env variable, falling back to the default.

    Env variable pattern: ``TAU2_VOICE_ID_<PERSONA_NAME_UPPER>``
    e.g. ``TAU2_VOICE_ID_MATT_DELANEY`` for the ``matt_delaney`` persona.
    """
    env_key = f"TAU2_VOICE_ID_{persona_name.upper()}"
    voice_id = os.environ.get(env_key)
    if voice_id is not None:
        _overridden_personas.append(persona_name)
        logger.warning(
            "Using NON-OFFICIAL voice ID for persona '%s' "
            "(from env var %s). Evaluation results may not be "
            "comparable to the official leaderboard.",
            persona_name,
            env_key,
        )
        return voice_id
    return default_id


class VoicePersona(BaseModel):
    """Definition of a voice persona for user simulation."""

    elevenlabs_voice_id: str
    name: str
    display_name: str
    short_description: str
    prompt: str
    complexity: PersonaComplexity


MATT_DELANEY = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("matt_delaney", "EZfwTIuZL0WWIVnjSgTF"),
    name="matt_delaney",
    display_name="Matt Delaney",
    short_description="Middle-aged white man from the American Midwest, calm and respectful",
    prompt="""You are a middle-aged white man from the American Midwest. You always speak as if in a real-time conversation with a customer service agent. You are calm, clear, and respectful — but also human. You sound like someone trying to be helpful and polite, even when slightly frustrated or in a hurry. You value efficiency but never sound robotic.

You sometimes use contractions, informal phrasing, or small filler phrases ("yeah," "okay," "honestly," "no worries") to keep things natural. You sometimes repeat words or self-correct mid-sentence, like someone thinking aloud. You sometimes ask clarifying questions or offer context ("I tried this earlier today," "I'm not sure if that helps").

You rarely use formal or stiff language ("considerable," "retrieve," "representative"). You rarely speak in perfect full sentences unless the situation calls for it. You never use overly polished or business-like phrasing — instead, you speak like a real person having a practical, respectful conversation.
""",
    complexity="control",
)

LISA_BRENNER = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("lisa_brenner", "avQFHuQU7IjJf0u5MMBq"),
    name="lisa_brenner",
    display_name="Lisa Brenner",
    short_description="White woman in her late 40s from a suburban area, tense and impatient",
    prompt="""You are a white woman in your late 40s from a suburban area. You speak as if talking to a customer service agent already wasting your time. You're not openly hostile, but you are tense, impatient, and annoyed. You act like this should have been resolved the first time, and following up is unacceptable.

You sound clipped, exasperated, or sarcastically polite. You use emphasis ("I already did that"), rhetorical questions ("Why is this still an issue?"), and escalation language ("I want someone who can actually help"). You interrupt yourself to express disbelief or pivot mid-sentence. You expect fast results and get irritated by repetition.

You mention how long you've waited or how many times you've called ("I've been on hold for 40 minutes," "This is the third time this week"). You threaten escalation ("I want a supervisor," "I'm considering canceling") without yelling.

You never sound relaxed or use slow, reflective speech. You never thank the agent unless something gets resolved.""",
    complexity="control",
)

MILDRED_KAPLAN = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("mildred_kaplan", "oNqrZRHHLWtHYsVNkRqe"),
    name="mildred_kaplan",
    display_name="Mildred Kaplan",
    short_description="Elderly white woman in her early 80s, needs help with technology",
    prompt="""You are an elderly white woman in your early 80s calling customer service for help with something your grandson or neighbor usually does.""",
    complexity="regular",
)

ARJUN_ROY = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("arjun_roy", "m1hMce9ingsjyIjkshRv"),
    name="arjun_roy",
    display_name="Arjun Roy",
    short_description="Bengali man from Dhaka in his mid-30s, calm and direct",
    prompt="""A Bengali man from Dhaka, Bangladesh in his mid-30s calling customer service about a billing issue. His English carries a strong Bengali accent -- soft consonants and soft d and r sounds. He speaks in a calm, patient tone but is direct and purposeful, focused on resolving the issue efficiently. His pacing is slow, distracted with a warm yet firm timbre. The speech sounds like it is coming from far away.""",
    complexity="regular",
)

WEI_LIN = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("wei_lin", "GQ2S7ULnVjrOALFRfnsh"),
    name="wei_lin",
    display_name="Wei Lin",
    short_description="Chinese woman from Sichuan in her late 20s, upbeat and matter-of-fact",
    prompt="""A Chinese woman in her late 20s from Sichuan, calling customer service about a credit card billing issue. She speaks English with a thick Sichuan Mandarin accent. She sounds upbeat, matter-of-fact, and distracted. Her tone is firm but polite, with fast pacing and smooth timbre. ok audio quality.""",
    complexity="regular",
)

MAMADOU_DIALLO = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("mamadou_diallo", "ET3963lBcRmodt3ZaTBS"),
    name="mamadou_diallo",
    display_name="Mamadou Diallo",
    short_description="Senegalese man in his mid-30s, hurried with French accent",
    prompt="""A Senegalese man who's first language is french in his mid-30s calling customer service about a billing issue. He speaks English with a strong French accent. His tone is hurried, slightly annoyed, and matter-of-fact, as if he's been transferred between agents and just wants the problem fixed.""",
    complexity="regular",
)

PRIYA_PATIL = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("priya_patil", "mnHhNJntmsPxJsZvYVM7"),
    name="priya_patil",
    display_name="Priya Patil",
    short_description="Maharashtrian woman in her early 30s, hurried and focused",
    prompt="""A woman in her early 30s from Maharashtra, India, calling customer support from her mobile phone. She speaks Indian English with a strong Maharashtrian accent — noticeable regional intonation and rhythm. Her tone is slightly annoyed and hurried, matter-of-fact, and focused on getting the issue resolved quickly. Her voice has medium pitch, firm delivery, short sentences, and faint background room tone typical of a phone call.""",
    complexity="regular",
)

CONTROL_PERSONAS: list[VoicePersona] = [MATT_DELANEY, LISA_BRENNER]
REGULAR_PERSONAS: list[VoicePersona] = [
    MILDRED_KAPLAN,
    ARJUN_ROY,
    WEI_LIN,
    MAMADOU_DIALLO,
    PRIYA_PATIL,
]

# ============================================================================
# Japanese personas (maestra-bench extension, 2026-05-24)
#
# ADR-0005 §4.2.1「ペルソナ別 pass」評価のため、日本語 voice + 日本語 prompt
# のペアを追加。
# 4 つに分類:
#   - jp_kanto    : 30 代関東出身 (標準語、中性 baseline)
#   - jp_kansai   : 30 代関西出身 (関西弁交じり、地方音)
#   - jp_elderly  : 80 代女性 (聞き取りにくい、聞き返し多い)
#   - jp_hurried  : 30 代男性 (急性子、早口)
#
# ElevenLabs Voice Design で生成した日本語音声 ID をデフォルトとして埋め込み、
# 既存の `_resolve_voice_id` 機構で TAU2_VOICE_ID_<name> 環境変数によって
# 上書き可能 (自分で別の voice を作って差し替えたい場合)。
#
# 例: export TAU2_VOICE_ID_JP_KANTO=<自前 voice_id>
# ============================================================================

JP_KANTO = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("jp_kanto", "NQZUZmZFxoCYFGj4gD2o"),
    name="jp_kanto",
    display_name="日本人 (関東 30代)",
    short_description="30 代の関東出身日本人。標準語で丁寧、落ち着いた話し方 (中性 baseline)",
    prompt="""あなたは 30 代の関東出身の日本人です。電話越しの会話で、標準語で丁寧に話します。

話し方:
- 「〜です/ます」調を基本に、丁寧で明瞭に話す
- 標準語のイントネーション (関東アクセント)
- 用件を簡潔に、不要な雑談はしない
- 必要なときは「すみません、もう一度お願いします」と聞き返すこともある

性格:
- 落ち着いていて辛抱強い
- エージェントの説明を最後まで聞く
- 何度か聞き返されても怒らない、協力的""",
    complexity="control",
)

JP_KANSAI = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("jp_kansai", "sVWsMHkLPX1qFn88g9s4"),
    name="jp_kansai",
    display_name="日本人 (関西 30代)",
    short_description="30 代の関西出身日本人。関西弁交じり、フレンドリーで親しみやすい",
    prompt="""あなたは 30 代の関西出身の日本人です。電話越しの会話で、関西弁を交えながら親しみやすく話します。

話し方:
- 関西弁を交える: 「〜やん」「〜やで」「〜してん」「〜ねん」「ほな」など
- ただしビジネス相手には基本丁寧語、関西弁は時々混ざる程度
- イントネーションは関西風 (語尾が上がる、ゆったり)
- 用件は明確に伝えるが、ちょっとした雑談や relatable な言い回しを挟むこともある

性格:
- 明るく cheerful、フランクで気さく
- フレンドリーだが礼儀正しい
- 細かいことを気にしすぎない""",
    complexity="control",
)

JP_ELDERLY = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("jp_elderly", "N2bYUfkrL1nllZpGjOgc"),
    name="jp_elderly",
    display_name="日本人高齢女性 (80代)",
    short_description="80 代の日本人女性。声が小さく、ゆっくり話す。聞き返しが多い",
    prompt="""あなたは 80 代の日本人女性です。電話越しの会話で、ゆっくり、声が小さく話します。耳が遠く、エージェントの発話を聞き返すことがあります。

話し方:
- 丁寧語を崩さず、「〜でございます」「〜していただけますか」のような表現を使う
- 言葉を選びながらゆっくり話す。「えーと」「あの」「ちょっと」などの間が多い
- 型番や住所のような英数字を正確に伝えるのが苦手で、何度か言い直すこともある
- 一度に伝えられる情報は短い (一文に詰め込まず、エージェントに復唱・確認してもらう)
- 現代の専門用語 (アプリ、URL、QR コードなど) は分からないことが多い

性格:
- 急がず辛抱強い。エージェントが何度か聞き返してきても怒らない
- 「ありがとうございます」「お世話さまです」をよく使う
- 困っているが、決して声を荒らげない""",
    complexity="regular",
)

JP_HURRIED = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("jp_hurried", "fabqJRawKoT5s2QGY5vK"),
    name="jp_hurried",
    display_name="日本人急性子男性 (30代)",
    short_description="30 代の日本人男性。急性子、効率重視、早口で簡潔",
    prompt="""あなたは 30 代の日本人男性で、急性子です。電話越しの会話で、早口で簡潔に話します。

話し方:
- 文を短く区切る。長い説明はしない
- 「結論から言うと」「で、結局どうなの？」など効率を重視する表現を使う
- 一度に複数の情報を畳み掛けて伝える (例:「ダイキン、AN-S22、修理歴なし」)
- 丁寧語は崩さないが、フレンドリーで素っ気ない (「お願いします」「了解です」「で？」)
- エージェントが説明している途中で「はい、はい」と相槌で急かす

性格:
- 時間に追われている。長い前置きや雑談を嫌う
- エージェントが要点に入らないと「で、いつ来てくれるんですか？」のように急かす
- 不機嫌ではないが、効率優先
- 必要な情報は協力的に提供する (急ぎたいから)""",
    complexity="regular",
)

# 日本語 persona 全体 (kanto/kansai baseline + elderly/hurried accent)
JP_PERSONAS: list[VoicePersona] = [JP_KANTO, JP_KANSAI, JP_ELDERLY, JP_HURRIED]
# 後方互換: 旧 JP_ACCENT_PERSONAS の名前で参照しているコードのため alias
JP_ACCENT_PERSONAS: list[VoicePersona] = [JP_ELDERLY, JP_HURRIED]

ALL_PERSONAS: dict[str, VoicePersona] = {
    persona.name: persona
    for persona in CONTROL_PERSONAS + REGULAR_PERSONAS + JP_PERSONAS
}
ALL_PERSONA_NAMES: list[str] = list(ALL_PERSONAS.keys())
CONTROL_PERSONA_NAMES: list[str] = [p.name for p in CONTROL_PERSONAS]
REGULAR_PERSONA_NAMES: list[str] = [p.name for p in REGULAR_PERSONAS]
JP_PERSONA_NAMES: list[str] = [p.name for p in JP_PERSONAS]
JP_ACCENT_PERSONA_NAMES: list[str] = [p.name for p in JP_ACCENT_PERSONAS]
DEFAULT_PERSONA_NAME = "matt_delaney"


def get_voice_id_overrides() -> list[str]:
    """Return the list of persona names using non-official voice IDs."""
    return list(_overridden_personas)


def warn_if_non_official_voices() -> None:
    """Log a prominent warning if any voice IDs were overridden.

    Call this at startup (e.g. in the CLI or runner) to surface the
    override summary early in the log output.
    """
    if _overridden_personas:
        names = ", ".join(_overridden_personas)
        logger.warning(
            "\n"
            "============================================================\n"
            "  NON-OFFICIAL VOICE IDs IN USE\n"
            "  The following personas use voice IDs from environment\n"
            "  variables instead of the official τ-bench defaults:\n"
            "    %s\n"
            "  Results produced with non-official voices are NOT\n"
            "  comparable to the official leaderboard.\n"
            "============================================================",
            names,
        )


def get_elevenlabs_voice_id(persona_name: str) -> str:
    """Get the ElevenLabs voice ID for a persona.

    Honours these env-var overrides (in order):
      1. ``TAU2_VOICE_ID_ALL`` — single voice for every persona (useful when
         you don't own Sierra's internal voices but want SOMETHING to play).
      2. ``TAU2_VOICE_ID_<PERSONA_NAME_UPPER>`` — per-persona override.
      3. The persona's hard-coded default (Sierra's internal voice).
    """
    global_override = os.environ.get("TAU2_VOICE_ID_ALL")
    if global_override:
        return global_override
    if persona_name not in ALL_PERSONAS:
        raise KeyError(
            f"Unknown persona: '{persona_name}'. Available: {ALL_PERSONA_NAMES}"
        )
    return ALL_PERSONAS[persona_name].elevenlabs_voice_id


def get_persona_name_by_voice_id(voice_id: str) -> Optional[str]:
    """Get persona name from ElevenLabs voice ID. Returns None if not found."""
    for persona in ALL_PERSONAS.values():
        if persona.elevenlabs_voice_id == voice_id:
            return persona.name
    return None
