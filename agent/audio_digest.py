"""Audio Digest — converts briefing markdown to spoken-word audio via TTS."""

import io
import logging
import re
import shutil
from datetime import date
from pathlib import Path

import yaml

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load digest_sources.yaml."""
    path = Path(settings.digest_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Digest config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _get_digest_dir(config: dict) -> Path:
    """Return the digest output directory, creating it if needed."""
    notes_dir = config.get("settings", {}).get(
        "digest_notes_dir", "~/Documents/cairn/digests"
    )
    digest_dir = Path(notes_dir).expanduser()
    digest_dir.mkdir(parents=True, exist_ok=True)
    return digest_dir


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

SCRIPT_PROMPT = """\
Convert this markdown research digest into a natural audio script for listening \
while driving or walking. Follow these rules strictly:

1. Start with: "Here's your cairn research digest for {date}. Today we have \
{article_count} articles from {source_count} sources."
2. Remove ALL URLs, markdown links, and formatting. When a source is mentioned, \
say "from [source name]" naturally.
3. Remove the table of contents entirely.
4. Remove metadata lines (Relevance scores, Cross-encoder scores, Source labels).
5. Remove horizontal rules and any footer/compilation stats.
6. Between articles, add a brief spoken transition like "Next up", "Moving on", \
"Our next story", "Shifting gears". Vary these — don't repeat the same one.
7. Expand acronyms on first use: "LLM, or Large Language Model", \
"RAG, Retrieval Augmented Generation".
8. Convert any code, function names, or technical notation to spoken form.
9. If a summary mentions "from snippet" or "full article not accessible", \
don't mention that — just present the summary naturally.
10. End with: "That wraps up today's cairn digest. {article_count} articles \
from {source_count} sources. Happy listening."
11. Keep the same content and depth — do NOT summarize further. This is a \
format conversion, not a re-summarization.

Markdown digest:
{briefing_markdown}
"""


def _strip_markdown_fallback(text: str) -> str:
    """Regex-based markdown stripping for when LLM is unavailable."""
    # Remove horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    # Remove markdown headers (keep the text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Convert links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove standalone URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove metadata lines (Source: ... | Relevance: ... | Cross-encoder: ...)
    text = re.sub(r"^(Source|Relevance|Cross-encoder):.*$", "", text, flags=re.MULTILINE)
    # Remove snippet-fallback notes
    text = re.sub(r"_\[Summary from snippet.*?\]_", "", text)
    # Remove footer stats
    text = re.sub(r"^\*\d+ articles compiled.*$", "", text, flags=re.MULTILINE)
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _count_sources(briefing_markdown: str) -> int:
    """Count unique sources mentioned in the briefing."""
    sources = set(re.findall(r"\*\*Source:\*\*\s*([^|]+)", briefing_markdown))
    return max(len(sources), 1)


def _count_articles(briefing_markdown: str) -> int:
    """Count articles in the briefing by numbered headers."""
    return len(re.findall(r"^### \d+\.", briefing_markdown, flags=re.MULTILINE))


def generate_audio_script(
    briefing_markdown: str,
    date_str: str,
    article_count: int,
    source_count: int,
) -> str:
    """Transform briefing markdown into a spoken-word audio script.

    Uses the local Qwen 32B model via Ollama. Falls back to regex-based
    markdown stripping if the LLM is unavailable.
    """
    from agent.utils import get_llm

    prompt = SCRIPT_PROMPT.format(
        date=date_str,
        article_count=article_count,
        source_count=source_count,
        briefing_markdown=briefing_markdown,
    )

    try:
        llm = get_llm("local")
        response = llm.invoke(prompt)
        text = response.content if isinstance(response.content, str) else str(response.content)
        # Strip <think> blocks from Qwen output
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        if len(text) < 100:
            logger.warning("LLM produced very short script (%d chars), using fallback", len(text))
            return _strip_markdown_fallback(briefing_markdown)

        return text
    except Exception as e:
        logger.warning("LLM script generation failed: %s — using regex fallback", e)
        return _strip_markdown_fallback(briefing_markdown)


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 4000  # Safe for OpenAI TTS limit; Kokoro handles longer


def split_script_into_chunks(script: str) -> list[str]:
    """Split audio script at article boundaries.

    Looks for transition patterns (numbered articles, transition phrases)
    and splits there. Each chunk stays under MAX_CHUNK_CHARS.
    """
    # Try splitting on double newlines that precede article transitions
    # Patterns: "Next up", "Moving on", "Our next", "Shifting gears", numbered items
    transition_re = re.compile(
        r"\n\n(?=(?:Next up|Moving on|Our next|Shifting gears|"
        r"Turning to|Now,|Finally,|And lastly|Let's move|\d+\.\s))",
        re.IGNORECASE,
    )

    parts = transition_re.split(script)
    parts = [p.strip() for p in parts if p.strip()]

    # If no transitions found, split on double newlines
    if len(parts) <= 1:
        parts = [p.strip() for p in script.split("\n\n") if p.strip()]

    # Merge small chunks and split oversized ones
    chunks = []
    current = ""
    for part in parts:
        if len(current) + len(part) + 2 > MAX_CHUNK_CHARS and current:
            chunks.append(current)
            current = part
        else:
            current = f"{current}\n\n{part}" if current else part

    if current:
        chunks.append(current)

    # Final pass: split any remaining oversized chunks at sentence boundaries
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= MAX_CHUNK_CHARS:
            final_chunks.append(chunk)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            sub = ""
            for sent in sentences:
                if len(sub) + len(sent) + 1 > MAX_CHUNK_CHARS and sub:
                    final_chunks.append(sub)
                    sub = sent
                else:
                    sub = f"{sub} {sent}" if sub else sent
            if sub:
                final_chunks.append(sub)

    return final_chunks if final_chunks else [script]


# ---------------------------------------------------------------------------
# TTS synthesis
# ---------------------------------------------------------------------------

def _synthesize_kokoro(text: str, voice: str, speed: float) -> bytes | None:
    """Synthesize speech using Kokoro TTS locally.

    Returns WAV bytes, or None if Kokoro is not installed or fails.
    """
    try:
        from kokoro import KPipeline
    except ImportError:
        logger.info("kokoro not installed — skipping local TTS")
        return None

    try:
        import numpy as np
        import soundfile as sf

        pipeline = KPipeline(lang_code="a")  # American English
        audio_segments = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            audio_segments.append(audio)

        if not audio_segments:
            logger.warning("Kokoro produced no audio segments")
            return None

        full_audio = np.concatenate(audio_segments)

        buffer = io.BytesIO()
        sf.write(buffer, full_audio, samplerate=24000, format="WAV")
        return buffer.getvalue()

    except Exception as e:
        logger.warning("Kokoro TTS failed: %s", e)
        return None


def _synthesize_openai(text: str, voice: str, speed: float) -> bytes | None:
    """Synthesize speech using OpenAI TTS API.

    Returns MP3 bytes, or None if unavailable.
    Cost: ~$0.015 per 1,000 characters.
    """
    if not settings.openai_api_key:
        logger.warning("No OPENAI_API_KEY — cannot use OpenAI TTS fallback")
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=speed,
            response_format="mp3",
        )

        return response.content

    except Exception as e:
        logger.warning("OpenAI TTS failed: %s", e)
        return None


def synthesize_audio(
    text: str,
    provider: str = "auto",
    voice: str | None = None,
    speed: float = 1.0,
) -> tuple[bytes | None, str]:
    """Synthesize speech from text using the configured TTS provider.

    Returns:
        Tuple of (audio_bytes, provider_name). audio_bytes is None if all fail.
    """
    if provider == "off":
        return None, "off"

    if provider in ("auto", "kokoro"):
        kokoro_voice = voice or settings.tts_voice
        result = _synthesize_kokoro(text, kokoro_voice, speed)
        if result is not None:
            return result, "kokoro"
        if provider == "kokoro":
            return None, "kokoro"

    if provider in ("auto", "openai"):
        openai_voice = voice or settings.tts_openai_voice
        result = _synthesize_openai(text, openai_voice, speed)
        if result is not None:
            return result, "openai"

    logger.error("All TTS providers failed for chunk (%d chars)", len(text))
    return None, "none"


# ---------------------------------------------------------------------------
# Audio assembly
# ---------------------------------------------------------------------------

def assemble_audio(
    chunks: list[bytes],
    silence_ms: int = 1500,
    output_format: str = "mp3",
) -> tuple[bytes, str]:
    """Concatenate audio chunks with silence between them.

    Returns:
        Tuple of (audio_bytes, actual_format). Falls back to WAV if
        ffmpeg is not available for MP3 encoding.
    """
    from pydub import AudioSegment

    silence = AudioSegment.silent(duration=silence_ms)
    combined = AudioSegment.empty()

    for i, chunk_bytes in enumerate(chunks):
        buf = io.BytesIO(chunk_bytes)
        # Detect format from bytes header
        if chunk_bytes[:4] == b"RIFF":
            segment = AudioSegment.from_wav(buf)
        else:
            segment = AudioSegment.from_mp3(buf)

        if i > 0:
            combined += silence
        combined += segment

    # Export
    actual_format = output_format
    if output_format == "mp3" and not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found — saving as WAV instead of MP3")
        actual_format = "wav"

    buffer = io.BytesIO()
    combined.export(buffer, format=actual_format)
    return buffer.getvalue(), actual_format


def get_audio_duration(audio_bytes: bytes, fmt: str) -> float:
    """Return duration in seconds from audio bytes."""
    try:
        from pydub import AudioSegment

        buf = io.BytesIO(audio_bytes)
        if fmt == "wav":
            seg = AudioSegment.from_wav(buf)
        else:
            seg = AudioSegment.from_mp3(buf)
        return len(seg) / 1000.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_audio_digest(
    briefing_path: str | None = None,
    since: str | None = None,
) -> dict:
    """Generate audio from the briefing digest.

    If briefing_path is provided, reads that file directly.
    Otherwise, looks for today's briefing in the digests directory.

    Returns:
        Dict with audio_path, duration_seconds, provider_used,
        char_count, and errors.
    """
    errors: list[str] = []
    config = _load_config()
    digest_dir = _get_digest_dir(config)

    # --- Locate briefing file ---
    if briefing_path:
        bp = Path(briefing_path).expanduser()
    else:
        today = date.today().strftime("%Y-%m-%d")
        bp = digest_dir / f"{today}_digest_briefing.md"

    if not bp.exists():
        return {
            "audio_path": "",
            "duration_seconds": 0.0,
            "provider_used": "",
            "char_count": 0,
            "errors": [f"Briefing file not found: {bp}"],
        }

    briefing_md = bp.read_text(encoding="utf-8")
    logger.info("Read briefing (%d chars) from %s", len(briefing_md), bp)

    # --- Extract metadata ---
    article_count = _count_articles(briefing_md)
    source_count = _count_sources(briefing_md)
    # Extract date from filename or use today
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", bp.name)
    date_str = date_match.group(1) if date_match else date.today().strftime("%Y-%m-%d")

    # --- Generate audio script ---
    logger.info("Generating audio script via LLM...")
    script = generate_audio_script(briefing_md, date_str, article_count, source_count)
    logger.info("Audio script: %d chars", len(script))

    # --- Split into chunks ---
    chunks = split_script_into_chunks(script)
    logger.info("Split script into %d chunks", len(chunks))

    # --- Synthesize each chunk ---
    provider = settings.tts_provider
    speed = settings.tts_speed
    audio_chunks: list[bytes] = []
    provider_used = ""

    for i, chunk_text in enumerate(chunks):
        logger.info("Synthesizing chunk %d/%d (%d chars)...", i + 1, len(chunks), len(chunk_text))
        audio_bytes, prov = synthesize_audio(chunk_text, provider=provider, speed=speed)
        if audio_bytes is None:
            errors.append(f"TTS failed for chunk {i + 1}/{len(chunks)}")
            continue
        audio_chunks.append(audio_bytes)
        if not provider_used:
            provider_used = prov

    if not audio_chunks:
        return {
            "audio_path": "",
            "duration_seconds": 0.0,
            "provider_used": provider_used or "none",
            "char_count": len(script),
            "errors": errors or ["All TTS synthesis failed"],
        }

    # --- Assemble final audio ---
    logger.info("Assembling %d audio chunks...", len(audio_chunks))
    try:
        final_audio, actual_format = assemble_audio(
            audio_chunks,
            silence_ms=1500,
            output_format=settings.tts_format,
        )
    except Exception as e:
        logger.error("Audio assembly failed: %s", e)
        return {
            "audio_path": "",
            "duration_seconds": 0.0,
            "provider_used": provider_used,
            "char_count": len(script),
            "errors": errors + [f"Audio assembly failed: {e}"],
        }

    # --- Save ---
    audio_filename = f"{date_str}_digest_audio.{actual_format}"
    audio_path = digest_dir / audio_filename
    audio_path.write_bytes(final_audio)
    logger.info("Saved audio digest to %s", audio_path)

    duration = get_audio_duration(final_audio, actual_format)

    # --- Notify ---
    try:
        from agent.notifications import notify

        duration_min = duration / 60
        notify(
            "Audio Digest Ready",
            f"{article_count} articles, {duration_min:.1f} min — {audio_path.name}",
        )
    except Exception:
        pass

    return {
        "audio_path": str(audio_path),
        "duration_seconds": duration,
        "provider_used": provider_used,
        "char_count": len(script),
        "errors": errors,
    }
