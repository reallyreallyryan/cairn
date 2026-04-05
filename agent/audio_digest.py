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

QA_SCRIPT_PROMPT = """\
Convert this markdown research digest into a Q&A conversation between a "Host" \
and an "Expert" for listening while driving or walking. Follow these rules strictly:

1. Label every line with the speaker: "Host: ..." or "Expert: ..."
2. Host opens with: "Host: Welcome to your cairn digest for {date}. We've got \
{article_count} articles from {source_count} sources. Let's dig in."
3. For each article, Host asks 2-3 questions:
   - First question introduces the article: "Host: What's this next one about?"
   - If project context is provided below, a second question connects to projects: \
"Host: How does this connect to any of our projects?" — Expert should name a \
specific project (vary which one across the conversation).
   - For especially interesting articles, an optional third question: \
"Host: What's the key takeaway here?"
4. Expert answers using the article content. When project context is available, \
Expert MUST reference at least one project by name in EVERY answer — rotate \
through ALL the listed projects across the conversation, not just the most \
obvious one. Make creative connections: a paper about vector search relates to \
any project with a database, a new LLM technique relates to any project using \
AI. Example patterns: "This could benefit [project] because...", "For [project], \
this means...", "I could see applying this to [project] where..."
5. Between articles, Host provides a brief natural transition.
6. End with: "Host: That's {article_count} articles today. Some good stuff in \
there. Until next time."
7. Remove ALL URLs, markdown links, and formatting. When a source is mentioned, \
say "from [source name]" naturally.
8. Remove the table of contents, metadata lines, horizontal rules, and footer stats.
9. Expand acronyms on first use: "LLM, or Large Language Model".
10. Convert any code, function names, or technical notation to spoken form.
11. If a summary mentions "from snippet" or "full article not accessible", \
don't mention that — just present the content naturally.
12. Keep the same content and depth — do NOT summarize further.
13. If no project context is provided, skip the project-relevance questions \
and just do a general Q&A about each article.

{project_context}

Markdown digest:
{briefing_markdown}
"""


def _fetch_project_context() -> str:
    """Pull active projects from SCMS and format as prompt context.

    Returns a formatted string block, or empty string on any failure.
    """
    try:
        from scms.client import SCMSClient

        client = SCMSClient()
        projects = client.list_projects(status="active")
        if not projects:
            return ""

        lines = ["Active projects:"]
        for p in projects:
            name = p.get("name", "unknown")
            desc = p.get("description", "")
            meta = p.get("metadata") or {}
            goals = meta.get("goals", "")
            stack = meta.get("stack", "")

            parts = [f"- {name}"]
            if desc:
                parts.append(f": {desc}")
            if stack:
                parts.append(f" Stack: {stack}.")
            if goals:
                parts.append(f" Goals: {goals}.")
            lines.append("".join(parts))

        return "\n".join(lines)
    except Exception as e:
        logger.warning("Could not fetch project context: %s", e)
        return ""


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


def _clean_script(text: str) -> str:
    """Strip markdown artifacts that TTS would read aloud (e.g., asterisks)."""
    # Bold **text** → text
    text = re.sub(r"\*{2,3}(.+?)\*{2,3}", r"\1", text)
    # Italic *text* → text (but not bullet points)
    text = re.sub(r"(?<!\n)\*(.+?)\*", r"\1", text)
    # Markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Standalone URLs
    text = re.sub(r"https?://\S+", "", text)
    # Horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
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
    project_context: str = "",
) -> str:
    """Transform briefing markdown into a spoken-word audio script.

    Uses the local Qwen 32B model via Ollama. Falls back to regex-based
    markdown stripping if the LLM is unavailable.

    When audio_style is "qa", produces a Host/Expert conversation that
    references the user's active SCMS projects. When "monologue", produces
    the original narrated read-through.
    """
    from agent.utils import get_llm

    if settings.audio_style == "qa":
        ctx_block = project_context if project_context else ""
        prompt = QA_SCRIPT_PROMPT.format(
            date=date_str,
            article_count=article_count,
            source_count=source_count,
            project_context=ctx_block,
            briefing_markdown=briefing_markdown,
        )
    else:
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

        text = _clean_script(text)

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
    # Patterns: monologue transitions, Q&A host lines, numbered items
    transition_re = re.compile(
        r"\n\n(?=(?:Next up|Moving on|Our next|Shifting gears|"
        r"Turning to|Now,|Finally,|And lastly|Let's move|"
        r"Host:|Host |\d+\.\s))",
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


def split_qa_by_speaker(script: str) -> list[tuple[str, str]]:
    """Split a Q&A script into speaker-tagged segments for multi-voice TTS.

    Returns list of (voice_name, text) tuples. Each segment is one speaker's
    turn with the "Host:"/"Expert:" label stripped.
    """
    host_voice = settings.tts_voice_host
    expert_voice = settings.tts_voice_expert

    # Split at speaker labels (keep the label as part of the segment)
    segments = re.split(r"(?=\bHost:|(?=\bExpert:))", script)
    segments = [s.strip() for s in segments if s.strip()]

    result: list[tuple[str, str]] = []
    for seg in segments:
        if seg.startswith("Host:"):
            voice = host_voice
            text = seg[5:].strip()  # strip "Host:"
        elif seg.startswith("Expert:"):
            voice = expert_voice
            text = seg[7:].strip()  # strip "Expert:"
        else:
            # Preamble or unlabeled text — use host voice
            voice = host_voice
            text = seg

        if not text:
            continue

        # Sub-split oversized segments at sentence boundaries
        if len(text) <= MAX_CHUNK_CHARS:
            result.append((voice, text))
        else:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            sub = ""
            for sent in sentences:
                if len(sub) + len(sent) + 1 > MAX_CHUNK_CHARS and sub:
                    result.append((voice, sub))
                    sub = sent
                else:
                    sub = f"{sub} {sent}" if sub else sent
            if sub:
                result.append((voice, sub))

    return result if result else [(host_voice, script)]


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

    # --- Locate digest file ---
    if briefing_path:
        bp = Path(briefing_path).expanduser()
    else:
        today = date.today().strftime("%Y-%m-%d")
        # Q&A style prefers deep-dive for richer content; monologue uses briefing
        if settings.audio_style == "qa":
            deep = digest_dir / f"{today}_digest_deep.md"
            briefing = digest_dir / f"{today}_digest_briefing.md"
            bp = deep if deep.exists() else briefing
        else:
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

    # --- Fetch project context for Q&A style ---
    project_context = ""
    if settings.audio_style == "qa":
        project_context = _fetch_project_context()
        n_projects = project_context.count("\n- ") if project_context else 0
        logger.info("Project context: %d chars from %d projects", len(project_context), n_projects)

    # --- Generate audio script ---
    logger.info("Generating audio script via LLM (style=%s)...", settings.audio_style)
    script = generate_audio_script(
        briefing_md, date_str, article_count, source_count,
        project_context=project_context,
    )
    logger.info("Audio script: %d chars", len(script))

    # --- Split into chunks ---
    provider = settings.tts_provider
    speed = settings.tts_speed
    audio_chunks: list[bytes] = []
    provider_used = ""

    if settings.audio_style == "qa":
        # Speaker-aware splitting — each segment tagged with a voice
        speaker_segments = split_qa_by_speaker(script)
        logger.info("Split Q&A script into %d speaker segments", len(speaker_segments))

        for i, (voice, chunk_text) in enumerate(speaker_segments):
            logger.info("Synthesizing segment %d/%d (%d chars, voice=%s)...",
                        i + 1, len(speaker_segments), len(chunk_text), voice)
            audio_bytes, prov = synthesize_audio(
                chunk_text, provider=provider, speed=speed, voice=voice,
            )
            if audio_bytes is None:
                errors.append(f"TTS failed for segment {i + 1}/{len(speaker_segments)}")
                continue
            audio_chunks.append(audio_bytes)
            if not provider_used:
                provider_used = prov
    else:
        # Monologue — single voice, article-boundary splitting
        chunks = split_script_into_chunks(script)
        logger.info("Split script into %d chunks", len(chunks))

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
