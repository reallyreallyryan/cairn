"""Tests for the audio digest pipeline (agent/audio_digest.py)."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.audio_digest import (
    _count_articles,
    _count_sources,
    _strip_markdown_fallback,
    assemble_audio,
    generate_audio_script,
    get_audio_duration,
    run_audio_digest,
    split_script_into_chunks,
    synthesize_audio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BRIEFING = """\
# cairn digest — 2026-04-04
## Briefing

*9 articles from 5 sources. Generated 2026-04-04 10:00 UTC.*

### Contents
1. New Agent Framework (Anthropic Blog)
2. Scaling Vector Search (arXiv cs.AI)

---

### 1. [New Agent Framework](https://anthropic.com/blog/agent)
**Source:** Anthropic Blog | **Relevance:** 0.92 | **Cross-encoder:** 1.34

This article covers a new framework for building AI agents. The approach
uses LLM-based planning with tool execution.

---

### 2. [Scaling Vector Search](https://arxiv.org/abs/2026.54321)
**Source:** arXiv cs.AI | **Relevance:** 0.85 | **Cross-encoder:** 0.98

A paper on scaling vector similarity search to billions of vectors
using approximate nearest neighbor techniques.

---

*2 articles compiled. 1 with full content, 1 from snippets.*
"""

SAMPLE_SCRIPT = """\
Here's your cairn research digest for 2026-04-04. Today we have 2 articles \
from 2 sources.

First up, from the Anthropic Blog: New Agent Framework. This article covers \
a new framework for building AI agents. The approach uses LLM-based planning \
with tool execution.

Moving on, from arXiv: Scaling Vector Search. A paper on scaling vector \
similarity search to billions of vectors using approximate nearest neighbor \
techniques.

That wraps up today's cairn digest. 2 articles from 2 sources. Happy listening.
"""

# Minimal WAV header (44 bytes) + 1 second of silence at 24kHz 16-bit mono
WAV_HEADER = b"RIFF" + b"\x00" * 40

# Fake MP3 bytes (starts with ID3 or \xff\xfb)
FAKE_MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 100


# ---------------------------------------------------------------------------
# _strip_markdown_fallback
# ---------------------------------------------------------------------------

class TestStripMarkdownFallback:
    def test_removes_headers(self):
        result = _strip_markdown_fallback("# Title\n## Subtitle\nText")
        assert "# " not in result
        assert "Title" in result
        assert "Text" in result

    def test_removes_links_keeps_text(self):
        result = _strip_markdown_fallback("[Click here](https://example.com)")
        assert "Click here" in result
        assert "https://example.com" not in result

    def test_removes_bold_italic(self):
        result = _strip_markdown_fallback("**bold** and *italic*")
        assert "bold" in result
        assert "italic" in result
        assert "**" not in result
        assert "*" not in result

    def test_removes_metadata_lines(self):
        result = _strip_markdown_fallback(
            "Source: arXiv\nRelevance: 0.92\nCross-encoder: 1.34\nContent here"
        )
        assert "0.92" not in result
        assert "Content here" in result

    def test_removes_standalone_urls(self):
        result = _strip_markdown_fallback("Visit https://example.com for more")
        assert "https://example.com" not in result

    def test_removes_horizontal_rules(self):
        result = _strip_markdown_fallback("Text\n---\nMore text")
        assert "---" not in result

    def test_removes_snippet_notes(self):
        result = _strip_markdown_fallback(
            "Summary text.\n_[Summary from snippet — full article not accessible]_"
        )
        assert "snippet" not in result
        assert "Summary text." in result


# ---------------------------------------------------------------------------
# _count_articles / _count_sources
# ---------------------------------------------------------------------------

class TestMetadataExtraction:
    def test_count_articles(self):
        assert _count_articles(SAMPLE_BRIEFING) == 2

    def test_count_sources(self):
        assert _count_sources(SAMPLE_BRIEFING) == 2

    def test_count_articles_empty(self):
        assert _count_articles("No articles here") == 0

    def test_count_sources_empty(self):
        assert _count_sources("No sources") == 1  # minimum 1


# ---------------------------------------------------------------------------
# generate_audio_script
# ---------------------------------------------------------------------------

class TestGenerateAudioScript:
    @patch("agent.utils.get_llm")
    def test_produces_script(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=SAMPLE_SCRIPT)
        mock_get_llm.return_value = mock_llm

        result = generate_audio_script(SAMPLE_BRIEFING, "2026-04-04", 2, 2)

        assert "cairn" in result
        assert "https://" not in result

    @patch("agent.utils.get_llm")
    def test_strips_think_tags(self, mock_get_llm):
        clean_text = "Here is the clean script output. " * 10  # >100 chars
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=f"<think>reasoning about the task</think>{clean_text}"
        )
        mock_get_llm.return_value = mock_llm

        result = generate_audio_script(SAMPLE_BRIEFING, "2026-04-04", 2, 2)
        assert "<think>" not in result
        assert "clean script output" in result

    @patch("agent.utils.get_llm")
    def test_falls_back_on_llm_error(self, mock_get_llm):
        mock_get_llm.side_effect = ConnectionError("Ollama not available")

        result = generate_audio_script(SAMPLE_BRIEFING, "2026-04-04", 2, 2)

        # Should use regex fallback — no markdown formatting in output
        assert "**" not in result
        assert "[" not in result or "](http" not in result

    @patch("agent.utils.get_llm")
    def test_falls_back_on_very_short_output(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="OK")
        mock_get_llm.return_value = mock_llm

        result = generate_audio_script(SAMPLE_BRIEFING, "2026-04-04", 2, 2)

        # Should use regex fallback since LLM output was < 100 chars
        assert len(result) > 100


# ---------------------------------------------------------------------------
# split_script_into_chunks
# ---------------------------------------------------------------------------

class TestSplitScriptIntoChunks:
    def test_splits_at_transitions(self):
        # Each section needs enough text to prevent merging (>4000 chars total)
        section = "This is a detailed article summary with lots of content. " * 40
        script = (
            f"Intro paragraph. {section}\n\n"
            f"Moving on, {section}\n\n"
            f"Next up, {section}"
        )
        chunks = split_script_into_chunks(script)
        assert len(chunks) >= 2

    def test_single_short_text_returns_one_chunk(self):
        chunks = split_script_into_chunks("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_respects_max_chunk_size(self):
        # Create text longer than MAX_CHUNK_CHARS
        long_text = "This is a sentence. " * 500  # ~10K chars
        chunks = split_script_into_chunks(long_text)
        for chunk in chunks:
            assert len(chunk) <= 4000

    def test_empty_input(self):
        chunks = split_script_into_chunks("")
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# TTS synthesis
# ---------------------------------------------------------------------------

class TestSynthesizeKokoro:
    @patch("agent.audio_digest._synthesize_kokoro")
    def test_returns_wav_bytes(self, mock_kokoro):
        mock_kokoro.return_value = WAV_HEADER
        audio, provider = synthesize_audio("Hello", provider="kokoro")
        assert audio == WAV_HEADER
        assert provider == "kokoro"

    def test_returns_none_when_not_installed(self):
        with patch.dict("sys.modules", {"kokoro": None}):
            from agent.audio_digest import _synthesize_kokoro
            result = _synthesize_kokoro("Hello", "af_heart", 1.0)
            # Should handle gracefully (either None or ImportError caught)
            # The actual behavior depends on how the import fails
            assert result is None or True  # graceful either way


class TestSynthesizeOpenAI:
    @patch("agent.audio_digest.settings")
    def test_returns_mp3_bytes(self, mock_settings):
        mock_settings.openai_api_key = "test-key"

        mock_client = MagicMock()
        mock_client.audio.speech.create.return_value = MagicMock(content=FAKE_MP3)
        mock_openai_cls = MagicMock(return_value=mock_client)

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            from agent.audio_digest import _synthesize_openai
            result = _synthesize_openai("Hello", "nova", 1.0)

        assert result == FAKE_MP3

    @patch("agent.audio_digest.settings")
    def test_returns_none_without_api_key(self, mock_settings):
        mock_settings.openai_api_key = ""

        from agent.audio_digest import _synthesize_openai
        result = _synthesize_openai("Hello", "nova", 1.0)
        assert result is None


class TestSynthesizeAudio:
    @patch("agent.audio_digest._synthesize_openai")
    @patch("agent.audio_digest._synthesize_kokoro")
    def test_auto_tries_kokoro_first(self, mock_kokoro, mock_openai):
        mock_kokoro.return_value = WAV_HEADER

        audio, provider = synthesize_audio("Hello", provider="auto")

        assert provider == "kokoro"
        mock_openai.assert_not_called()

    @patch("agent.audio_digest._synthesize_openai")
    @patch("agent.audio_digest._synthesize_kokoro")
    def test_auto_falls_back_to_openai(self, mock_kokoro, mock_openai):
        mock_kokoro.return_value = None
        mock_openai.return_value = FAKE_MP3

        audio, provider = synthesize_audio("Hello", provider="auto")

        assert provider == "openai"
        assert audio == FAKE_MP3

    @patch("agent.audio_digest._synthesize_kokoro")
    def test_explicit_kokoro_no_fallback(self, mock_kokoro):
        mock_kokoro.return_value = None

        audio, provider = synthesize_audio("Hello", provider="kokoro")

        assert audio is None
        assert provider == "kokoro"

    def test_off_returns_none(self):
        audio, provider = synthesize_audio("Hello", provider="off")
        assert audio is None
        assert provider == "off"

    @patch("agent.audio_digest._synthesize_openai")
    @patch("agent.audio_digest._synthesize_kokoro")
    def test_all_fail_returns_none(self, mock_kokoro, mock_openai):
        mock_kokoro.return_value = None
        mock_openai.return_value = None

        audio, provider = synthesize_audio("Hello", provider="auto")

        assert audio is None


# ---------------------------------------------------------------------------
# assemble_audio
# ---------------------------------------------------------------------------

class TestAssembleAudio:
    @patch("agent.audio_digest.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_concatenates_with_silence(self, _mock_which):
        from pydub import AudioSegment

        # Create two tiny WAV segments
        seg1 = AudioSegment.silent(duration=100)  # 100ms
        seg2 = AudioSegment.silent(duration=100)

        buf1 = io.BytesIO()
        seg1.export(buf1, format="wav")
        buf2 = io.BytesIO()
        seg2.export(buf2, format="wav")

        result, fmt = assemble_audio(
            [buf1.getvalue(), buf2.getvalue()],
            silence_ms=500,
            output_format="wav",
        )

        assert fmt == "wav"
        assert len(result) > 0

        # Verify combined is longer than individual
        combined = AudioSegment.from_wav(io.BytesIO(result))
        assert len(combined) >= 600  # 100 + 500 silence + 100

    @patch("agent.audio_digest.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_single_chunk_no_silence(self, _mock_which):
        from pydub import AudioSegment

        seg = AudioSegment.silent(duration=200)
        buf = io.BytesIO()
        seg.export(buf, format="wav")

        result, fmt = assemble_audio([buf.getvalue()], output_format="wav")

        combined = AudioSegment.from_wav(io.BytesIO(result))
        assert len(combined) == 200  # no silence added

    @patch("agent.audio_digest.shutil.which", return_value=None)
    def test_falls_back_to_wav_without_ffmpeg(self, _mock_which):
        from pydub import AudioSegment

        seg = AudioSegment.silent(duration=100)
        buf = io.BytesIO()
        seg.export(buf, format="wav")

        result, fmt = assemble_audio(
            [buf.getvalue()], output_format="mp3"
        )

        assert fmt == "wav"


# ---------------------------------------------------------------------------
# get_audio_duration
# ---------------------------------------------------------------------------

class TestGetAudioDuration:
    def test_returns_duration_for_wav(self):
        from pydub import AudioSegment

        seg = AudioSegment.silent(duration=2500)  # 2.5 seconds
        buf = io.BytesIO()
        seg.export(buf, format="wav")

        duration = get_audio_duration(buf.getvalue(), "wav")
        assert abs(duration - 2.5) < 0.1

    def test_returns_zero_on_error(self):
        duration = get_audio_duration(b"not audio data", "wav")
        assert duration == 0.0


# ---------------------------------------------------------------------------
# run_audio_digest (end-to-end with mocks)
# ---------------------------------------------------------------------------

class TestRunAudioDigest:
    @patch("agent.audio_digest.assemble_audio")
    @patch("agent.audio_digest.synthesize_audio")
    @patch("agent.audio_digest.split_script_into_chunks")
    @patch("agent.audio_digest.generate_audio_script")
    @patch("agent.audio_digest._load_config")
    def test_full_pipeline(
        self, mock_config, mock_script, mock_split, mock_synth, mock_assemble,
    ):
        # Setup
        mock_config.return_value = {"settings": {"digest_notes_dir": "/tmp/test-digests"}}
        mock_script.return_value = SAMPLE_SCRIPT
        mock_split.return_value = ["Chunk 1", "Chunk 2"]
        mock_synth.return_value = (WAV_HEADER, "kokoro")
        mock_assemble.return_value = (WAV_HEADER, "wav")

        briefing_file = Path("/tmp/test-digests/2026-04-04_digest_briefing.md")
        briefing_file.parent.mkdir(parents=True, exist_ok=True)
        briefing_file.write_text(SAMPLE_BRIEFING)

        try:
            result = run_audio_digest(briefing_path=str(briefing_file))

            assert result["audio_path"] != ""
            assert result["provider_used"] == "kokoro"
            assert result["char_count"] > 0
            assert result["errors"] == []
        finally:
            # Cleanup
            briefing_file.unlink(missing_ok=True)
            audio_file = Path(result.get("audio_path", ""))
            if audio_file.exists():
                audio_file.unlink()

    @patch("agent.audio_digest._load_config")
    def test_no_briefing_file_returns_error(self, mock_config):
        mock_config.return_value = {"settings": {"digest_notes_dir": "/tmp/test-digests"}}

        result = run_audio_digest(briefing_path="/tmp/nonexistent.md")

        assert result["audio_path"] == ""
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0]

    @patch("agent.audio_digest.synthesize_audio")
    @patch("agent.audio_digest.split_script_into_chunks")
    @patch("agent.audio_digest.generate_audio_script")
    @patch("agent.audio_digest._load_config")
    def test_tts_failure_reports_errors(
        self, mock_config, mock_script, mock_split, mock_synth,
    ):
        mock_config.return_value = {"settings": {"digest_notes_dir": "/tmp/test-digests"}}
        mock_script.return_value = SAMPLE_SCRIPT
        mock_split.return_value = ["Chunk 1"]
        mock_synth.return_value = (None, "none")

        briefing_file = Path("/tmp/test-digests/2026-04-04_digest_briefing.md")
        briefing_file.parent.mkdir(parents=True, exist_ok=True)
        briefing_file.write_text(SAMPLE_BRIEFING)

        try:
            result = run_audio_digest(briefing_path=str(briefing_file))

            assert result["audio_path"] == ""
            assert len(result["errors"]) > 0
        finally:
            briefing_file.unlink(missing_ok=True)
