"""LocalAgreement-2 commit logic against a scripted fake Whisper model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voicerecon import streaming


@dataclass
class FakeWord:
    word: str
    start: float
    end: float


@dataclass
class FakeSegment:
    words: list[FakeWord]


class ScriptedModel:
    """Returns the next scripted hypothesis on each ``transcribe`` call.

    The buffer's actual samples are ignored — tests set the hypothesis
    sequence directly. Word ``end`` times are used by
    :class:`StreamingTranscriber` to trim the buffer, so they must be
    consistent with the samples the test pushes.
    """

    def __init__(self, hypotheses: list[list[FakeWord]]):
        self._hypotheses = list(hypotheses)
        self.calls: list[int] = []
        self.prompts: list[str | None] = []

    def transcribe(
        self,
        audio,
        language=None,
        vad_filter=False,
        word_timestamps=True,
        initial_prompt=None,
    ):
        self.calls.append(int(audio.size))
        self.prompts.append(initial_prompt)
        if not self._hypotheses:
            return iter([FakeSegment(words=[])]), None
        hyp = self._hypotheses.pop(0)
        return iter([FakeSegment(words=list(hyp))]), None


def _seconds(n: float) -> np.ndarray:
    return np.zeros(int(n * streaming.SAMPLE_RATE), dtype=np.float32)


def _st(hypotheses, *, min_chunk_seconds: float = 1.0) -> streaming.StreamingTranscriber:
    model = ScriptedModel(hypotheses)
    return streaming.StreamingTranscriber(
        lambda: model, min_chunk_seconds=min_chunk_seconds
    )


def test_commit_step_returns_empty_before_min_chunk():
    st = _st([])
    st.feed(_seconds(0.3))
    assert st.commit_step() == ""


def test_first_hypothesis_commits_nothing():
    """LA-2 needs two agreeing runs. The very first hypothesis has no
    previous run to compare against, so nothing is committed yet."""
    hyp1 = [FakeWord(" hello", 0.0, 0.4), FakeWord(" world", 0.4, 0.8)]
    st = _st([hyp1])
    st.feed(_seconds(1.5))
    assert st.commit_step() == ""


def test_two_agreeing_hypotheses_commit_common_prefix():
    hyp1 = [FakeWord(" hello", 0.0, 0.4), FakeWord(" world", 0.4, 0.8)]
    hyp2 = [
        FakeWord(" hello", 0.0, 0.4),
        FakeWord(" world", 0.4, 0.8),
        FakeWord(" today", 0.8, 1.2),
    ]
    st = _st([hyp1, hyp2])
    st.feed(_seconds(1.5))
    assert st.commit_step() == ""  # priming
    st.feed(_seconds(1.0))
    assert st.commit_step() == " hello world"


def test_disagreement_commits_only_the_agreed_prefix():
    hyp1 = [
        FakeWord(" hello", 0.0, 0.4),
        FakeWord(" word", 0.4, 0.8),
        FakeWord(" earth", 0.8, 1.2),
    ]
    hyp2 = [
        FakeWord(" hello", 0.0, 0.4),
        FakeWord(" world", 0.4, 0.8),
    ]
    st = _st([hyp1, hyp2])
    st.feed(_seconds(1.5))
    st.commit_step()  # priming
    st.feed(_seconds(0.5))
    assert st.commit_step() == " hello"


def test_buffer_is_trimmed_after_commit():
    """After committing, the next transcription runs on audio *after* the
    last committed word's end time — not the whole buffer."""
    hyp1 = [FakeWord(" a", 0.0, 0.5), FakeWord(" b", 0.5, 1.0)]
    hyp2 = [FakeWord(" a", 0.0, 0.5), FakeWord(" b", 0.5, 1.0)]
    model = ScriptedModel([hyp1, hyp2])
    st = streaming.StreamingTranscriber(lambda: model, min_chunk_seconds=1.0)
    st.feed(_seconds(1.5))
    st.commit_step()  # priming — sees 1.5s
    st.feed(_seconds(0.5))
    st.commit_step()  # commits " a b"; end=1.0 → 1.0s trimmed off
    # After trim: 2.0s − 1.0s = 1.0s remaining in the buffer.
    assert model.calls == [24000, 32000]  # 1.5s and 2.0s at 16kHz


def test_no_double_commit_across_steps():
    """A word committed once must not be re-emitted on the next step."""
    hyp1 = [FakeWord(" hi", 0.0, 0.5)]
    hyp2 = [FakeWord(" hi", 0.0, 0.5)]
    # After trimming to end=0.5s, the tail is empty; the third run also
    # returns empty, so nothing further should be committed.
    hyp3: list[FakeWord] = []
    st = _st([hyp1, hyp2, hyp3])
    st.feed(_seconds(1.5))
    st.commit_step()
    st.feed(_seconds(0.2))
    first = st.commit_step()
    st.feed(_seconds(1.0))
    second = st.commit_step()
    assert first == " hi"
    assert second == ""


def test_finalize_returns_pending_tail_and_resets():
    hyp = [FakeWord(" tail", 0.0, 0.4)]
    st = _st([hyp])
    st.feed(_seconds(1.0))
    text = st.finalize()
    assert text == " tail"
    # After finalize, state is reset — a subsequent commit_step needs a
    # fresh min-chunk of audio.
    assert st.commit_step() == ""


def test_finalize_on_empty_buffer_is_a_noop():
    st = _st([])
    assert st.finalize() == ""


def test_reset_clears_buffer_and_prev_hypothesis():
    hyp1 = [FakeWord(" a", 0.0, 0.5)]
    hyp2 = [FakeWord(" a", 0.0, 0.5)]
    st = _st([hyp1, hyp2])
    st.feed(_seconds(1.5))
    st.commit_step()  # priming; prev_words populated
    st.reset()
    # Without reset the second call would have committed " a"; after
    # reset the prev-hypothesis is gone so commit needs another priming.
    st.feed(_seconds(1.5))
    assert st.commit_step() == ""


def test_backend_failure_returns_empty_without_advancing_state():
    class ExplodingModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("no model")

    st = streaming.StreamingTranscriber(lambda: ExplodingModel(), min_chunk_seconds=1.0)
    st.feed(_seconds(1.5))
    assert st.commit_step() == ""
    assert st.finalize() == ""


def test_feed_caps_buffer_at_max_seconds_and_resets_priming():
    """Non-stop audio without a VAD end must not grow the buffer unbounded."""
    hyp = [FakeWord(" a", 0.0, 0.5)]
    st = _st([hyp, hyp])
    st.feed(_seconds(1.5))
    st.commit_step()  # priming — prev_words populated
    # Now dump 40s of audio in one shot — well past MAX_BUFFER_SECONDS.
    st.feed(_seconds(40))
    # Buffer capped and priming reset, so the next commit_step needs
    # another priming pass rather than immediately committing.
    st.feed(_seconds(0.1))
    assert st.commit_step() == ""


def test_prefix_comparison_tolerates_leading_space_and_case_drift():
    """Whisper's tokenisation shifts between passes (leading space,
    capitalisation) on the same audio; a byte-exact compare would defeat
    agreement on real speech, so the check normalises before comparing.
    The current pass's formatting is emitted as-is."""
    hyp1 = [FakeWord(" the", 0.0, 0.3), FakeWord(" Terraform", 0.3, 0.9)]
    hyp2 = [FakeWord("The", 0.0, 0.3), FakeWord(" terraform", 0.3, 0.9)]
    st = _st([hyp1, hyp2])
    st.feed(_seconds(1.5))
    st.commit_step()  # priming
    st.feed(_seconds(0.5))
    committed = st.commit_step()
    # Both words agree after strip + lower even though hyp2 differs in
    # leading-space and capitalisation, so the whole prefix is emitted.
    assert committed.strip().lower() == "the terraform"
    assert committed == "The terraform"


def test_committed_text_is_fed_back_as_initial_prompt():
    """The just-committed words go to Whisper as ``initial_prompt`` on the
    next pass so the decoder produces consistent tokenisation for the
    fresh audio (the standard whisper-streaming stabiliser)."""
    hyp1 = [FakeWord(" hello", 0.0, 0.4), FakeWord(" world", 0.4, 0.8)]
    hyp2 = [FakeWord(" hello", 0.0, 0.4), FakeWord(" world", 0.4, 0.8)]
    hyp3 = [FakeWord(" hello", 0.0, 0.4), FakeWord(" world", 0.4, 0.8)]
    model = ScriptedModel([hyp1, hyp2, hyp3])
    st = streaming.StreamingTranscriber(lambda: model, min_chunk_seconds=1.0)
    st.feed(_seconds(1.5))
    st.commit_step()  # priming; no prompt yet
    st.feed(_seconds(0.5))
    st.commit_step()  # commits " hello world"; still no prompt in flight
    st.feed(_seconds(1.0))
    st.commit_step()  # third pass sees committed text as initial_prompt
    assert model.prompts[0] is None
    assert model.prompts[1] is None
    assert model.prompts[2] == "hello world"


def test_model_loaded_lazily_on_first_transcribe():
    built: list[int] = []

    class LazyModel:
        def transcribe(self, *args, **kwargs):
            return iter([FakeSegment(words=[])]), None

    def factory():
        built.append(1)
        return LazyModel()

    st = streaming.StreamingTranscriber(factory, min_chunk_seconds=1.0)
    st.feed(_seconds(0.3))
    st.commit_step()  # too short — model not touched
    assert built == []
    st.feed(_seconds(1.0))
    st.commit_step()
    assert built == [1]
