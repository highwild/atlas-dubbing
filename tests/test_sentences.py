"""Joining Whisper's half-sentences back into whole ones (stages/sentences.py)."""
from __future__ import annotations

from ytdub.models import Segment
from ytdub.stages.sentences import ends_sentence, is_continuation, join_fragments


def seg(i, start, end, text, speaker=None):
    return Segment(index=i, start=start, end=end, text=text, speaker=speaker, confidence=0.9)


def test_a_pause_split_half_sentence_is_joined():
    """The whole reason this exists: the splitter flushes on a pause over 0.6s, so a
    sentence with a breath in the middle arrives in two pieces."""
    segs = [seg(0, 0.0, 3.0, "completely out of action, a fantastic depot and a"),
            seg(1, 3.1, 4.4, "test track.")]
    out = join_fragments(segs)
    assert len(out) == 1
    assert out[0].text == "completely out of action, a fantastic depot and a test track."
    assert (out[0].start, out[0].end) == (0.0, 4.4)
    assert out[0].sources == [0, 1] and out[0].index == 0


def test_a_lowercase_continuation_is_joined_even_after_a_full_stop():
    """Whisper capitalises sentence starts, so a lowercase start means mid-sentence even
    when the piece before it happens to end in punctuation."""
    segs = [seg(0, 0.0, 2.0, "the driving controls are the same."),
            seg(1, 2.1, 3.4, "the brakes are the same.")]
    out = join_fragments(segs)
    assert len(out) == 1
    assert out[0].text == "the driving controls are the same. the brakes are the same."


def test_two_complete_sentences_are_left_alone():
    segs = [seg(0, 0.0, 2.0, "This is a rebuilt class 08 locomotive."),
            seg(1, 2.1, 4.0, "It has been transformed.")]
    out = join_fragments(segs)
    assert [s.text for s in out] == ["This is a rebuilt class 08 locomotive.",
                                     "It has been transformed."]
    assert [s.index for s in out] == [0, 1]


def test_a_one_word_line_joins_even_when_it_ends_a_sentence():
    """`vacuum.` and `test track.` are whole words and whole stops, and are still not
    something to hand a voice-cloning synthesizer on their own."""
    segs = [seg(0, 0.0, 2.0, "we can operate in air or"), seg(1, 2.0, 2.4, "vacuum.")]
    out = join_fragments(segs)
    assert len(out) == 1 and out[0].text == "we can operate in air or vacuum."

    # And the other direction: a short line followed by a long one.
    segs = [seg(0, 0.0, 0.4, "vacuum."), seg(1, 0.5, 3.0, "The main changes are the alarms.")]
    assert len(join_fragments(segs)) == 1


def test_never_joined_across_speakers():
    """A join moves words into one voice. Only the same speaker's lines may be joined."""
    segs = [seg(0, 0.0, 2.0, "and the different alarms that can go off which we need",
                speaker="SPK0"),
            seg(1, 2.1, 3.0, "to react to.", speaker="SPK1")]
    out = join_fragments(segs)
    assert [s.text for s in out] == ["and the different alarms that can go off which we need",
                                     "to react to."]


def test_a_long_silence_is_not_bridged():
    segs = [seg(0, 0.0, 2.0, "the idea being we've installed a power supply so"),
            seg(1, 9.0, 10.0, "we can plug it in.")]
    assert len(join_fragments(segs, max_gap=2.0)) == 2


def test_a_monologue_with_no_full_stops_is_still_cut_into_lines():
    """Caps keep a paragraph with no sentence ends from becoming one enormous cue."""
    segs = [seg(i, i * 3.0, i * 3.0 + 2.9, "and then something else entirely happened")
            for i in range(10)]
    out = join_fragments(segs, max_seconds=8.0)
    assert len(out) >= 4
    assert all(s.end - s.start <= 8.0 for s in out)
    assert all(sum(len(x) for x in [s.text]) <= 300 for s in out)
    # Nothing lost: every source line is accounted for exactly once.
    assert sorted(i for s in out for i in s.sources) == list(range(10))


def test_a_single_voice_recording_is_joined_too():
    """No diarization means no labels, and the fragments are just as bad."""
    segs = [seg(0, 0.0, 2.0, "so let's find out what the Hydroshunter is and"),
            seg(1, 2.1, 3.0, "how it works.")]
    out = join_fragments(segs)
    assert len(out) == 1 and out[0].text.endswith("how it works.")


def test_empty_input_is_not_a_crash():
    assert join_fragments([]) == []


def test_ends_sentence_ignores_trailing_quotes_and_brackets():
    assert ends_sentence('he said "yes."')
    assert not ends_sentence("(and that was that)")
    assert ends_sentence("(and that was that.)")
    assert not ends_sentence("and a")
    assert ends_sentence("क्या तुम ठीक हो?")


def test_is_continuation_reads_both_signals():
    long_prev = seg(0, 0.0, 2.0, "we can operate in air or")
    full_prev = seg(0, 0.0, 2.0, "we can operate.")
    lower = seg(1, 2.1, 3.0, "and then the brakes")
    upper = seg(1, 2.1, 3.0, "And then the brakes")
    assert is_continuation(long_prev, lower)
    assert is_continuation(long_prev, upper)
    assert is_continuation(full_prev, lower)
    assert not is_continuation(full_prev, upper)
