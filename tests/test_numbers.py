"""Digit-to-words expansion for the TTS input: what gets expanded, what deliberately
does not, and why. No model, no audio — the rules are the whole subject here."""

from __future__ import annotations

import pytest

from ytdub.stages import numbers
from ytdub.stages.numbers import (
    SUPPORTED,
    Change,
    expand_text,
    expansion_pair,
    protected_spans,
)


def spoken(text: str, lang: str = "pl", protected=None) -> str:
    return expand_text(text, lang, protected=protected)[0]


# --- the cases that occur in real content -----------------------------------------


@pytest.mark.parametrize(("text", "expected"), [
    # Ratings and decimals, in the target language's own separator style.
    ("Daję mu 4,5", "Daję mu cztery przecinek pięć"),
    ("Daję mu 4.5", "Daję mu cztery przecinek pięć"),
    ("prawdopodobnie dziś daję mu 7,8", "prawdopodobnie dziś daję mu siedem przecinek osiem"),
    # "9 out of 10" is two plain integers, and both are spoken.
    ("Ocena 9 na 10", "Ocena dziewięć na dziesięć"),
    # "około" and "z" take the genitive in Polish; both are handled (see below) and are
    # here as the reminder that the plain cardinal is not always the right form.
    ("W sumie, daję mu 9", "W sumie, daję mu dziewięć"),
    ("mam 10 lat", "mam dziesięć lat"),
    ("mam 22 filmy", "mam dwadzieścia dwa filmy"),
])
def test_ratings_decimals_and_integers_are_spoken(text, expected):
    assert spoken(text) == expected


def test_decimal_separator_follows_the_language_not_the_source():
    # Polish writes 7,8; German 7,8; French 7,8; English 7.8. All are read as the
    # language's own decimal word, which is num2words' job, not ours.
    assert "przecinek" in spoken("ocena 7,8", "pl")
    assert "Komma" in spoken("Note 7,8", "de")
    assert "virgule" in spoken("note 7,8", "fr")
    assert "point" in spoken("score 7.8", "en")


def test_a_year_with_a_date_cue_is_spoken_as_a_year():
    # The cue can come before ("in 2019") or after ("2019 roku"), which is where Polish
    # puts it.
    assert spoken("W 2026 roku zrobiłem 3 filmy") \
        == "W dwa tysiące dwadzieścia sześć roku zrobiłem trzy filmy"
    assert spoken("zimą 2019 roku", "pl") == "zimą dwa tysiące dziewiętnaście roku"
    assert spoken("in 2019 he left", "en") == "in twenty nineteen he left"
    # German and Dutch say nineteen-hundred rather than one-thousand-nine-hundred, and
    # num2words' year form is what gets that right.
    assert spoken("im Jahr 1985", "de") == "im Jahr neunzehnhundertfünfundachtzig"
    assert "negentien" in spoken("in 1985", "nl")


def test_a_bare_four_digit_quantity_is_not_read_as_a_year():
    # No date cue, so it is a quantity: the plain cardinal, never the year form.
    out, changes = expand_text("mam 2000 punktów", "pl")
    assert out == "mam dwa tysiące punktów"
    assert [c.kind for c in changes] == ["integer"]
    # Outside the plausible-year range the question does not arise.
    assert spoken("mam 2160 punktów") == "mam dwa tysiące sto sześćdziesiąt punktów"
    assert expand_text("mam 2160 punktów", "pl")[1][0].kind == "integer"


# --- what must be left alone -------------------------------------------------------


@pytest.mark.parametrize(("text", "lang"), [
    ("wersja 4.5.1 wyszła dziś", "pl"),          # version
    ("192.168.0.1 to adres", "pl"),              # IP
    ("spotkajmy się o 7:30", "pl"),              # clock time
    ("data to 10.06.2024", "pl"),                # date
    ("mam 12GB RAM", "pl"),                      # unit glued on
    ("karta RTX 3080 jest szybka", "pl"),        # model number
    ("karta PS 5 jest szybka", "pl"),            # model number, two capitals then space
    ("to jest 3-4 razy lepsze", "pl"),           # range
    ("chcę 100% pewności", "pl"),                # percentage
    ("mam 1080p monitor", "pl"),                 # resolution
    ("to GeForce RTX 4090", "pl"),
])
def test_codes_versions_times_and_ranges_stay_as_digits(text, lang):
    assert spoken(text, lang) == text


def test_a_brand_with_a_space_is_read_as_a_word_not_kept():
    # Documented limit of the ALL-CAPS product rule: "PlayStation 5" is not all caps, so
    # the 5 is spoken. It is read as a number either way, which is the point; add it to
    # hints.txt (or the glossary) if it must stay a digit.
    assert spoken("The PlayStation 5 is here", "en") == "The PlayStation five is here"
    assert spoken("The PlayStation 5 is here", "en", protected=["PlayStation 5"]) \
        == "The PlayStation 5 is here"


def test_a_protected_term_beats_the_genitive_rule():
    # Prepositions come before protection: what must never happen is a protected term
    # being rewritten, and what must also not happen is a preposition left with a digit
    # it should have inflected. Protection wins, because the term is a name.
    assert spoken("z 10 Rec Room Tokens", "pl", protected=["10 Rec Room Tokens"]) \
        == "z 10 Rec Room Tokens"


def test_a_number_inside_a_protected_term_is_not_touched():
    # The digit belongs to the name: "PlayStation 5" is one token, and 5 must stay.
    # "od" is a genitive trigger, so the 5 outside the term is inflected and the 5
    # inside it is not: both halves of the rule in one line.
    assert spoken("Gram w PlayStation 5 od 5 lat", "pl", protected=["PlayStation 5"]) \
        == "Gram w PlayStation 5 od pięciu lat"
    # And a protected term that contains the digits keeps them wherever it appears.
    assert spoken("Mam Rec Room Tokens 2 i 3 gry", "pl",
                  protected=["Rec Room Tokens 2"]) \
        == "Mam Rec Room Tokens 2 i trzy gry"
    spans = protected_spans("Rec Room Tokens 2 to nazwa", ["Rec Room Tokens 2"])
    assert spans and spans[0] == (0, 17)
    # A number *next to* a protected term rather than inside it is still a quantity.
    assert spoken("Mam 500 Rec Room Tokens", "pl", protected=["Rec Room Tokens"]) \
        == "Mam pięćset Rec Room Tokens"


def test_a_single_letter_after_a_number_is_a_word_not_a_unit():
    # "9 à 10" is two numbers and a preposition: the glued-letter rule must not eat it.
    assert spoken("9 à 10 en français", "fr") == "neuf à dix en français"
    assert spoken("Ocena 9 na 10") == "Ocena dziewięć na dziesięć"
    assert spoken("9 people came", "en") == "nine people came"


def test_text_without_digits_is_returned_unchanged():
    for text in ("", "bez liczb", "Dobrze, dobrze"):
        assert expand_text(text, "pl") == (text, [])


# --- language coverage -------------------------------------------------------------


def test_unsupported_language_keeps_its_digits():
    # Hindi has no num2words converter at all (NotImplementedError on every call), and
    # Hindi is a default target language: the honest outcome is to leave the digits and
    # report it, not to guess at Devanagari numerals.
    assert "hi" not in SUPPORTED
    assert expansion_pair("7", "hi") is None
    assert spoken("मुझे 5 फिल्में पसंद हैं", "hi") == "मुझे 5 फिल्में पसंद हैं"


def test_every_shipped_target_language_is_known_one_way_or_the_other():
    from ytdub.config import DEFAULT_LANGUAGES

    for lang in DEFAULT_LANGUAGES:
        if lang in SUPPORTED:
            assert expansion_pair("7", lang)
        else:
            assert expansion_pair("7", lang) is None  # reported, not silently mangled


# --- mechanics --------------------------------------------------------------------


def test_changes_report_what_moved_so_a_wrong_number_is_diagnosable():
    out, changes = expand_text("Daję mu 4,5 i 9", "pl")
    assert out == "Daję mu cztery przecinek pięć i dziewięć"
    assert [(c.kind, c.digits, c.words) for c in changes] == [
        ("decimal", "4,5", "cztery przecinek pięć"), ("integer", "9", "dziewięć")]
    # Spans point at the original text, which is what makes the debug log useful.
    for change in changes:
        assert "Daję mu 4,5 i 9"[change.start:change.end] == change.digits


def test_expansion_pair_refuses_what_it_cannot_do_well():
    assert expansion_pair("5", "pl") == "pięć"
    assert expansion_pair("7,8", "pl") == "siedem przecinek osiem"
    assert expansion_pair("5", "xx") is None
    assert expansion_pair("not-a-number", "pl") is None


def test_the_expander_never_reaches_a_model():
    # The whole point of doing this deterministically: no prompt, no request, and no
    # way for the answer to vary between runs.
    from pathlib import Path

    source = Path(numbers.__file__).read_text(encoding="utf-8")
    assert "urllib" not in source and "ollama" not in source.lower()
    assert isinstance(Change(kind="integer", digits="1", words="one", start=0, end=1), Change)


# --- case: Polish needs the genitive after a preposition ---------------------------


def test_polish_uses_the_genitive_after_a_preposition():
    # The case this was written for: "10 brown leaves" is a rating out of ten, and the
    # Polish is "z dziesięciu", not "z dziesięć". num2words only produces the nominative,
    # so it cannot be delegated — and a confidently wrong number is worse than a digit.
    assert spoken("z 10 brązowych liści") == "z dziesięciu brązowych liści"
    assert spoken("daję mu około 9") == "daję mu około dziewięciu"
    assert spoken("z 12 mil") == "z dwunastu mil"
    assert spoken("z 250 osób") == "z dwustu pięćdziesięciu osób"


def test_a_number_outside_the_genitive_table_stays_a_digit():
    # Above 999 the compound forms are not worth guessing, so it is left as written
    # rather than said wrongly.
    assert spoken("z 1000 osób") == "z 1000 osób"


def test_the_genitive_is_only_applied_after_those_prepositions():
    # Everywhere else the nominative is correct and is what num2words gives.
    assert spoken("mam 10 lat") == "mam dziesięć lat"
    assert spoken("Ocena 9 na 10") == "Ocena dziewięć na dziesięć"


def test_other_languages_need_no_case_handling():
    # German, French, Spanish and Dutch use the same cardinal form in these positions, so
    # num2words' output is already right and no table is needed.
    assert spoken("aus 10 Blättern", "de") == "aus zehn Blättern"
    assert spoken("sur 10 feuilles", "fr") == "sur dix feuilles"
    assert spoken("de 10 hojas", "es") == "de diez hojas"
    assert spoken("van 10 bladeren", "nl") == "van tien bladeren"


def test_a_class_or_model_number_is_a_name_not_a_quantity():
    # "Class 08" is a locomotive class. Spelling it out gives "Class huit" / "klasy
    # osiem", which is not something anyone says — and the translation inflects the word,
    # so the check has to be case-insensitive and cover the inflected forms.
    assert spoken("D'une locomotive Class 08 inutilisée", "fr") == \
        "D'une locomotive Class 08 inutilisée"
    assert spoken("Od nieczynnej lokomotywy klasy 08", "pl") == \
        "Od nieczynnej lokomotywy klasy 08"
    assert spoken("a rebuilt class 08 locomotive", "en") == \
        "a rebuilt class 08 locomotive"
    assert spoken("Class 66 diesel", "en") == "Class 66 diesel"
    assert spoken("Typ 4 Wagen", "de") == "Typ 4 Wagen"
    assert spoken("Route 66", "en") == "Route 66"
    # ...while an ordinary number in the same sentence is still spoken.
    assert spoken("Class 08 does 9 miles an hour", "en") == \
        "Class 08 does nine miles an hour"
