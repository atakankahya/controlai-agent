"""Repair of PDF text extraction artifacts.

Some of the textbooks in the library are typeset with Type 1 symbol fonts whose
embedded ToUnicode map is wrong. Every extractor -- pypdf and PyMuPDF alike --
therefore returns the same mojibake for their inline mathematics: the Laplace
transform definition comes out as `L½ f ðtÞ/C138 ¼FðsÞ` instead of
`L[f(t)] = F(s)`.

This mattered more than it looks. 71.5% of the chunks from Nise's *Control
Systems Engineering* -- the most heavily used classical-control reference in the
corpus -- were damaged this way, which is why retrieval for a query as ordinary
as "Routh-Hurwitz table construction" returned book index pages instead of the
relevant section.

The substitution is deterministic and was read off from context, not guessed:
`e/C0ð sþaÞt` is `e^-(s+a)t`, `5:6 /C2 10/C0 6s` is `5.6 x 10^-6 s`,
`26:57/C14` is `26.57°`, `þ/C1/C1/C1þ` is `+ ... +`. Codes that turn out to be
pieces of a multi-line bracket rather than a character of their own carry no
meaning on their own and are dropped.
"""

from __future__ import annotations

import re

# Single characters the broken font map emits in place of ASCII math.
_CHAR_MAP = str.maketrans({
    "ð": "(",
    "Þ": ")",
    "¼": "=",
    "þ": "+",
    "½": "[",
})

# `/Cnn` glyph references, by the number that follows.
_GLYPH_MAP = {
    "0": "-",     # minus:  1899 /C0 3761z  ->  1899 - 3761z
    "1": "·",  # middle dot:  þ/C1/C1/C1þ  ->  + ... +
    "2": "×",  # times:  5:6 /C2 10  ->  5.6 x 10
    "12": "|",    # evaluation bar
    "14": "°",  # degree:  26:57/C14  ->  26.57 deg
    "15": "•",  # bullet in feature lists
    "138": "]",   # closing bracket, pairing with the "[" above
}
# Pieces of tall multi-line brackets. They are layout, not content, and the
# extractor emits them after the expression they were meant to enclose, so
# rendering them as brackets would be actively misleading.
_LAYOUT_GLYPHS = {"3", "6", "16", "17", "18", "19", "20", "21"}

_GLYPH_RE = re.compile(r"/C(\d+)")
# Detects whether a document needs any of this at all.
_DAMAGE_RE = re.compile(r"/C\d+|[ðÞ¼þ]")


def looks_damaged(text: str, threshold: int = 5) -> bool:
    """True if `text` carries enough artifacts to be worth repairing."""
    return len(_DAMAGE_RE.findall(text)) >= threshold


def repair(text: str) -> str:
    """Undo the broken font mapping. Safe to call on undamaged text."""
    if not _DAMAGE_RE.search(text):
        return text

    def _glyph(match: re.Match) -> str:
        code = match.group(1)
        if code in _GLYPH_MAP:
            return _GLYPH_MAP[code]
        if code in _LAYOUT_GLYPHS:
            return " "
        return " "  # an unrecognised glyph is noise; a space beats a token of junk

    text = _GLYPH_RE.sub(_glyph, text)
    text = text.translate(_CHAR_MAP)
    # The extractor also stamps a watermark onto every page of one scan.
    text = text.replace("Apago PDF Enhancer", " ")
    return re.sub(r"[ \t]{2,}", " ", text).strip()
