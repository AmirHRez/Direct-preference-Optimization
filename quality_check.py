import re
from config import SHORT_ANSWER_MAX_WORDS, SHORT_ANSWER_WORD_THRESHOLD

# Word-boundary markers. Split into three groups because \b does not behave
# correctly directly before an apostrophe (there's no word/non-word
# transition there), so apostrophe-led forms need their own pattern.
_WORD_MARKERS = re.compile(
    r"\b(thee|thou|thy|thine|doth|dost|hath|art|ye|unto|whilst|forsooth|verily|"
    r"wherefore|shalt|wouldst|canst|prithee|methinks|methought|nay|aye|quoth|"
    r"betwixt|oft|ere|hither|thither|yond|yonder|mayhap|hast|begotten|sooth|"
    r"be)\b",
    re.IGNORECASE,
)
_APOSTROPHE_MARKERS = re.compile(r"'tis\b|'twas\b|'twixt\b|o'er\b|e'er\b|ne'er\b", re.IGNORECASE)
_ETH_SUFFIX = re.compile(r"\b\w+eth\b", re.IGNORECASE)

# Words that often signal the model invented a flourish or swapped out a
# technical term for a vague archaic-sounding stand-in, rather than doing a
# straightforward style rewrite. Not a hard rule (some of these are fine in
# context) — just a trigger for a human to glance at the pair.
SUSPICIOUS_SUBSTITUTIONS = [
    "puissance", "wondrous", "mighty", "guise", "sundry", "strange",
    "marvel", "wondrously", "prodigious",
]


def archaic_marker_count(text: str) -> int:
    return (
        len(_WORD_MARKERS.findall(text))
        + len(_APOSTROPHE_MARKERS.findall(text))
        + len(_ETH_SUFFIX.findall(text))
    )


def length_ratio(chosen: str, rejected: str) -> float:
    r = len(rejected.split())
    return len(chosen.split()) / r if r > 0 else 999


def has_meaning_drift(chosen: str, rejected: str) -> bool:
    chosen_l, rejected_l = chosen.lower(), rejected.lower()
    return any(w in chosen_l and w not in rejected_l for w in SUSPICIOUS_SUBSTITUTIONS)


def compute_flags(chosen: str, rejected: str, max_len_ratio: float) -> list:
    """Returns a list of flag strings. Empty list = clean, no review needed."""
    flags = []

    if chosen.strip() == rejected.strip():
        flags.append("identical")

    if archaic_marker_count(chosen) == 0:
        flags.append("zero_archaic_marker")

    n_words_rejected = len(rejected.split())
    n_words_chosen = len(chosen.split())

    if n_words_rejected <= SHORT_ANSWER_WORD_THRESHOLD:
        if n_words_chosen > SHORT_ANSWER_MAX_WORDS:
            flags.append("short_answer_expanded")
    else:
        ratio = length_ratio(chosen, rejected)
        if ratio > max_len_ratio:
            flags.append("length_ratio_high")

    if has_meaning_drift(chosen, rejected):
        flags.append("possible_meaning_drift")

    return flags