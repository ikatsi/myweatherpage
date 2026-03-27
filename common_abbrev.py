# -*- coding: utf-8 -*-
import re

ABBREV_RULES = [
    (r"\bΣτρατιωτική\s+Σχολή\s+Ευελπίδων\b", "Στρατ. Σχ. Ευελπίδων"),
    (r"\bΙερά\s+Μονή\b", "Ι.Μ."),
    (r"\bΚαταφύγιο\b", "Καταφ."),
    (r"\bΟροπέδιο\b", "Οροπ."),
    (r"\bΝομισματοκοπείο\b", "Νομισμ."),
    (r"\bΖωολογικό\b", "Ζωολ."),
    (r"\bΠυροσβεστικού Σώματος\b", "Πυροσβ. Σώμ."),
    (r"\bΔιεθνές Αεροδρόμιο\b", "Α/Δ"),
    (r"\bΑεροδρόμιο\b", "Α/Δ"),
    (r"\bΠανεπιστήμιο\b", "Παν."),
    (r"\bΝοσοκομείο\b", "Νοσ."),
    (r"\bΧιονοδρομικό\s+κέντρο\b", "Χ/Κ"),
    (r"\bΌρος\b", "Όρ."),
    (r"\bΆνω\b", "Ά."),
    (r"\bΚάτω\b", "Κ."),
    (r"\bΆγιος\b", "Άγ."),
    (r"\bΑγία\b", "Αγ."),
    (r"\bΆγιοι\b", "Αγ."),
    (r"\bΠαλαιός\b", "Παλ."),
    (r"\bΠαλαιά\b", "Παλ."),
    (r"\bΠαλαιό\b", "Παλ."),
    (r"\bΜεγάλος\b", "Μεγ."),
    (r"\bΜεγάλη\b", "Μεγ."),
    (r"\bΜεγάλο\b", "Μεγ."),
    (r"\bΜέγα\b", "Μεγ."),
    (r"\bΜεγάλα\b", "Μεγ."),
    (r"\bσταθμός\b", "στ."),
    (r"\bλόφος\b", "λόφ."),
]

_COMPILED = [(re.compile(pat, flags=re.IGNORECASE), repl) for pat, repl in ABBREV_RULES]

def prettify_station_name(s):
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return s

    s = s.replace("Μητροπολιτικό Πάρκο", "Πάρκο")

    for rx, repl in _COMPILED:
        s = rx.sub(repl, s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

def ellipsize(s, max_chars=42):
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars - 1].rstrip() + "…"

def shorten_for_box(name, max_chars=26):
    s = prettify_station_name(name)

    if "(" in s and ")" in s:
        base = s.split("(", 1)[0].strip()
        if base:
            s = base

    s = s.replace("«", "").replace("»", "").replace('"', "").replace("'", "")
    s = re.sub(r"\s+", " ", s).strip()

    return ellipsize(s, max_chars=max_chars)
