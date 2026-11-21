import re
import unicodedata
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

CURRENCY = r"(?:₹|\binr\b|\brs\.|\brs(?=\s*\d)|\brupees\b)"
AMOUNT = r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
VALUE = (
    rf"(?P<prefix>{CURRENCY})?\s*(?P<amount>{AMOUNT})"
    rf"(?![.,]\d|\w)\s*(?P<suffix>{CURRENCY})?"
)
START = (
    r"\b(?P<rule>under|below|less than|up to|at most|within|"
    r"no more than|not more than|maximum(?: price| budget)?|"
    r"(?:my |a )?budget(?: of| is| is up to)?)\b"
)
BEFORE = re.compile(START + rf"\s*[:=]?\s*{VALUE}", re.I)
AFTER = re.compile(VALUE + r"\s+or\s+(?P<rule>less|cheaper)\b", re.I)
FOREIGN = re.compile(r"[$€£¥]|\b(?:usd|eur|gbp|cad|aud|dollars?)\b", re.I)
CURRENCY_MARKER = re.compile(CURRENCY, re.I)
BUDGET_MARKER = re.compile(r"\b(?:budget|price cap|affordable under)\b", re.I)


GROUPED = re.compile(r"\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3}")


def maximum_price(query):
    text = unicodedata.normalize("NFKC", query)
    if FOREIGN.search(text):
        raise ValueError("Use an INR budget. Currency conversion is unavailable.")
    ceilings = []
    recognized = []
    for pattern in (BEFORE, AFTER):
        for match in pattern.finditer(text):
            currency = match["prefix"] or match["suffix"]
            rule = match["rule"].casefold()
            if not currency and "budget" not in rule and "price" not in rule:
                if text[match.end():].strip(" \t.?!,"):
                    continue
            raw = match["amount"]
            if "," in raw and not GROUPED.fullmatch(raw.split(".")[0]):
                raise ValueError("Invalid grouping in the INR amount.")
            amount = Decimal(raw.replace(",", ""))
            strict = rule in {"under", "below", "less than"}
            rounding = ROUND_CEILING if strict else ROUND_FLOOR
            ceilings.append(int(amount.to_integral_value(rounding)) - strict)
            recognized.append(match.span())
    if has_unrecognized_budget(text, recognized):
        raise ValueError("Specify a maximum such as 'up to INR 500'.")
    return min(ceilings) if ceilings else None


def has_unrecognized_budget(text, recognized):
    if not recognized and BUDGET_MARKER.search(text):
        return True
    return any(
        not any(start <= marker.start() < end for start, end in recognized)
        for marker in CURRENCY_MARKER.finditer(text)
    )

