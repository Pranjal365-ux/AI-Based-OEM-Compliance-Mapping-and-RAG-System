import re

_MODEL_CODE_RE = re.compile(
    r"\b("
    r"F[GPIM]{1,3}-\d{3,5}[A-Z]{0,4}(?:-\d{1,2})?(?:-DC)?"
    r"|"
    r"PA-\d{3,5}[A-Z]{0,2}"
    r"|"
    r"(?:BIG-IP\s+)?[ir]\d{4,5}"
    r"|"
    r"[A-Z]{2,6}-\d{3,5}[A-Z]{0,3}"
    r")",
    re.VERBOSE
)

text = "We have FG-7081F and PA-5450 and BIG-IP i5800. Also FIM-7921F and FG-7121F-DC."

print("Matches:", _MODEL_CODE_RE.findall(text))
