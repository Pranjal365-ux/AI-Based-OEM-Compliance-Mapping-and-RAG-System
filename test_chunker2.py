import re
models = sorted(['FG-7081F', 'FG-7081F-DC'], key=len, reverse=True)
h_clean = "FG-7081F / FG-7081F-DC"
matched = set()
for known_model in models:
    if re.search(rf"\b{re.escape(known_model)}\b", h_clean, re.IGNORECASE):
        # To prevent smaller models from matching inside larger models,
        # we can replace the matched text with spaces before continuing, 
        # or just let it match both if they are both present!
        pass

# Actually, if h_clean is "FG-7081F-DC", \bFG-7081F\b will NOT match!
# Because after F there is a hyphen (which is \W). Wait!
print(bool(re.search(r"\bFG-7081F\b", "FG-7081F-DC")))
