models = ['FG-7081F', 'FG-7081F-DC']
h_clean = "FG-7081F-DC"
matched = []
for known_model in models:
    if known_model.lower() in h_clean.lower():
        matched.append(known_model)
print("Matched:", matched)
