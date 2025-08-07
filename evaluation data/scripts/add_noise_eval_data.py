import pandas as pd
import random

# ===== Global Parameters =====
input_file = "evaluation_data.csv"

noise_settings = {
    "10": 0.1,
    "30": 0.3,
    "50": 0.5,
    "100": 1
}

substitution_prob = 0.1  # probability of character confusion
deletion_prob = 0.02     # probability of deletion
swap_prob = 0.02         # probability of swapping characters

# ===== Dictionnaire de confusion pour texte copte =====
# confusion_map = {
#     'ⲁ': ['ⲟ', 'ⲉ'], 'ⲟ': ['ⲁ', 'ⲉ'], 'ⲉ': ['ⲁ', 'ⲟ'],
#     'ⲓ': ['ⲏ', 'ⲩ'], 'ⲏ': ['ⲓ', 'ⲉ'], 'ⲩ': ['ⲓ', 'ⲛ'],
#     'ⲥ': ['ⲓ', 'ⲏ'], 'ϣ': ['ϥ', 'ϫ'], 'ϥ': ['ϣ', 'ϫ'],
#     'ϫ': ['ϣ', 'ϥ'], 'ϯ': ['ⲧ', 'ⲑ'], 'ⲧ': ['ϯ', 'ⲑ'],
#     'ⲑ': ['ⲧ', 'ϯ'], 'ⲛ': ['ⲩ'], 'ⲙ': ['ⲛ'], 'ⲣ': ['ⲩ', 'ⲙ']
# }

# ===== Confusion dictionary for romanized text =====
confusion_map = {
    'a': ['o', 'e'], 'o': ['a', 'e'], 'e': ['a', 'o'],
    'i': ['l', 'j'], 'l': ['i', '1'], 'c': ['e'],
    'u': ['v'], 'v': ['u'], 'n': ['m'], 'm': ['n'],
    'r': ['n', 's'], 's': ['r'], 't': ['f'], 'f': ['t']
}

def add_substitution_noise(text, noise_level=0.1):
    new_text = ""
    for char in str(text):
        if char in confusion_map and random.random() < noise_level:
            new_text += random.choice(confusion_map[char])
        else:
            new_text += char
    return new_text

def add_typo_noise(text, deletion_prob=0.02, swap_prob=0.02):
    chars = list(text)
    i = 0
    while i < len(chars):
        if random.random() < deletion_prob:
            chars[i] = '[]'  # simulate deletion
            i += 1
            continue
        if i < len(chars) - 1 and random.random() < swap_prob:
            chars[i], chars[i+1] = chars[i+1], chars[i]
            i += 2
        else:
            i += 1
    return ''.join(chars)

def apply_noise_to_coptic_text(text):
    text = add_substitution_noise(text, substitution_prob)
    text = add_typo_noise(text, deletion_prob, swap_prob)
    return text

# ===== Load initial data =====
df_clean = pd.read_csv(input_file)

# ===== Generate noisy datasets =====
for label, noise_prob in noise_settings.items():
    df = df_clean.copy()
    new_coptic_texts = []
    for text in df['coptic_text_romanized']:
        if random.random() < noise_prob:
            new_coptic_texts.append(apply_noise_to_coptic_text(text))
        else:
            new_coptic_texts.append(text)
    df['coptic_text_romanized'] = new_coptic_texts
    output_file = f"evaluation_data_noisy_{label}.csv"
    df.to_csv(output_file, index=False)

    # Report
    nb_noisy = sum(1 for orig, noisy in zip(df_clean['coptic_text_romanized'], new_coptic_texts) if orig != noisy)
    print(f"=== {label}% noise completed ===")
    print(f"Total number of verses: {len(df)}")
    print(f"Number of modified verses: {nb_noisy}")
    print(f"File saved as: {output_file}")
    print("-----------------------------")
