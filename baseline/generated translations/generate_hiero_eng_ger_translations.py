import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Load model
tokenizer = AutoTokenizer.from_pretrained("mattiadc/hiero-transformer")
model = AutoModelForSeq2SeqLM.from_pretrained("mattiadc/hiero-transformer").eval()

lang_to_id = {
    'ea': 'ar',
    'tnt': 'lo',
    'en': 'en',
    'de': 'de'
}

def translate_hiero(input_text, src_lang, tgt_lang):
    tokenizer.src_lang = lang_to_id[src_lang]
    tokenizer.tgt_lang = lang_to_id[tgt_lang]
    inputs = tokenizer([input_text], return_tensors="pt")
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            num_beams=10,
            max_length=32,
            forced_bos_token_id=tokenizer.get_lang_id(lang_to_id[tgt_lang])
        )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# === Load test data
with open("../data/test_data.json", "r") as f:
    test_data = json.load(f)

# === Filter data by language and remove empty references
all_data = [item for item in test_data
            if item["metadata"]["target_lang"] in ["en", "de"]
            and item["target"] and item["target"].strip() != ""]

print(f"Total examples with 'en' or 'de' and non-empty reference: {len(all_data)}")

# Limit to 162 samples
all_data = all_data[:162]

rows = []
for item in tqdm(all_data, desc="Translating"):
    input_text = item["transliteration"]
    ref_text = item["target"]
    src_lang = item["metadata"]["source_lang"]
    tgt_lang = item["metadata"]["target_lang"]

    generated_translation = translate_hiero(input_text, src_lang, tgt_lang)

    rows.append({
        "source": item["source"],
        "transliteration": input_text,
        "reference_translation": ref_text,
        "generated_translation": generated_translation,
        "source_lang": src_lang,
        "target_lang": tgt_lang
    })

# === Save to CSV
df = pd.DataFrame(rows)
df.to_csv("generated_translations_hiero_baseline.csv", index=False)
print("✅ Saved to generated_translations_hiero_baseline.csv")
