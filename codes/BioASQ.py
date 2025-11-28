# ============================================================
# 2️⃣ Dataset Loading (BioASQ)
# ============================================================
print("📥 Loading BioASQ dataset...")
# On prend 400 pour le Challenge, ou 200 pour le test rapide
bioasq = load_dataset("jmhb/bioasq_factoid", split="test[:400]")
texts = []
for item in bioasq:
    for key in ["snippet", "body", "question", "context"]:
        if key in item and isinstance(item[key], str) and len(item[key]) > 30:
            texts.append(item[key]); break

print(f"✅ Loaded {len(texts)} biomedical texts")
print(f"🧠 Sample: {texts[0][:100]}...")