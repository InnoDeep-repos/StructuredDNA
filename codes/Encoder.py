# ============================================================
# 3️⃣ StructuredDNA: Bio-Physical Encoding
# ============================================================
print("🔬 Loading Bio_ClinicalBERT model...")
word_model = models.Transformer("emilyalsentzer/Bio_ClinicalBERT", max_seq_length=256)
pooling = models.Pooling(word_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_model, pooling], device=device)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Comptage précis des tokens pour le calcul EUD
sdna_token_counts = 0
for t in texts:
    inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=256)
    sdna_token_counts += inputs["input_ids"].shape[1]
print(f"📊 Total Token Count: {sdna_token_counts}")

# Encodage (C'est ici qu'on mesure la puissance SDNA 'Batch')
print("⚡ Encoding texts into Semantic Energy Space...")
start = time.time(); p1 = gpu_power()
emb = model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
p2 = gpu_power(); sdna_time = time.time() - start
sdna_power_avg = (p1 + p2)/2

# Matrice d'Énergie
Es = 1 - cosine_similarity(emb, emb)
print(f"✅ Encoding Complete. Time: {sdna_time:.4f}s")