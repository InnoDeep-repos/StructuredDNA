# ============================================================
# 5️⃣ Baseline: Switch Transformer (Dense Routing Cost)
# ============================================================
moe_name = "google/switch-base-8"
print(f"🤖 Loading Baseline: {moe_name}...")
moe_tokenizer = AutoTokenizer.from_pretrained(moe_name)
moe_model = AutoModelForSeq2SeqLM.from_pretrained(moe_name).to(device)

print("⚡ Running Dense Inference (Baseline)...")
token_counts_moe = 0; all_embs = []
start = time.time(); p_init = gpu_power()

# Batch processing simulation
for t in texts:
    inputs = moe_tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(device)
    token_counts_moe += inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = moe_model.encoder(inputs["input_ids"])
        # Mean pooling pour simuler une représentation comparable
        h = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embs.append(h)

p_end = gpu_power()
moe_time = time.time() - start
moe_power_avg = (p_init + p_end)/2

# Calcul SSI Baseline
moe_emb = np.vstack(all_embs)
Es_moe = 1 - cosine_similarity(moe_emb, moe_emb)
kmeans_moe = KMeans(n_clusters=k, random_state=42).fit(Es_moe)
SSI_moe = 1 - np.mean([np.var(Es_moe[np.ix_(kmeans_moe.labels_==i, kmeans_moe.labels_==i)]) for i in range(k) if np.sum(kmeans_moe.labels_==i)>1])
energy_density_moe = np.mean([np.mean(Es_moe[np.ix_(kmeans_moe.labels_==i, kmeans_moe.labels_==i)]) for i in range(k) if np.sum(kmeans_moe.labels_==i)>1])

print(f"✅ Switch-MoE SSI: {SSI_moe:.4f}")
print(f"⏱️ Baseline Time: {moe_time:.4f}s")