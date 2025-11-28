# ============================================================
# 6️⃣ True Perplexity Reference (BLOOM-560M)
# ============================================================
def compute_true_perplexity(model, tokenizer, texts, max_length=128):
    model.eval(); total_loss, count = 0.0, 0
    # Échantillon pour la rapidité
    for txt in texts[:50]:
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        if np.isfinite(loss) and loss < 100:
            total_loss += loss; count += 1
    avg_loss = total_loss / max(count,1)
    return math.exp(min(avg_loss, 100))

print("🧮 Computing Reference Perplexity...")
causal_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m").to(device)
causal_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
baseline_ppl = compute_true_perplexity(causal_model, causal_tokenizer, texts)

# Perplexité Sémantique (Approximation via Entropie d'Énergie)
entropy_approx = -np.mean(np.log(np.clip(Es + 1e-9, 1e-9, 1)))
structured_ppl = np.exp(entropy_approx)

print(f"✅ Baseline PPL: {baseline_ppl:.2f}")
print(f"✅ Structured PPL: {structured_ppl:.2f}")