# ============================================================
# 4️⃣ Energy Minimization & Semantic Stability (SSI)
# ============================================================
k = 50
print(f"🧬 Forming {k} Semantic Codons (Experts)...")
kmeans = KMeans(n_clusters=k, random_state=42).fit(Es)
labels = kmeans.labels_

# Calcul SSI (1 - variance intra-cluster)
cluster_vars = [np.var(Es[np.ix_(labels==i, labels==i)]) for i in range(k) if np.sum(labels==i)>1]
SSI_sdna = 1 - np.mean(cluster_vars)

# Calcul Densité d'Énergie Moyenne
cluster_means = [np.mean(Es[np.ix_(labels==i, labels==i)]) for i in range(k) if np.sum(labels==i)>1]
energy_density_norm = np.mean(cluster_means)

print(f"✅ Optimization Complete.")
print(f"🔹 StructuredDNA SSI: {SSI_sdna:.4f}")
print(f"🔹 Energy Density:    {energy_density_norm:.4f}")