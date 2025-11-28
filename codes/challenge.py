# ============================================================
# 8️⃣ Challenge 2: Architectural Scalability Analysis (K=1024)
# ============================================================
print("\n--- Analysing Challenge 2: Architectural Scalability (K=1024) ---")

# Paramètres
EMBEDDING_DIM = 768  # d
K_INDUSTRIAL = 1024  # K cible
BYTES_PER_FLOAT = 4  # FP32

# 1. Coût Statique VRAM
vram_cost_bytes = EMBEDDING_DIM * K_INDUSTRIAL * BYTES_PER_FLOAT
vram_cost_MB = vram_cost_bytes / (1024 * 1024)

print(f"1. Embedding Dimension (d): {EMBEDDING_DIM}")
print(f"2. Industrial Granularity (K): {K_INDUSTRIAL}")
print(f"3. VRAM Cost (MB/Layer, Static): {vram_cost_MB:.2f} MB")
print(f"\nConclusion:")
print(f" - Le coût VRAM est négligeable (~3MB).")
print(f" - L'avantage EUD (-97%+) vient de la densité de calcul (Distance vs GEMM) et non du stockage.")