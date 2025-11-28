# ============================================================
# 7️⃣ Final Metrics & EUD Calculation
# ============================================================

# EUD = (Power * Time) / Tokens
EUD_moe = (moe_power_avg * moe_time) / token_counts_moe
EUD_sdna = (sdna_power_avg * sdna_time) / sdna_token_counts

# Deltas
Δ_ppl = (structured_ppl - baseline_ppl) / baseline_ppl * 100
Δ_energy = (energy_density_norm - energy_density_moe) / energy_density_moe * 100
Δ_SSI = (SSI_sdna - SSI_moe) / SSI_moe * 100
Δ_time = (sdna_time - moe_time) / moe_time * 100
Δ_power = (sdna_power_avg - moe_power_avg) / moe_power_avg * 100
Δ_eud = (EUD_sdna - EUD_moe) / EUD_moe * 100

results = pd.DataFrame({
    "Metric": ["Perplexity ↓","Energy Density ↓","SSI ↑","Inference Time (Batch) ↓","GPU Power (W) ↓","EUD (J/token) ↓"],
    "Switch-MoE": [round(baseline_ppl,2), round(energy_density_moe,3), f"{SSI_moe:.3f}",
                   round(moe_time,3), round(moe_power_avg,2), f"{EUD_moe:.6f}"],
    "StructuredDNA": [round(structured_ppl,2), round(energy_density_norm,3), f"{SSI_sdna:.3f}",
                      round(sdna_time,3), round(sdna_power_avg,2), f"{EUD_sdna:.6f}"],
    "Δ (%)": [f"{Δ_ppl:.1f}%",f"{Δ_energy:.1f}%",f"{Δ_SSI:.1f}%",f"{Δ_time:.1f}%",f"{Δ_power:.1f}%",f"{Δ_eud:.1f}%"]
})

print("\n🏆 FINAL VALIDATION RESULTS 🏆")
display(results)