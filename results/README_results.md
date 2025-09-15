# Protein corona results

Best model: **XGBRanker**, NDCG@5=0.637.
Top global features: Blood concentration - Conc. blood MS [pg/L], B9A064 Immunoglobulin lambda-like polypeptide 5免疫球蛋白 lambda样*RNA mouse brain regional specificity score, Dispersion medium pH*Blood concentration - Conc. blood IM [pg/L], Interactions, Incubation culture_ mouse plasma

MoE vs Global: 
  Global NDCG@5=nan; MoE NDCG@5=0.366.

Group-level trends:
- Albumin: R2_mean=-3.000, Spearman_mean=0.278
- Apolipoprotein: R2_mean=-0.330, Spearman_mean=0.240
- Complement: R2_mean=-0.223, Spearman_mean=0.189
- Other: R2_mean=-42829963.726, Spearman_mean=0.230