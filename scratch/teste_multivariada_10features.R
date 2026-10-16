# Teste de Análise Multivariada com as 10 Features
suppressPackageStartupMessages({
  library(dplyr)
  library(MASS)
})

source("R/funcoes_estatistica.R")

df <- read.csv("dados/dataset_multivariado_features.csv", stringsAsFactors = FALSE)
ordem_niveis <- c("Ensino Fundamental I", "Ensino Fundamental II", "Ensino Médio", "Ensino Superior")
df$nivel_completo <- factor(df$nivel_completo, levels = ordem_niveis)

features_10 <- c(
  "prop_cv",
  "pont_exp_prop",
  "formula_prop",
  "asl",
  "sconj_densidade",
  "razao_nominal",
  "cconj_densidade",
  "prop_polissilabos",
  "virgulas_por_sentenca",
  "metadiscurso_prop"
)

cat("Verificação do dataset:", nrow(df), "linhas x", length(features_10), "features.\n")

# 1. VIF
cat("\n=== VIF (10 Features) ===\n")
vifs <- calcular_vif(df, features_10)
print(round(vifs, 3))

# 2. MANOVA
cat("\n=== MANOVA (10 Features) ===\n")
manova_res <- calcular_manova(df, features_10, grupo = "nivel_completo")
print(manova_res$tabela)

# 3. PCA
cat("\n=== PCA (10 Features) ===\n")
pca_res <- calcular_pca(df, features_10, scale = TRUE)
print(pca_res$tabela_variancia)

cat("\nLoadings PC1 a PC3:\n")
print(round(pca_res$rotation[, 1:3], 3))

# 4. LDA
cat("\n=== LDA (10 Features) ===\n")
lda_res <- calcular_lda(df, features_10, grupo = "nivel_completo")
print(lda_res$prop_traco)
print(lda_res$confusao)
cat("Acurácia LDA Linear (10 Features):", round(lda_res$acuracia * 100, 2), "%\n")

# 5. Distâncias de Mahalanobis
cat("\n=== Distâncias de Mahalanobis (10 Features) ===\n")
dist_mah <- calcular_distancias_mahalanobis(df, features_10, grupo = "nivel_completo")
print(round(dist_mah, 3))

# 6. Hotelling T2
cat("\n=== Hotelling T2 (Ensino Médio vs. Outros em R^10) ===\n")
hot_res <- calcular_hotelling_ensino_medio(df, features_10, grupo = "nivel_completo")
print(hot_res)
