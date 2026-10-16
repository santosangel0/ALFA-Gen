# Teste de Análise Multivariada Completa
suppressPackageStartupMessages({
  library(dplyr)
  library(MASS)
})

df_cv <- read.csv("dados/features_canonicidade.csv", stringsAsFactors = FALSE)
df_pt <- read.csv("dados/features_pontuacao_formula.csv", stringsAsFactors = FALSE)
df_asl <- read.csv("dados/features_asl.csv", stringsAsFactors = FALSE)
df_sc <- read.csv("dados/features_subordinativas.csv", stringsAsFactors = FALSE)

# Merge seguro por caminho_relativo
df_m <- merge(df_cv[, c("caminho_relativo", "id_arquivo", "nivel_ordem", "prop_cv")],
              df_pt[, c("caminho_relativo", "pont_exp_prop", "formula_prop")], by = "caminho_relativo")
df_m <- merge(df_m, df_asl[, c("caminho_relativo", "asl")], by = "caminho_relativo")
df_m <- merge(df_m, df_sc[, c("caminho_relativo", "sconj_densidade")], by = "caminho_relativo")

ordem_niveis <- c("Ensino Fundamental I", "Ensino Fundamental II", "Ensino Médio", "Ensino Superior")
df_m$nivel_completo <- factor(ordem_niveis[df_m$nivel_ordem], levels = ordem_niveis)

cat("Tamanho verificado:", nrow(df_m), "x", ncol(df_m), "\n")
cat("Distribuição por nível:\n")
print(table(df_m$nivel_completo))

features <- c("prop_cv", "pont_exp_prop", "formula_prop", "asl", "sconj_densidade")
X <- df_m[, features]

# 1. Correlações
cat("\n=== Correlações de Pearson ===\n")
cor_p <- cor(X)
print(round(cor_p, 4))

cat("\n=== Correlações de Spearman ===\n")
cor_s <- cor(X, method = "spearman")
print(round(cor_s, 4))

# Multicolinearidade: VIF via lm dummy
vifs <- sapply(features, function(f) {
  form <- as.formula(paste(f, "~ ."))
  fit <- lm(form, data = X)
  1 / (1 - summary(fit)$r.squared)
})
cat("\n=== VIF (Variance Inflation Factors) ===\n")
print(round(vifs, 3))

# 2. MANOVA
cat("\n=== MANOVA ===\n")
fit_manova <- manova(as.matrix(X) ~ nivel_completo, data = df_m)
print(summary(fit_manova, test = "Pillai"))
print(summary(fit_manova, test = "Wilks"))
print(summary(fit_manova, test = "Hotelling-Lawley"))
print(summary(fit_manova, test = "Roy"))

# 3. PCA
cat("\n=== PCA (scale = TRUE) ===\n")
pca <- prcomp(X, scale. = TRUE)
print(summary(pca))
cat("\nLoadings PCA:\n")
print(round(pca$rotation, 4))

# 4. LDA
cat("\n=== LDA ===\n")
lda_fit <- lda(nivel_completo ~ prop_cv + pont_exp_prop + formula_prop + asl + sconj_densidade, data = df_m)
print(lda_fit)

pred_lda <- predict(lda_fit)
tab_conf <- table(Real = df_m$nivel_completo, Predito = pred_lda$class)
cat("\nMatriz de Confusão LDA:\n")
print(tab_conf)
acuracia <- sum(diag(tab_conf)) / sum(tab_conf)
cat("Acurácia Linear Prévia:", round(acuracia * 100, 2), "%\n")

# 5. Distâncias de Mahalanobis entre centróides
cat("\n=== Distâncias de Mahalanobis entre Centróides ===\n")
# Matriz de covariância comum intraclasse (pooled covariance)
residuos <- residuals(fit_manova)
cov_pooled <- cov(residuos)

centroides <- aggregate(as.matrix(X) ~ nivel_completo, data = df_m, FUN = mean)
rownames(centroides) <- centroides$nivel_completo
mu_mat <- as.matrix(centroides[, features])

dist_mah <- matrix(0, nrow = 4, ncol = 4, dimnames = list(ordem_niveis, ordem_niveis))
for (i in 1:4) {
  for (j in 1:4) {
    diff_ij <- mu_mat[i, ] - mu_mat[j, ]
    dist_mah[i, j] <- sqrt(t(diff_ij) %*% solve(cov_pooled) %*% diff_ij)
  }
}
print(round(dist_mah, 3))
