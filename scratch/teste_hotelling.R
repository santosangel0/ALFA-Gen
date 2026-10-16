# Hotelling T2 test for Ensino Médio vs outros
suppressPackageStartupMessages({
  library(MASS)
})

# Hotelling T2 function
hotelling_t2 <- function(X1, X2) {
  n1 <- nrow(X1)
  n2 <- nrow(X2)
  p <- ncol(X1)
  
  m1 <- colMeans(X1)
  m2 <- colMeans(X2)
  d <- m1 - m2
  
  S1 <- cov(X1)
  S2 <- cov(X2)
  Sp <- ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
  
  T2 <- (n1 * n2 / (n1 + n2)) * t(d) %*% solve(Sp) %*% d
  T2 <- as.numeric(T2)
  
  # F approximation
  F_stat <- (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * T2
  df1 <- p
  df2 <- n1 + n2 - p - 1
  p_val <- 1 - pf(F_stat, df1, df2)
  
  c(T2 = T2, F_stat = F_stat, df1 = df1, df2 = df2, p_val = p_val)
}

df_m <- read.csv("dados/features_canonicidade.csv", stringsAsFactors = FALSE)
df_pt <- read.csv("dados/features_pontuacao_formula.csv", stringsAsFactors = FALSE)
df_asl <- read.csv("dados/features_asl.csv", stringsAsFactors = FALSE)
df_sc <- read.csv("dados/features_subordinativas.csv", stringsAsFactors = FALSE)

df_all <- merge(df_m[, c("caminho_relativo", "id_arquivo", "nivel_ordem", "prop_cv")],
                df_pt[, c("caminho_relativo", "pont_exp_prop", "formula_prop")], by = "caminho_relativo")
df_all <- merge(df_all, df_asl[, c("caminho_relativo", "asl")], by = "caminho_relativo")
df_all <- merge(df_all, df_sc[, c("caminho_relativo", "sconj_densidade")], by = "caminho_relativo")

ordem_niveis <- c("Ensino Fundamental I", "Ensino Fundamental II", "Ensino Médio", "Ensino Superior")
df_all$nivel_completo <- factor(ordem_niveis[df_all$nivel_ordem], levels = ordem_niveis)

features <- c("prop_cv", "pont_exp_prop", "formula_prop", "asl", "sconj_densidade")

X_em <- df_all[df_all$nivel_completo == "Ensino Médio", features]
for (lvl in c("Ensino Fundamental I", "Ensino Fundamental II", "Ensino Superior")) {
  X_other <- df_all[df_all$nivel_completo == lvl, features]
  res <- hotelling_t2(X_em, X_other)
  cat("\nEnsino Médio vs", lvl, ":\n")
  print(res)
}
