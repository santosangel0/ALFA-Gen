df_cv <- read.csv("dados/features_canonicidade.csv", stringsAsFactors = FALSE)
df_pt <- read.csv("dados/features_pontuacao_formula.csv", stringsAsFactors = FALSE)
df_asl <- read.csv("dados/features_asl.csv", stringsAsFactors = FALSE)
df_sc <- read.csv("dados/features_subordinativas.csv", stringsAsFactors = FALSE)

cat("Linhas totais:\n")
cat("cv:", nrow(df_cv), "pt:", nrow(df_pt), "asl:", nrow(df_asl), "sc:", nrow(df_sc), "\n")

cat("\nDuplicatas em id_arquivo:\n")
cat("cv:", sum(duplicated(df_cv$id_arquivo)), "\n")
cat("pt:", sum(duplicated(df_pt$id_arquivo)), "\n")
cat("asl:", sum(duplicated(df_asl$id_arquivo)), "\n")
cat("sc:", sum(duplicated(df_sc$id_arquivo)), "\n")

if (sum(duplicated(df_cv$id_arquivo)) > 0) {
  cat("\nExemplo de id_arquivo duplicado em df_cv:\n")
  print(head(df_cv$id_arquivo[duplicated(df_cv$id_arquivo)]))
  print(df_cv[df_cv$id_arquivo == df_cv$id_arquivo[duplicated(df_cv$id_arquivo)][1], c("id_arquivo", "caminho_relativo")])
}
