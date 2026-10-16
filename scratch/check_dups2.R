df_cv <- read.csv("dados/features_canonicidade.csv", stringsAsFactors = FALSE)
cat("Duplicatas em caminho_relativo:", sum(duplicated(df_cv$caminho_relativo)), "\n")
cat("Total de linhas:", nrow(df_cv), "\n")
