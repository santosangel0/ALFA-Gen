# Diagnóstico Estatístico das 5 Novas Features
suppressPackageStartupMessages({
  library(dplyr)
})

source("R/funcoes_estatistica.R")

ordem_niveis <- c("Ensino Fundamental I", "Ensino Fundamental II", "Ensino Médio", "Ensino Superior")

features_info <- list(
  list(csv = "dados/features_razao_nominal.csv", var_prop = "razao_nominal", var_qtd = "substantivos_qtd", nome = "Razão Nominal"),
  list(csv = "dados/features_coordenativas.csv", var_prop = "cconj_densidade", var_qtd = "cconj_qtd", nome = "Conjunções Coordenativas"),
  list(csv = "dados/features_polissilabos.csv", var_prop = "prop_polissilabos", var_qtd = "polissilabos_qtd", nome = "Palavras Polissílabas"),
  list(csv = "dados/features_virgulas.csv", var_prop = "virgulas_por_sentenca", var_qtd = "virgulas_qtd", nome = "Vírgulas por Sentença"),
  list(csv = "dados/features_metadiscurso.csv", var_prop = "metadiscurso_prop", var_qtd = "metadiscurso_qtd", nome = "Metadiscurso Acadêmico")
)

for (f in features_info) {
  cat("\n================================================================================\n")
  cat("FEATURE:", f$nome, "(", f$var_prop, ")\n")
  cat("================================================================================\n")
  df <- read.csv(f$csv, stringsAsFactors = FALSE)
  df$nivel_completo <- factor(df$nivel_completo, levels = ordem_niveis)
  
  # Médias por nível
  medias <- df %>%
    group_by(nivel_completo) %>%
    summarise(
      N = n(),
      Media_Prop = mean(.data[[f$var_prop]], na.rm = TRUE),
      SD_Prop = sd(.data[[f$var_prop]], na.rm = TRUE),
      Media_Qtd = mean(.data[[f$var_qtd]], na.rm = TRUE),
      .groups = "drop"
    )
  print(as.data.frame(medias))
  
  # Kruskal-Wallis
  kw <- kruskal.test(df[[f$var_prop]] ~ df$nivel_completo)
  cat(sprintf("\nKruskal-Wallis: chi2 = %.2f, p = %.2e\n", kw$statistic, kw$p.value))
  
  # Comparação com Ensino Médio
  comp_em <- executar_comparacoes_pares_em(df, f$var_prop)
  print(as.data.frame(comp_em[, c("Comparação", "Média Grupo", "Média Médio", "p-adj (Bonferroni)", "Cohen d", "Decisão (alpha=0.05)")]))
}
