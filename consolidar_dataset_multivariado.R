# ================================================================================
# PROJETO ALFA-GEN: CONSOLIDAÇÃO DO DATASET MULTIVARIADO (15 FEATURES)
# ================================================================================
# Descrição: Unifica as 15 features linguísticas extraídas para os 2.076 textos
#            em um único arquivo tabular dados/dataset_multivariado_features.csv.
# ================================================================================

suppressPackageStartupMessages({
  library(dplyr)
})

cat("Carregando datasets individuais das 15 features...\n")
df_cv   <- read.csv("dados/features_canonicidade.csv", stringsAsFactors = FALSE)
df_pt   <- read.csv("dados/features_pontuacao_formula.csv", stringsAsFactors = FALSE)
df_asl  <- read.csv("dados/features_asl.csv", stringsAsFactors = FALSE)
df_sc   <- read.csv("dados/features_subordinativas.csv", stringsAsFactors = FALSE)
df_rn   <- read.csv("dados/features_razao_nominal.csv", stringsAsFactors = FALSE)
df_cc   <- read.csv("dados/features_coordenativas.csv", stringsAsFactors = FALSE)
df_pl   <- read.csv("dados/features_polissilabos.csv", stringsAsFactors = FALSE)
df_vg   <- read.csv("dados/features_virgulas.csv", stringsAsFactors = FALSE)
df_md   <- read.csv("dados/features_metadiscurso.csv", stringsAsFactors = FALSE)
df_ca   <- read.csv("dados/features_citacao_academica.csv", stringsAsFactors = FALSE)
df_nm   <- read.csv("dados/features_nominalizacao.csv", stringsAsFactors = FALSE)
df_cn   <- read.csv("dados/features_conectivos_argumentativos.csv", stringsAsFactors = FALSE)
df_pe   <- read.csv("dados/features_pont_estruturante.csv", stringsAsFactors = FALSE)
df_ps   <- read.csv("dados/features_pessoalidade.csv", stringsAsFactors = FALSE)

# Merge sequencial rigoroso por caminho_relativo (chave unívoca)
df_multi <- df_cv %>%
  dplyr::select(
    caminho_relativo,
    id_arquivo,
    nivel_ordem,
    total_palavras,
    palavras_cv,
    prop_cv
  ) %>%
  dplyr::inner_join(
    df_pt %>%
      dplyr::select(
        caminho_relativo,
        pont_exp_qtd,
        pont_exp_prop,
        formula_qtd,
        formula_prop
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_asl %>%
      dplyr::select(
        caminho_relativo,
        total_sentencas,
        asl
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_sc %>%
      dplyr::select(
        caminho_relativo,
        sconj_qtd,
        sconj_densidade
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_rn %>%
      dplyr::select(
        caminho_relativo,
        substantivos_qtd,
        verbos_qtd,
        razao_nominal
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_cc %>%
      dplyr::select(
        caminho_relativo,
        cconj_qtd,
        cconj_densidade,
        razao_cconj_sconj
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_pl %>%
      dplyr::select(
        caminho_relativo,
        polissilabos_qtd,
        prop_polissilabos,
        silabas_por_palavra
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_vg %>%
      dplyr::select(
        caminho_relativo,
        virgulas_qtd,
        virgulas_por_sentenca
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_md %>%
      dplyr::select(
        caminho_relativo,
        metadiscurso_qtd,
        metadiscurso_prop
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_ca %>%
      dplyr::select(
        caminho_relativo,
        citacao_academica_qtd,
        citacao_academica_prop,
        citacao_academica_por_mil
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_nm %>%
      dplyr::select(
        caminho_relativo,
        nominalizacao_qtd,
        nominalizacao_prop
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_cn %>%
      dplyr::select(
        caminho_relativo,
        conectivos_argumentativos_qtd,
        conectivos_argumentativos_prop,
        conectivos_argumentativos_por_mil
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_pe %>%
      dplyr::select(
        caminho_relativo,
        ponto_virgula_qtd,
        dois_pontos_qtd,
        parenteses_qtd,
        travessoes_qtd,
        pont_estruturante_qtd,
        pont_estruturante_prop,
        pont_estruturante_por_sentenca
      ),
    by = "caminho_relativo"
  ) %>%
  dplyr::inner_join(
    df_ps %>%
      dplyr::select(
        caminho_relativo,
        pessoalidade_qtd,
        pessoalidade_prop
      ),
    by = "caminho_relativo"
  )

# Definir níveis pedagógicos ordenados
ordem_niveis <- c(
  "Ensino Fundamental I",
  "Ensino Fundamental II",
  "Ensino Médio",
  "Ensino Superior"
)
df_multi$nivel_completo <- factor(ordem_niveis[df_multi$nivel_ordem], levels = ordem_niveis)

# Reordenar colunas
df_multi <- df_multi %>%
  dplyr::select(
    caminho_relativo,
    id_arquivo,
    nivel_ordem,
    nivel_completo,
    total_palavras,
    total_sentencas,
    palavras_cv,
    prop_cv,
    pont_exp_qtd,
    pont_exp_prop,
    formula_qtd,
    formula_prop,
    asl,
    sconj_qtd,
    sconj_densidade,
    substantivos_qtd,
    verbos_qtd,
    razao_nominal,
    cconj_qtd,
    cconj_densidade,
    razao_cconj_sconj,
    polissilabos_qtd,
    prop_polissilabos,
    silabas_por_palavra,
    virgulas_qtd,
    virgulas_por_sentenca,
    metadiscurso_qtd,
    metadiscurso_prop,
    citacao_academica_qtd,
    citacao_academica_prop,
    citacao_academica_por_mil,
    nominalizacao_qtd,
    nominalizacao_prop,
    conectivos_argumentativos_qtd,
    conectivos_argumentativos_prop,
    conectivos_argumentativos_por_mil,
    ponto_virgula_qtd,
    dois_pontos_qtd,
    parenteses_qtd,
    travessoes_qtd,
    pont_estruturante_qtd,
    pont_estruturante_prop,
    pont_estruturante_por_sentenca,
    pessoalidade_qtd,
    pessoalidade_prop
  )

# Verificação de integridade
stopifnot(nrow(df_multi) == 2076)
cat(sprintf("[SUCESSO] Dataset multivariado consolidado: %d observações e %d colunas.\n", 
            nrow(df_multi), ncol(df_multi)))

# Resumo por nível
cat("\nDistribuição por nível:\n")
print(table(df_multi$nivel_completo))

# Salvar CSV final
caminho_saida <- "dados/dataset_multivariado_features.csv"
write.csv(df_multi, caminho_saida, row.names = FALSE)
cat(sprintf("Arquivo salvo em: %s\n", caminho_saida))
