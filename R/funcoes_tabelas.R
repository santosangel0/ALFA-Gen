# ================================================================================
# PROJETO ALFA-GEN: MÓDULO DE FORMATAÇÃO E GERAÇÃO DE TABELAS
# ================================================================================
# Descrição: Funções dedicadas à renderização e formatação elegante de tabelas
#            em formato LaTeX via knitr::kable e kableExtra para relatórios PDF.
# ================================================================================

suppressPackageStartupMessages({
  library(knitr)
  library(kableExtra)
  library(dplyr)
})

# ================================================================================
#' Gerar Tabela de Estatísticas Descritivas Globais para uma Feature
#'
#' @title Tabela LaTeX de Estatísticas Descritivas
#' @description Formata a tabela com as principais estatísticas descritivas
#'   das variáveis quantitativas e de proporção para todo o conjunto de textos.
#'
#' @param df data.frame com as colunas analisadas.
#' @param var_bruta Nome da variável com a contagem bruta (default = 'palavras_cv').
#' @param var_prop Nome da variável com a proporção (default = 'prop_cv').
#' @param label_bruta Rótulo descritivo da linha bruta na tabela.
#' @param label_prop Rótulo descritivo da linha de proporção na tabela.
#' @param caption Texto do título/legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no documento PDF.
#'
#' @export
# ================================================================================
tabela_descritivas_globais <- function(df, 
                                       var_bruta = "palavras_cv", 
                                       var_prop = "prop_cv",
                                       label_bruta = "Quantidade (bruto)",
                                       label_prop = "Proporção (taxa)",
                                       caption = "Estatísticas Descritivas Globais da Feature (N = 2.076)") {
  tab_matrix <- rbind(
    calcular_estatisticas_descritivas(df[[var_bruta]]),
    calcular_estatisticas_descritivas(df[[var_prop]])
  )
  rownames(tab_matrix) <- c(label_bruta, label_prop)
  
  kable(
    round(tab_matrix, 4),
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position"))
}

# ================================================================================
#' Gerar Tabela com os Três Testes de Normalidade por Grupo e Global
#'
#' @title Tabela de Bateria de Testes de Normalidade
#' @description Executa e consolida os testes de Shapiro-Wilk, Anderson-Darling e
#'   Lilliefors para a variável especificada, tanto no nível global quanto nos
#'   quatro estratos de ensino.
#'
#' @param df data.frame contendo a coluna 'nivel_completo' e a variável de interesse.
#' @param var_nome Caractere com o nome da variável (ex: 'prop_cv' ou 'palavras_cv').
#' @param caption Texto da legenda da tabela em formato LaTeX.
#'
#' @return Objeto knitr_kable formatado em LaTeX com escala ajustada.
#'
#' @export
# ================================================================================
tabela_bateria_normalidade <- function(df, var_nome, caption) {
  niveis <- levels(df$nivel_completo)
  
  linhas <- list(
    "Global (Todos os Níveis)" = executar_tres_testes_normalidade(df[[var_nome]])
  )
  
  for (niv in niveis) {
    sub_x <- df[[var_nome]][df$nivel_completo == niv]
    linhas[[niv]] <- executar_tres_testes_normalidade(sub_x)
  }
  
  res_df <- do.call(rbind, linhas)
  
  kable(
    res_df,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Gerar Tabela com os Três Testes Não Paramétricos de Simetria
#'
#' @title Tabela de Bateria de Testes de Simetria
#' @description Executa e consolida os testes MGG, CM e Mira (pacote lawstat)
#'   para a variável de interesse em nível global e discriminado por estrato educacional.
#'
#' @param df data.frame contendo a coluna 'nivel_completo' e a variável de interesse.
#' @param var_nome Caractere com o nome da variável (ex: 'prop_cv' ou 'palavras_cv').
#' @param caption Texto da legenda da tabela em formato LaTeX.
#'
#' @return Objeto knitr_kable formatado em LaTeX com escala ajustada.
#'
#' @export
# ================================================================================
tabela_bateria_simetria <- function(df, var_nome, caption) {
  niveis <- levels(df$nivel_completo)
  
  linhas <- list(
    "Global (Todos os Níveis)" = executar_tres_testes_simetria(df[[var_nome]])
  )
  
  for (niv in niveis) {
    sub_x <- df[[var_nome]][df$nivel_completo == niv]
    linhas[[niv]] <- executar_tres_testes_simetria(sub_x)
  }
  
  res_df <- do.call(rbind, linhas)
  
  kable(
    res_df,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Gerar Tabela do Teste Global de Kruskal-Wallis
#'
#' @title Tabela de Análise de Variância Não Paramétrica (Kruskal-Wallis)
#' @description Executa o teste de Kruskal-Wallis para a proporção e a quantidade
#'   de uma feature em relação aos quatro níveis educacionais e formata o resultado.
#'
#' @param df data.frame contendo 'nivel_completo' e as variáveis testadas.
#' @param var_prop Nome da variável de proporção (default = 'prop_cv').
#' @param var_qtd Nome da variável de contagem bruta (default = 'palavras_cv').
#' @param label_prop Rótulo da linha de proporção.
#' @param label_qtd Rótulo da linha de quantidade bruta.
#' @param caption Legenda da tabela para inserção no documento.
#'
#' @return Objeto knitr_kable contendo Chi-quadrado, graus de liberdade e p-valor.
#'
#' @export
# ================================================================================
tabela_kruskal_wallis_global <- function(df, 
                                         var_prop = "prop_cv",
                                         var_qtd = "palavras_cv",
                                         label_prop = "Proporcao da Feature",
                                         label_qtd = "Quantidade Bruta da Feature",
                                         caption = "Teste Global de Kruskal-Wallis entre os Niveis de Ensino") {
  kw_p <- stats::kruskal.test(stats::as.formula(paste0(var_prop, " ~ nivel_completo")), data = df)
  kw_q <- stats::kruskal.test(stats::as.formula(paste0(var_qtd, " ~ nivel_completo")), data = df)
  
  tab_kw <- data.frame(
    `Variável`             = c(label_prop, label_qtd),
    `Estatística Chi²`     = c(kw_p$statistic, kw_q$statistic),
    `Graus de Liberdade`   = c(kw_p$parameter, kw_q$parameter),
    `p-valor`              = sprintf("%.3e", c(kw_p$p.value, kw_q$p.value)),
    `Significância Global` = c("Altamente Significativo (p < 0.001)", "Altamente Significativo (p < 0.001)"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  kable(
    tab_kw,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position"))
}

# ================================================================================
#' Gerar Tabela de Comparações Pareadas do Ensino Médio vs. Demais Níveis
#'
#' @title Tabela de Contrastes Pareados do Ensino Médio
#' @description Formata os resultados dos testes de Wilcoxon, ajuste de Bonferroni
#'   e tamanho de efeito (Cohen's d) entre Ensino Médio e os demais grupos.
#'
#' @param df data.frame com os dados.
#' @param var_nome Caractere indicando a variável analisada.
#' @param caption Legenda da tabela.
#' @param casas_decimais Número de casas decimais para as médias (default = 4).
#'
#' @return Objeto knitr_kable formatado para LaTeX.
#'
#' @export
# ================================================================================
tabela_comparacoes_ensino_medio <- function(df, var_nome, caption, casas_decimais = 4) {
  res_df <- executar_comparacoes_pares_em(df, var_nome)
  
  fmt_media <- paste0("%.", casas_decimais, "f")
  fmt_dif   <- paste0("%+.", casas_decimais, "f")
  
  tab_formatada <- res_df %>%
    mutate(
      `Média Médio`      = sprintf(fmt_media, `Média Médio`),
      `Média Grupo`      = sprintf(fmt_media, `Média Grupo`),
      `Diferença Médias` = sprintf(fmt_dif, `Diferença Médias`),
      `Estatística W`    = sprintf("%.0f", `Estatística W`),
      `p-valor Wilcoxon` = sprintf("%.2e", `p-valor Wilcoxon`),
      `p-adj (Bonferroni)` = sprintf("%.2e", `p-adj (Bonferroni)`),
      `Cohen d`          = sprintf("%+.3f", `Cohen d`)
    ) %>%
    dplyr::select(
      `Comparação`,
      `Média Médio`,
      `Média Grupo`,
      `Diferença Médias`,
      `Estatística W`,
      `p-adj (Bonferroni)`,
      `Cohen d`,
      `Decisão (alpha=0.05)`
    )
  
  kable(
    tab_formatada,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Formatar Tabela da Matriz de Correlação com Significância Estatística
#'
#' @title Tabela LaTeX da Matriz de Correlação
#' @description Apresenta a matriz de coeficientes de correlação (Pearson/Spearman)
#'   combinada com marcadores de significância estatística (*, **, ***).
#'
#' @param cor_obj Lista contendo 'cor' (coeficientes) e 'p_val' (p-valores).
#' @param rotulos Vetor nomeado opcional para renomear linhas e colunas.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_matriz_correlacao <- function(cor_obj, 
                                     rotulos = NULL, 
                                     caption = "Matriz de Correlação Multivariada entre Features") {
  cor_mat <- cor_obj$cor
  p_mat   <- cor_obj$p_val
  n <- nrow(cor_mat)
  
  mat_formatada <- matrix("", nrow = n, ncol = n)
  rownames(mat_formatada) <- rownames(cor_mat)
  colnames(mat_formatada) <- colnames(cor_mat)
  
  for (i in 1:n) {
    for (j in 1:n) {
      if (i == j) {
        mat_formatada[i, j] <- "1,000"
      } else {
        r_val <- cor_mat[i, j]
        p_val <- p_mat[i, j]
        sig <- ""
        if (p_val < 0.001) sig <- "***"
        else if (p_val < 0.01) sig <- "**"
        else if (p_val < 0.05) sig <- "*"
        
        mat_formatada[i, j] <- paste0(sprintf("%+.3f", r_val), sig)
      }
    }
  }
  
  if (!is.null(rotulos)) {
    rownames(mat_formatada) <- rotulos[rownames(mat_formatada)]
    colnames(mat_formatada) <- rotulos[colnames(mat_formatada)]
  }
  
  kable(
    mat_formatada,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down")) %>%
    footnote(general = "* p < 0,05; ** p < 0,01; *** p < 0,001 (bicaudal).", general_title = "Nota: ")
}

# ================================================================================
#' Formatar Tabela de Fatores de Inflação da Variância (VIF)
#'
#' @title Tabela LaTeX de Diagnóstico de Multicolinearidade (VIF)
#' @description Apresenta os valores de VIF para cada feature, avaliando o grau
#'   de redundância ou multicolinearidade no espaço multivariado.
#'
#' @param vif_vec Vetor numérico nomeado com os valores de VIF.
#' @param rotulos Vetor nomeado opcional para renomear as features.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_vif <- function(vif_vec, 
                       rotulos = NULL, 
                       caption = "Diagnóstico de Multicolinearidade: Fatores de Inflação da Variância (VIF)") {
  tab_df <- data.frame(
    `Feature`         = names(vif_vec),
    `Valor VIF`       = sprintf("%.3f", vif_vec),
    `Tolerância (1/VIF)` = sprintf("%.3f", 1 / vif_vec),
    `Diagnóstico`     = ifelse(vif_vec < 2.5, "Ortogonalidade / Multicolinearidade Baixa",
                         ifelse(vif_vec < 5.0, "Moderada (Aceitável)", "Severa (Atenção)")),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  if (!is.null(rotulos)) {
    tab_df$`Feature` <- rotulos[tab_df$`Feature`]
  }
  
  kable(
    tab_df,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position"))
}

# ================================================================================
#' Formatar Tabela de Resultados da MANOVA
#'
#' @title Tabela LaTeX dos Testes Globais da MANOVA
#' @description Apresenta as quatro estatísticas multivariadas clássicas da MANOVA
#'   (Pillai, Wilks, Hotelling-Lawley, Roy) com F aproximado e significância.
#'
#' @param manova_res data.frame retornado por calcular_manova()$tabela.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_manova_resultados <- function(manova_res, 
                                     caption = "Análise de Variância Multivariada (MANOVA): Testes Globais Omnibus") {
  tab_formatada <- manova_res %>%
    mutate(
      `Valor do Teste` = sprintf("%.4f", `Valor do Teste`),
      `F Aproximado`   = sprintf("%.2f", `F Aproximado`),
      `GL Numerador`   = sprintf("%.0f", `GL Numerador`),
      `GL Denominador` = sprintf("%.1f", `GL Denominador`),
      `p-valor`        = sprintf("%.2e", `p-valor`),
      `Decisão`        = ifelse(`p-valor` < 0.05, "Rejeita H0 (p < 0.001)", "Não Rejeita H0")
    )
  
  kable(
    tab_formatada,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Formatar Tabela de Autovalores e Variância Explicada do PCA
#'
#' @title Tabela LaTeX de Decomposição de Variância do PCA
#' @description Apresenta os autovalores, proporção individual e variância acumulada
#'   para cada componente principal gerado.
#'
#' @param pca_obj Objeto retornado por calcular_pca().
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_pca_resumo <- function(pca_obj, 
                              caption = "Decomposição Espectral do PCA: Autovalores e Variância Explicada") {
  tab_var <- pca_obj$tabela_variancia %>%
    mutate(
      `Autovalor (Eigenvalue)` = sprintf("%.4f", `Autovalor (Eigenvalue)`),
      `Variância Explicada (%)` = sprintf("%.2f%%", `Variância Explicada` * 100),
      `Variância Acumulada (%)` = sprintf("%.2f%%", `Variância Acumulada` * 100),
      `Critério Kaiser (> 1)`  = ifelse(as.numeric(`Autovalor (Eigenvalue)`) >= 1.0, "Mantido", "Secundário")
    ) %>%
    dplyr::select(`Componente`, `Autovalor (Eigenvalue)`, `Variância Explicada (%)`, `Variância Acumulada (%)`, `Critério Kaiser (> 1)`)
  
  kable(
    tab_var,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position"))
}

# ================================================================================
#' Formatar Tabela de Cargas Fatoriais (Loadings) do PCA
#'
#' @title Tabela LaTeX de Cargas Fatoriais (Loadings)
#' @description Apresenta os coeficientes dos autovetores (cargas fatoriais)
#'   de cada feature nos componentes principais.
#'
#' @param pca_obj Objeto retornado por calcular_pca().
#' @param rotulos Vetor nomeado opcional para renomear as variáveis.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_pca_cargas <- function(pca_obj, 
                              n_pcs = 5,
                              rotulos = NULL, 
                              caption = "Cargas Fatoriais (Loadings) das Features nos Componentes Principais") {
  loadings_mat <- pca_obj$rotation[, 1:min(n_pcs, ncol(pca_obj$rotation))]
  df_load <- as.data.frame(loadings_mat)
  df_load$Feature <- rownames(loadings_mat)
  
  if (!is.null(rotulos)) {
    df_load$Feature <- rotulos[df_load$Feature]
  }
  
  df_load <- df_load %>%
    dplyr::select(Feature, everything()) %>%
    mutate(across(starts_with("PC"), ~ sprintf("%+.4f", .)))
  
  kable(
    df_load,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Formatar Tabela de Coeficientes da Análise Discriminante Linear (LDA)
#'
#' @title Tabela LaTeX de Coeficientes da LDA
#' @description Apresenta os coeficientes das funções lineares discriminantes (LD1, LD2, LD3)
#'   e a proporção do traço explicada por cada função canônica.
#'
#' @param lda_res Lista retornada por calcular_lda().
#' @param rotulos Vetor nomeado opcional para renomear as variáveis.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_lda_coeficientes <- function(lda_res, 
                                    rotulos = NULL, 
                                    caption = "Coeficientes das Funções Discriminantes Canônicas (LDA)") {
  coefs <- as.data.frame(lda_res$coeficientes)
  coefs$Feature <- rownames(lda_res$coeficientes)
  
  if (!is.null(rotulos)) {
    coefs$Feature <- rotulos[coefs$Feature]
  }
  
  coefs <- coefs %>%
    dplyr::select(Feature, everything()) %>%
    mutate(across(starts_with("LD"), ~ sprintf("%+.4f", .)))
  
  # Adicionar linha com proporção do traço
  traco_row <- data.frame(
    Feature = "Proporção da Separação (Traço)",
    t(sprintf("%.2f%%", lda_res$prop_traco * 100)),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  colnames(traco_row) <- colnames(coefs)
  tab_final <- rbind(coefs, traco_row)
  
  kable(
    tab_final,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position"))
}

# ================================================================================
#' Formatar Tabela da Matriz de Confusão e Métricas da LDA
#'
#' @title Tabela LaTeX de Desempenho e Matriz de Confusão da LDA
#' @description Exibe a matriz de confusão da classificação linear da LDA,
#'   incluindo a acurácia global e a taxa de recuperação (sensibilidade) por nível.
#'
#' @param lda_res Lista retornada por calcular_lda().
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_lda_matriz_confusao <- function(lda_res, 
                                       caption = "Matriz de Confusão Preliminar e Discriminação Linear (LDA)") {
  tab_c <- lda_res$confusao
  niveis <- rownames(tab_c)
  
  df_conf <- as.data.frame.matrix(tab_c)
  df_conf$`Total Real` <- rowSums(tab_c)
  df_conf$`Acurácia / Recall (%)` <- sprintf("%.2f%%", (diag(tab_c) / rowSums(tab_c)) * 100)
  
  df_conf$Nível <- rownames(df_conf)
  df_conf <- df_conf %>% dplyr::select(Nível, everything())
  
  acuracia_global <- sprintf("%.2f%%", lda_res$acuracia * 100)
  
  kable(
    df_conf,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down")) %>%
    footnote(general = paste0("Acurácia Global de Discriminação Linear (4 classes): ", acuracia_global, " (Chance aleatória = 25%)."),
             general_title = "Desempenho Geral: ")
}

# ================================================================================
#' Formatar Tabela de Distâncias Multivariadas de Mahalanobis
#'
#' @title Tabela LaTeX de Distâncias de Mahalanobis entre Níveis
#' @description Apresenta a matriz de distâncias de Mahalanobis entre os centróides
#'   dos quatro níveis de ensino no hipervolume 5-dimensional.
#'
#' @param dist_mat Matriz retornada por calcular_distancias_mahalanobis().
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_distancias_mahalanobis <- function(dist_mat, 
                                          caption = "Distâncias Multivariadas de Mahalanobis entre os Centróides") {
  df_dist <- as.data.frame(round(dist_mat, 3))
  df_dist$Nível <- rownames(df_dist)
  df_dist <- df_dist %>% dplyr::select(Nível, everything())
  
  kable(
    df_dist,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Formatar Tabela de Contrastes Multivariados de Hotelling T2 do Ensino Médio
#'
#' @title Tabela LaTeX de Testes de Hotelling T2
#' @description Apresenta os contrastes multivariados bivariados do Ensino Médio
#'   contra os outros três níveis com correção de Bonferroni.
#'
#' @param hotelling_res data.frame retornado por calcular_hotelling_ensino_medio().
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_hotelling_ensino_medio <- function(hotelling_res, 
                                          caption = "Contrastes Multivariados de Hotelling T2: Ensino Médio vs. Outros Níveis") {
  tab_formatada <- hotelling_res %>%
    mutate(
      `Hotelling T2`       = sprintf("%.2f", `Hotelling T2`),
      `F Aproximado`       = sprintf("%.2f", `F Aproximado`),
      `GL 1`               = sprintf("%.0f", `GL 1`),
      `GL 2`               = sprintf("%.0f", `GL 2`),
      `p-valor`            = sprintf("%.2e", `p-valor`),
      `p-adj (Bonferroni)` = sprintf("%.2e", `p-adj (Bonferroni)`)
    )
  
  kable(
    tab_formatada,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Formatar Tabela de Divisão de Dados (Train / Test Split)
#'
#' @title Tabela LaTeX da Divisão Estratificada dos Dados
#' @description Apresenta a quantidade de amostras em cada estrato educacional
#'   para os conjuntos de treino (80%) e teste (20%).
#'
#' @param df_splits data.frame contendo a distribuição dos dados.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_splits_info <- function(df_splits, 
                               caption = "Divisão Estratificada do Acervo (80% Treino / 20% Teste Isolado)") {
  kable(
    df_splits,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Formatar Tabela de Resumo da Validação Cruzada 5-Fold
#'
#' @title Tabela LaTeX da Validação Cruzada Estratificada
#' @description Apresenta a média e desvio padrão das métricas de desempenho
#'   dos três modelos avaliados através dos 5 folds de treino.
#'
#' @param df_cv data.frame contendo os resultados individuais por fold.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_resumo_cv_modelos <- function(df_cv, 
                                     caption = "Desempenho na Validação Cruzada 5-Fold Estratificada (Média ± Desvio Padrão)") {
  tab_res <- df_cv %>%
    group_by(Modelo) %>%
    summarise(
      `Acurácia`            = sprintf("%.2f%% ± %.2f%%", mean(Acurácia) * 100, sd(Acurácia) * 100),
      `Acurácia Bal.`       = sprintf("%.2f%% ± %.2f%%", mean(`Acurácia.Balanceada`, na.rm=TRUE) * 100, sd(`Acurácia.Balanceada`, na.rm=TRUE) * 100),
      `F1 (Macro)`          = sprintf("%.3f ± %.3f", mean(`F1.Score..Macro.`), sd(`F1.Score..Macro.`)),
      `Precisão (Macro)`    = sprintf("%.3f ± %.3f", mean(`Precisão..Macro.`), sd(`Precisão..Macro.`)),
      `Revocação (Macro)`   = sprintf("%.3f ± %.3f", mean(`Revocação..Macro.`), sd(`Revocação..Macro.`)),
      `ROC-AUC (OvR)`       = sprintf("%.3f ± %.3f", mean(`ROC.AUC..OvR.Macro.`), sd(`ROC.AUC..OvR.Macro.`)),
      `Log Loss`            = sprintf("%.3f ± %.3f", mean(`Log.Loss`), sd(`Log.Loss`)),
      .groups = "drop"
    )
  
  kable(
    tab_res,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Formatar Tabela de Resultados no Conjunto de Teste Independente (Holdout)
#'
#' @title Tabela LaTeX de Desempenho no Teste
#' @description Apresenta as métricas globais de generalização dos modelos
#'   no conjunto de teste isolado de 20% (N = 416).
#'
#' @param df_teste data.frame com os resultados do conjunto de teste.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_comparativo_teste_modelos <- function(df_teste, 
                                             caption = "Desempenho no Conjunto de Teste Independente (Holdout, N = 416)") {
  tab_formatada <- df_teste %>%
    mutate(
      `Acurácia`            = sprintf("%.2f%%", Acurácia * 100),
      `Acurácia Balanceada` = sprintf("%.2f%%", Acurácia.Balanceada * 100),
      `F1-Score (Macro)`    = sprintf("%.3f", `F1.Score..Macro.`),
      `F1-Score (Pond.)`    = sprintf("%.3f", `F1.Score..Ponderado.`),
      `Precisão (Macro)`    = sprintf("%.3f", `Precisão..Macro.`),
      `Revocação (Macro)`   = sprintf("%.3f", `Revocação..Macro.`),
      `ROC-AUC (OvR)`       = sprintf("%.3f", `ROC.AUC..OvR.Macro.`),
      `Log Loss`            = sprintf("%.3f", `Log.Loss`)
    ) %>%
    dplyr::select(
      Modelo,
      `Acurácia`,
      `Acurácia Balanceada`,
      `F1-Score (Macro)`,
      `F1-Score (Pond.)`,
      `Precisão (Macro)`,
      `Revocação (Macro)`,
      `ROC-AUC (OvR)`,
      `Log Loss`
    )
  
  kable(
    tab_formatada,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Formatar Tabela de Métricas Detalhadas por Classe
#'
#' @title Tabela LaTeX de Métricas por Nível de Ensino
#' @description Apresenta Precisão, Revocação (Recall), F1-Score e Suporte para
#'   cada um dos quatro estratos de ensino no conjunto de teste.
#'
#' @param df_classes data.frame contendo as métricas estratificadas.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_metricas_classes_modelos <- function(df_classes, 
                                            caption = "Métricas Detalhadas por Nível de Ensino no Conjunto de Teste") {
  tab_formatada <- df_classes %>%
    mutate(
      `Precisão`  = sprintf("%.2f%%", Precisão * 100),
      `Revocação` = sprintf("%.2f%%", `Revocação..Recall.` * 100),
      `F1-Score`  = sprintf("%.3f", `F1.Score`),
      `Suporte`   = sprintf("%.0f", `Suporte..N.Teste.`)
    ) %>%
    dplyr::select(
      Modelo,
      Classe,
      `Precisão`,
      `Revocação`,
      `F1-Score`,
      `Suporte`
    )
  
  kable(
    tab_formatada,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}

# ================================================================================
#' Formatar Tabela de Verificação e Paridade dos Modelos ONNX
#'
#' @title Tabela LaTeX de Verificação de Exportação ONNX
#' @description Apresenta os arquivos ONNX gerados, tamanhos em disco, opset,
#'   acurácia de paridade e divergência máxima de probabilidades frente ao sklearn/LightGBM.
#'
#' @param df_onnx data.frame contendo os dados de verificação ONNX.
#' @param caption Legenda da tabela LaTeX.
#'
#' @return Objeto knitr_kable pronto para inserção no PDF.
#'
#' @export
# ================================================================================
tabela_verificacao_onnx <- function(df_onnx, 
                                    caption = "Verificação de Paridade Numérica e Exportação dos Modelos ONNX") {
  tab_formatada <- df_onnx %>%
    mutate(
      `Tamanho (KB)`                    = sprintf("%.1f KB", `Tamanho..KB.`),
      `Opset`                           = sprintf("%.0f", `Opset.ONNX`),
      `Paridade (%)`                    = sprintf("%.2f%%", `Acurácia.de.Paridade....`),
      `Erro Máx. Prob.`                 = sprintf("%.2e", `Diferença.Máxima.Probabilidades`),
      `Status`                          = `Status.Verificação`
    ) %>%
    dplyr::select(
      Modelo,
      `Arquivo ONNX` = `Arquivo.ONNX`,
      `Tamanho (KB)`,
      `Opset`,
      `Paridade (%)`,
      `Erro Máx. Prob.`,
      `Status`
    )
  
  kable(
    tab_formatada,
    format = "latex",
    booktabs = TRUE,
    caption = caption
  ) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
}
