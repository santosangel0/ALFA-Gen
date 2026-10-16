# ================================================================================
# PROJETO ALFA-GEN: MÓDULO DE FUNÇÕES ESTATÍSTICAS
# ================================================================================
# Descrição: Cálculos estatísticos descritivos, testes de normalidade (3 testes),
#            testes não paramétricos de simetria (3 testes) e contrastes pareados
#            entre o Ensino Médio e os demais níveis educacionais.
# ================================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(nortest)
  library(lawstat)
  library(moments)
  library(MASS)
})

# ================================================================================
#' Calcular Estatísticas Descritivas de um Vetor Numérico
#'
#' @title Estatísticas Descritivas Unidimensionais
#' @description Computa as principais medidas de posição, dispersão, assimetria
#'   e curtose para um vetor numérico.
#'
#' @param x Vetor numérico contendo os dados a serem sumarizados.
#' @param na.rm Lógico indicando se valores NA devem ser removidos (default = TRUE).
#'
#' @return Vetor nomeado contendo: N, Média, Desv. Padrão, Mediana, IQR,
#'   Mínimo, Máximo, Assimetria (Skewness) e Curtose (Kurtosis).
#'
#' @export
# ================================================================================
calcular_estatisticas_descritivas <- function(x, na.rm = TRUE) {
  if (na.rm) {
    x <- x[!is.na(x)]
  }
  
  res <- c(
    "N"                 = length(x),
    "Média"             = mean(x),
    "Desv. Padrão"      = sd(x),
    "Mediana"           = median(x),
    "IQR"               = IQR(x),
    "Mínimo"            = min(x),
    "Máximo"            = max(x),
    "Assimetria (Skew)" = moments::skewness(x),
    "Curtose (Kurt)"    = moments::kurtosis(x)
  )
  
  return(res)
}

# ================================================================================
#' Executar Três Testes de Normalidade para um Vetor Numérico
#'
#' @title Bateria de Testes de Normalidade
#' @description Aplica três testes formais de aderência à distribuição normal:
#'   1. Shapiro-Wilk (W)
#'   2. Anderson-Darling (A)
#'   3. Lilliefors / Kolmogorov-Smirnov (D)
#'
#' @param x Vetor numérico contendo as observações da amostra.
#'
#' @return data.frame com 1 linha e 3 colunas contendo a estatística de teste
#'   e o respectivo p-valor formatado em notação científica.
#'
#' @export
# ================================================================================
executar_tres_testes_normalidade <- function(x) {
  x <- x[!is.na(x)]
  
  # 1. Shapiro-Wilk
  t_sw <- stats::shapiro.test(x)
  # 2. Anderson-Darling
  t_ad <- nortest::ad.test(x)
  # 3. Lilliefors (K-S adaptado)
  t_lf <- nortest::lillie.test(x)
  
  res <- data.frame(
    `Shapiro-Wilk (W)`     = sprintf("%.4f (p = %.2e)", t_sw$statistic, t_sw$p.value),
    `Anderson-Darling (A)` = sprintf("%.4f (p = %.2e)", t_ad$statistic, t_ad$p.value),
    `Lilliefors (D)`       = sprintf("%.4f (p = %.2e)", t_lf$statistic, t_lf$p.value),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  return(res)
}

# ================================================================================
#' Executar Três Testes Não Paramétricos de Simetria em Torno da Mediana
#'
#' @title Bateria de Testes de Simetria Não Paramétricos
#' @description Aplica três testes não paramétricos de simetria (pacote lawstat)
#'   em torno da mediana, compensando o baixo poder de testes individuais:
#'   1. Miao-Gel-Gastwirth (MGG, 2006)
#'   2. Cabilio-Masaro (CM, 1996)
#'   3. Mira (M, 1999)
#'
#' @param x Vetor numérico contendo as observações da amostra.
#' @param boot Lógico indicando se bootstrap deve ser usado (default = FALSE para formulação assintótica).
#'
#' @return data.frame contendo o Skewness amostral e as estatísticas/p-valores dos 3 testes.
#'
#' @export
# ================================================================================
executar_tres_testes_simetria <- function(x, boot = FALSE) {
  x <- x[!is.na(x)]
  
  # 1. Teste MGG (Miao, Gel & Gastwirth)
  t_mgg <- lawstat::symmetry.test(x, option = "MGG", boot = boot)
  # 2. Teste CM (Cabilio & Masaro)
  t_cm  <- lawstat::symmetry.test(x, option = "CM", boot = boot)
  # 3. Teste M (Mira)
  t_m   <- lawstat::symmetry.test(x, option = "M", boot = boot)
  
  res <- data.frame(
    `Skewness Amostral` = sprintf("%.3f", moments::skewness(x)),
    `Teste MGG`         = sprintf("stat = %.3f (p = %.2e)", t_mgg$statistic, t_mgg$p.value),
    `Teste CM`          = sprintf("stat = %.3f (p = %.2e)", t_cm$statistic, t_cm$p.value),
    `Teste M`           = sprintf("stat = %.3f (p = %.2e)", t_m$statistic, t_m$p.value),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  return(res)
}

# ================================================================================
#' Comparar Estatisticamente o Ensino Médio com os Demais Níveis Educacionais
#'
#' @title Comparações Pareadas: Ensino Médio vs. Outros Níveis
#' @description Conduz testes de Wilcoxon-Mann-Whitney (não paramétrico) e testes t
#'   de Welch (paramétrico) comparando o grupo Ensino Médio contra Fundamental I,
#'   Fundamental II e Ensino Superior, incluindo correção de Bonferroni e Cohen's d.
#'
#' @param df data.frame contendo as colunas 'nivel_completo' e a variável analisada.
#' @param var_nome Caractere com o nome da variável a ser testada (ex: 'prop_cv' ou 'palavras_cv').
#'
#' @return data.frame com os resultados das três comparações pareadas, estatísticas,
#'   p-valores não corrigidos e ajustados, tamanho de efeito e decisão inferencial.
#'
#' @export
# ================================================================================
executar_comparacoes_pares_em <- function(df, var_nome) {
  em_vals <- df[[var_nome]][df$nivel_completo == "Ensino Médio"]
  outros_niveis <- c("Ensino Fundamental I", "Ensino Fundamental II", "Ensino Superior")
  
  linhas <- list()
  for (outro in outros_niveis) {
    comp_vals <- df[[var_nome]][df$nivel_completo == outro]
    
    # Wilcoxon-Mann-Whitney (não paramétrico)
    wt <- stats::wilcox.test(em_vals, comp_vals, exact = FALSE)
    # Welch t-test (paramétrico robusto)
    tt <- stats::t.test(em_vals, comp_vals)
    
    # Tamanho de efeito (Cohen's d)
    n1 <- length(em_vals)
    n2 <- length(comp_vals)
    s_pool <- sqrt(((n1 - 1) * var(em_vals) + (n2 - 1) * var(comp_vals)) / (n1 + n2 - 2))
    cohen_d <- (mean(em_vals) - mean(comp_vals)) / s_pool
    
    linhas[[outro]] <- data.frame(
      `Comparação`       = paste0("Ensino Médio vs. ", outro),
      `Média Médio`      = mean(em_vals),
      `Média Grupo`      = mean(comp_vals),
      `Diferença Médias` = mean(em_vals) - mean(comp_vals),
      `Estatística W`    = wt$statistic,
      `p-valor Wilcoxon` = wt$p.value,
      `p-valor Welch t`  = tt$p.value,
      `Cohen d`          = cohen_d,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }
  
  res_df <- dplyr::bind_rows(linhas)
  # Correção de Bonferroni para as 3 comparações
  res_df$`p-adj (Bonferroni)` <- stats::p.adjust(res_df$`p-valor Wilcoxon`, method = "bonferroni")
  res_df$`Decisão (alpha=0.05)` <- ifelse(
    res_df$`p-adj (Bonferroni)` < 0.05,
    "Rejeita H0 (Diferença Sig.)",
    "Não Rejeita H0 (Sem Diferença)"
  )
  
  return(res_df)
}

# ================================================================================
#' Calcular Matriz de Correlação e Matriz de p-valores
#'
#' @title Matriz de Correlação Multivariada
#' @description Computa a matriz de coeficientes de correlação (Pearson ou Spearman)
#'   e a matriz correspondente de significância estatística (p-valores bicaudais).
#'
#' @param df data.frame contendo as variáveis numéricas.
#' @param vars Vetor de caracteres com os nomes das colunas numéricas.
#' @param metodo Caractere indicando o método ('pearson' ou 'spearman').
#'
#' @return Lista contendo 'cor' (matriz de correlação) e 'p_val' (matriz de p-valores).
#'
#' @export
# ================================================================================
calcular_matriz_correlacao <- function(df, vars, metodo = "pearson") {
  X <- as.matrix(df[, vars])
  n_vars <- length(vars)
  
  cor_mat <- cor(X, method = metodo, use = "pairwise.complete.obs")
  p_mat   <- matrix(0, nrow = n_vars, ncol = n_vars, dimnames = list(vars, vars))
  
  for (i in 1:n_vars) {
    for (j in 1:n_vars) {
      if (i == j) {
        p_mat[i, j] <- 0
      } else {
        test_res <- stats::cor.test(X[, i], X[, j], method = metodo)
        p_mat[i, j] <- test_res$p.value
      }
    }
  }
  
  return(list(cor = cor_mat, p_val = p_mat))
}

# ================================================================================
#' Calcular Fatores de Inflação da Variância (VIF)
#'
#' @title Diagnóstico de Multicolinearidade (VIF)
#' @description Computa o Fator de Inflação da Variância (Variance Inflation Factor)
#'   para cada variável através da regressão linear da feature contra as demais.
#'
#' @param df data.frame contendo as variáveis numéricas.
#' @param vars Vetor de caracteres com os nomes das colunas numéricas.
#'
#' @return Vetor nomeado contendo o valor de VIF para cada variável analisada.
#'
#' @export
# ================================================================================
calcular_vif <- function(df, vars) {
  X <- df[, vars]
  vifs <- sapply(vars, function(v) {
    form <- as.formula(paste(v, "~ ."))
    fit <- stats::lm(form, data = X)
    r2 <- summary(fit)$r.squared
    1 / (1 - r2)
  })
  return(vifs)
}

# ================================================================================
#' Executar Análise de Variância Multivariada (MANOVA)
#'
#' @title Análise de Variância Multivariada (MANOVA)
#' @description Ajusta um modelo MANOVA para testar a hipótese de igualdade dos
#'   vetores de médias multivariadas entre os grupos educacionais, extraindo os
#'   quatro testes omnibus clássicos: Pillai, Wilks, Hotelling-Lawley e Roy.
#'
#' @param df data.frame contendo os dados.
#' @param vars Vetor de caracteres com os nomes das variáveis dependentes.
#' @param grupo Nome da coluna com o fator de agrupamento (default = 'nivel_completo').
#'
#' @return data.frame com os resultados dos quatro testes estatísticos da MANOVA.
#'
#' @export
# ================================================================================
calcular_manova <- function(df, vars, grupo = "nivel_completo") {
  Y <- as.matrix(df[, vars])
  G <- df[[grupo]]
  fit <- stats::manova(Y ~ G)
  
  testes <- c("Pillai", "Wilks", "Hotelling-Lawley", "Roy")
  linhas <- list()
  
  for (t_nome in testes) {
    s <- summary(fit, test = t_nome)
    row_g <- s$stats[1, ]
    linhas[[t_nome]] <- data.frame(
      `Estatística Multivariada` = t_nome,
      `Valor do Teste`          = unname(row_g[t_nome]),
      `F Aproximado`            = unname(row_g["approx F"]),
      `GL Numerador`            = unname(row_g["num Df"]),
      `GL Denominador`          = unname(row_g["den Df"]),
      `p-valor`                 = unname(row_g["Pr(>F)"]),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }
  
  res_df <- dplyr::bind_rows(linhas)
  return(list(fit = fit, tabela = res_df))
}

# ================================================================================
#' Executar Análise de Componentes Principais (PCA)
#'
#' @title Análise de Componentes Principais (PCA)
#' @description Realiza a decomposição em componentes principais sobre as variáveis
#'   padronizadas (scale = TRUE), extraindo autovalores, proporções de variância,
#'   matriz de cargas fatoriais (loadings) e coordenadas das amostras.
#'
#' @param df data.frame contendo as variáveis numéricas.
#' @param vars Vetor de caracteres com os nomes das colunas numéricas.
#' @param scale Lógico indicando se os dados devem ser padronizados (default = TRUE).
#'
#' @return Objeto prcomp enriquecido com tabela de variância e autovalores.
#'
#' @export
# ================================================================================
calcular_pca <- function(df, vars, scale = TRUE) {
  X <- df[, vars]
  pca_res <- stats::prcomp(X, center = TRUE, scale. = scale)
  
  eigs <- pca_res$sdev^2
  var_prop <- eigs / sum(eigs)
  var_cum  <- cumsum(var_prop)
  
  tab_var <- data.frame(
    `Componente`            = paste0("PC", seq_along(eigs)),
    `Autovalor (Eigenvalue)` = eigs,
    `Variância Explicada`   = var_prop,
    `Variância Acumulada`   = var_cum,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  pca_res$tabela_variancia <- tab_var
  return(pca_res)
}

# ================================================================================
#' Executar Análise Discriminante Linear (LDA)
#'
#' @title Análise Discriminante Linear (LDA)
#' @description Ajusta o classificador discriminante linear de Fisher/Rao (MASS::lda),
#'   calculando os coeficientes discriminantes canônicos, proporção de traço,
#'   projeções e matriz de confusão preliminar.
#'
#' @param df data.frame contendo os dados.
#' @param vars Vetor de caracteres com os nomes das variáveis explicativas.
#' @param grupo Nome da coluna com o fator de agrupamento (default = 'nivel_completo').
#'
#' @return Lista contendo modelo lda, coeficientes, proporção de traço, previsões,
#'   matriz de confusão e acurácia linear aparente.
#'
#' @export
# ================================================================================
calcular_lda <- function(df, vars, grupo = "nivel_completo") {
  form <- as.formula(paste(grupo, "~", paste(vars, collapse = " + ")))
  lda_model <- MASS::lda(form, data = df)
  
  pred <- stats::predict(lda_model)
  tab_conf <- table(Real = df[[grupo]], Predito = pred$class)
  acuracia <- sum(diag(tab_conf)) / sum(tab_conf)
  
  return(list(
    modelo       = lda_model,
    coeficientes = lda_model$scaling,
    prop_traco   = lda_model$svd^2 / sum(lda_model$svd^2),
    predicoes    = pred,
    confusao     = tab_conf,
    acuracia     = acuracia
  ))
}

# ================================================================================
#' Calcular Matriz de Distâncias de Mahalanobis entre Centróides
#'
#' @title Distâncias Multivariadas de Mahalanobis
#' @description Computa a matriz de distâncias de Mahalanobis entre os vetores de
#'   médias (centróides) dos grupos utilizando a matriz de covariância agrupada
#'   intraclasse (pooled covariance matrix) oriunda dos resíduos da MANOVA.
#'
#' @param df data.frame contendo os dados.
#' @param vars Vetor de caracteres com as variáveis numéricas.
#' @param grupo Nome da coluna com o fator de agrupamento (default = 'nivel_completo').
#'
#' @return Matriz simétrica com as distâncias de Mahalanobis entre os grupos.
#'
#' @export
# ================================================================================
calcular_distancias_mahalanobis <- function(df, vars, grupo = "nivel_completo") {
  X <- as.matrix(df[, vars])
  G <- df[[grupo]]
  niveis <- levels(G)
  if (is.null(niveis)) niveis <- unique(as.character(G))
  
  # MANOVA para extrair resíduos e covariância combinada (pooled)
  fit_m <- stats::manova(X ~ G)
  cov_pooled <- stats::cov(stats::residuals(fit_m))
  inv_cov_pooled <- solve(cov_pooled)
  
  centroides <- stats::aggregate(X ~ G, FUN = mean)
  rownames(centroides) <- as.character(centroides$G)
  mu_mat <- as.matrix(centroides[, vars])
  
  n_g <- length(niveis)
  dist_mat <- matrix(0, nrow = n_g, ncol = n_g, dimnames = list(niveis, niveis))
  
  for (i in 1:n_g) {
    for (j in 1:n_g) {
      diff_ij <- mu_mat[niveis[i], ] - mu_mat[niveis[j], ]
      dist_mat[i, j] <- sqrt(as.numeric(t(diff_ij) %*% inv_cov_pooled %*% diff_ij))
    }
  }
  
  return(dist_mat)
}

# ================================================================================
#' Calcular Contrastes Multivariados de Hotelling T2 contra o Ensino Médio
#'
#' @title Teste T2 de Hotelling (Ensino Médio vs. Demais Níveis)
#' @description Executa o teste biamostral multivariado de Hotelling T2 comparando
#'   o vetor de médias do Ensino Médio com cada um dos outros níveis educacionais,
#'   incluindo aproximação por F de Fisher e correção de Bonferroni.
#'
#' @param df data.frame contendo os dados.
#' @param vars Vetor de caracteres com as variáveis numéricas analisadas.
#' @param grupo Nome da coluna com o fator de agrupamento (default = 'nivel_completo').
#' @param grupo_referencia Nome do grupo fixo de referência (default = 'Ensino Médio').
#'
#' @return data.frame com T2, estatística F, graus de liberdade, p-valor e p-ajustado.
#'
#' @export
# ================================================================================
calcular_hotelling_ensino_medio <- function(df, 
                                            vars, 
                                            grupo = "nivel_completo",
                                            grupo_referencia = "Ensino Médio") {
  niveis <- levels(df[[grupo]])
  if (is.null(niveis)) niveis <- unique(as.character(df[[grupo]]))
  outros <- setdiff(niveis, grupo_referencia)
  
  X_ref <- as.matrix(df[df[[grupo]] == grupo_referencia, vars])
  n1 <- nrow(X_ref)
  p  <- length(vars)
  
  linhas <- list()
  for (outro in outros) {
    X_comp <- as.matrix(df[df[[grupo]] == outro, vars])
    n2 <- nrow(X_comp)
    
    m1 <- colMeans(X_ref)
    m2 <- colMeans(X_comp)
    d  <- m1 - m2
    
    S1 <- stats::cov(X_ref)
    S2 <- stats::cov(X_comp)
    Sp <- ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    
    T2 <- (n1 * n2 / (n1 + n2)) * as.numeric(t(d) %*% solve(Sp) %*% d)
    df1 <- p
    df2 <- n1 + n2 - p - 1
    F_stat <- (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * T2
    p_val <- 1 - stats::pf(F_stat, df1, df2)
    
    linhas[[outro]] <- data.frame(
      `Comparação`       = paste0(grupo_referencia, " vs. ", outro),
      `Hotelling T2`     = T2,
      `F Aproximado`     = F_stat,
      `GL 1`             = df1,
      `GL 2`             = df2,
      `p-valor`          = p_val,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }
  
  res_df <- dplyr::bind_rows(linhas)
  res_df$`p-adj (Bonferroni)` <- stats::p.adjust(res_df$`p-valor`, method = "bonferroni")
  res_df$`Decisão (alpha=0.05)` <- ifelse(
    res_df$`p-adj (Bonferroni)` < 0.05,
    "Rejeita H0 (Diferença Multivariada Sig.)",
    "Não Rejeita H0"
  )
  
  return(res_df)
}
