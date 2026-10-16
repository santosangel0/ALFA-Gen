# ================================================================================
# PROJETO ALFA-GEN: MÓDULO DE GERAÇÃO DE GRÁFICOS
# ================================================================================
# Descrição: Funções dedicadas à produção de painéis e gráficos estatísticos
#            em R base para visualização e diagnóstico das distribuições.
# ================================================================================

suppressPackageStartupMessages({
  library(dplyr)
})

# ================================================================================
#' Gerar Gráficos Globais da Distribuição de uma Feature (Bruta e Proporção)
#'
#' @title Gráficos de Densidade e Distribuição Global
#' @description Produz dois histogramas com densidade relativa (freq = FALSE) e
#'   curvas de densidade empírica para a quantidade bruta e a proporção da feature.
#'
#' @param df data.frame contendo as variáveis analisadas.
#' @param var_bruta Nome da variável com a contagem bruta (default = 'palavras_cv').
#' @param var_prop Nome da variável com a proporção (default = 'prop_cv').
#' @param label_bruta Rótulo do eixo X para a quantidade bruta.
#' @param label_prop Rótulo do eixo X para a proporção.
#'
#' @return Invisível (NULL). Executa a renderização gráfica no dispositivo atual.
#'
#' @export
# ================================================================================
plot_distribuicao_global <- function(df, 
                                     var_bruta = "palavras_cv", 
                                     var_prop = "prop_cv",
                                     label_bruta = "Quantidade Bruta", 
                                     label_prop = "Proporção / Taxa") {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  par(mfrow = c(1, 2), mar = c(4.2, 4.2, 3.8, 1))
  
  val_bruta <- df[[var_bruta]][!is.na(df[[var_bruta]])]
  val_prop  <- df[[var_prop]][!is.na(df[[var_prop]])]
  
  # Quebra inteligente para os títulos e eixos caberem sem truncamento
  titulo_bruta <- paste(strwrap(paste0("Histograma: ", label_bruta), width = 28), collapse = "\n")
  titulo_prop  <- paste(strwrap(paste0("Histograma: ", label_prop), width = 28), collapse = "\n")
  xlab_bruta   <- paste(strwrap(label_bruta, width = 32), collapse = "\n")
  xlab_prop    <- paste(strwrap(label_prop, width = 32), collapse = "\n")
  
  # 1. Histograma da Quantidade Absoluta
  hist(
    val_bruta,
    breaks = 30,
    freq = FALSE,
    col = "#4C72B0",
    border = "white",
    main = titulo_bruta,
    xlab = xlab_bruta,
    ylab = "Densidade",
    cex.main = 0.88,
    cex.lab = 0.82,
    cex.axis = 0.8
  )
  if (stats::sd(val_bruta) > 0) {
    lines(stats::density(val_bruta), col = "#C44E52", lwd = 2.5)
  }
  
  # 2. Histograma da Proporção
  hist(
    val_prop,
    breaks = 30,
    freq = FALSE,
    col = "#55A868",
    border = "white",
    main = titulo_prop,
    xlab = xlab_prop,
    ylab = "Densidade",
    cex.main = 0.88,
    cex.lab = 0.82,
    cex.axis = 0.8
  )
  if (stats::sd(val_prop) > 0) {
    lines(stats::density(val_prop), col = "#C44E52", lwd = 2.5)
  }
  
  invisible(NULL)
}

# ================================================================================
#' Gerar Painel 2x2 com Histogramas de Densidade por Nível de Ensino
#'
#' @title Painel 2x2 de Histogramas por Nível Educacional
#' @description Configura um painel 2x2 usando estritamente 'par(mfrow = c(2, 2))'
#'   e 'hist(freq = FALSE)' para comparar a distribuição da feature nos quatro níveis:
#'   Fundamental I, Fundamental II, Ensino Médio e Ensino Superior.
#'
#' @param df data.frame contendo 'nivel_completo' e a variável de interesse.
#' @param var_nome Caractere com o nome da variável analisada.
#' @param titulo_painel Texto principal exibido no topo externo do painel.
#' @param xlab_val Rótulo do eixo X (opcional, calculado automaticamente se NULL).
#' @param xlim_val Limites do eixo X (opcional, calculado automaticamente se NULL).
#'
#' @return Invisível (NULL). Plota o painel 2x2 no dispositivo ativo.
#'
#' @export
# ================================================================================
plot_painel_2x2_niveis <- function(df, 
                                   var_nome = "prop_cv", 
                                   titulo_painel = "Painel 2x2: Distribuição por Nível de Ensino",
                                   xlab_val = NULL,
                                   xlim_val = NULL) {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  # Margem externa superior ampliada (3.2 linhas) para o título principal não encavalar
  par(mfrow = c(2, 2), mar = c(4.5, 4.2, 2.8, 1.2), oma = c(0, 0, 3.2, 0))
  
  niveis_nomes <- levels(df$nivel_completo)
  cores <- c("#6BAED6", "#74C476", "#FD8D3C", "#9E9AC8")
  
  vals_todos <- df[[var_nome]][!is.na(df[[var_nome]])]
  
  if (is.null(xlab_val)) {
    xlab_val <- var_nome
  }
  xlab_sub <- paste(strwrap(xlab_val, width = 34), collapse = "\n")
  
  if (is.null(xlim_val)) {
    min_v <- min(vals_todos)
    max_v <- max(vals_todos)
    rng <- max_v - min_v
    if (rng == 0) rng <- 1
    xlim_val <- c(max(0, min_v - 0.05 * rng), max_v + 0.05 * rng)
  }
  
  for (i in 1:4) {
    niv <- niveis_nomes[i]
    sub_dados <- df[[var_nome]][df$nivel_completo == niv]
    sub_dados <- sub_dados[!is.na(sub_dados)]
    
    # Histograma com freq = FALSE conforme requerido
    h <- hist(
      sub_dados,
      breaks = 20,
      freq = FALSE,
      col = cores[i],
      border = "white",
      main = paste0(niv, " (N = ", length(sub_dados), ")"),
      xlab = xlab_sub,
      ylab = "Densidade",
      xlim = xlim_val,
      cex.main = 0.92,
      cex.lab = 0.82,
      cex.axis = 0.8
    )
    
    # Curva de densidade empírica se houver variabilidade
    if (stats::sd(sub_dados) > 0) {
      lines(stats::density(sub_dados, from = xlim_val[1], to = xlim_val[2]), col = "#D95F02", lwd = 2.5)
      
      # Curva normal de referência ajustada
      x_seq <- seq(xlim_val[1], xlim_val[2], length.out = 200)
      y_norm <- stats::dnorm(x_seq, mean = mean(sub_dados), sd = sd(sub_dados))
      lines(x_seq, y_norm, col = "black", lwd = 1.5, lty = 2)
    }
    
    if (i == 1 && stats::sd(sub_dados) > 0) {
      legend(
        "topright",
        legend = c("Densidade Real", "Normal Ajustada"),
        col = c("#D95F02", "black"),
        lty = c(1, 2),
        lwd = c(2.5, 1.5),
        bty = "n",
        cex = 0.75
      )
    }
  }
  
  # Escala adaptativa para o título caber 100% no espaço superior
  n_carac <- nchar(titulo_painel)
  cex_titulo <- if (n_carac > 70) 0.86 else if (n_carac > 55) 0.94 else 1.05
  mtext(
    titulo_painel,
    outer = TRUE,
    side = 3,
    line = 0.9,
    cex = cex_titulo,
    font = 2
  )
  
  invisible(NULL)
}

# ================================================================================
#' Gerar Gráfico de Médias com Intervalos de Confiança de 95%
#'
#' @title Gráfico de Médias e Intervalos de Confiança (95%)
#' @description Apresenta as médias da variável para cada um dos quatro níveis de ensino
#'   com barras de erro baseadas em erro padrão (IC de 95%), destacando a linha de
#'   referência do Ensino Médio.
#'
#' @param df data.frame com as variáveis 'nivel_completo' e a variável analisada.
#' @param var_nome Caractere indicando a variável de interesse (default = 'prop_cv').
#' @param ylab_val Rótulo do eixo Y.
#' @param titulo_grafico Título principal da figura.
#'
#' @return Invisível (NULL). Plota a figura no dispositivo gráfico ativo.
#'
#' @export
# ================================================================================
plot_medias_ci_niveis <- function(df, 
                                  var_nome = "prop_cv",
                                  ylab_val = "Proporção Média",
                                  titulo_grafico = "Comparação de Médias com IC de 95%") {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  df_ci <- df %>%
    group_by(nivel_completo) %>%
    summarise(
      n = n(),
      media = mean(.data[[var_nome]], na.rm = TRUE),
      se = sd(.data[[var_nome]], na.rm = TRUE) / sqrt(n),
      ci_inf = media - 1.96 * se,
      ci_sup = media + 1.96 * se,
      .groups = "drop"
    )
  
  y_min <- min(df_ci$ci_inf, na.rm = TRUE)
  y_max <- max(df_ci$ci_sup, na.rm = TRUE)
  margem <- (y_max - y_min) * 0.15
  if (margem == 0) margem <- 0.01
  
  # Ajustar escala do título de forma adaptativa ao comprimento do texto
  n_carac_ci <- nchar(titulo_grafico)
  cex_titulo_ci <- if (n_carac_ci > 60) 0.88 else if (n_carac_ci > 45) 0.98 else 1.10
  
  par(mar = c(4.5, 5.2, 3.5, 2))
  plot(
    1:4,
    df_ci$media,
    pch = 19,
    cex = 1.6,
    col = c("#2B83BA", "#ABDDA4", "#D7191C", "#FDAE61"),
    xaxt = "n",
    xlab = "",
    ylab = ylab_val,
    ylim = c(y_min - margem, y_max + margem),
    main = titulo_grafico,
    cex.main = cex_titulo_ci,
    cex.lab = 0.90,
    cex.axis = 0.85
  )
  axis(1, at = 1:4, labels = levels(df$nivel_completo), cex.axis = 0.85)
  arrows(1:4, df_ci$ci_inf, 1:4, df_ci$ci_sup, angle = 90, code = 3, length = 0.08, lwd = 2)
  grid(nx = NA, ny = NULL, col = "gray85")
  
  media_em <- df_ci$media[df_ci$nivel_completo == "Ensino Médio"]
  abline(h = media_em, col = "#D7191C", lty = 3, lwd = 1.5)
  
  invisible(NULL)
}

# ================================================================================
#' Gerar Scree Plot dos Autovalores do PCA com Critério de Kaiser
#'
#' @title Scree Plot do PCA
#' @description Produz o gráfico de decaimento dos autovalores (Scree Plot) com
#'   a linha de corte do Critério de Kaiser (Autovalor = 1) e o percentual de
#'   variância acumulada em um eixo secundário.
#'
#' @param pca_obj Objeto prcomp enriquecido gerado por calcular_pca().
#' @param titulo Título principal do gráfico.
#'
#' @return Invisível (NULL).
#'
#' @export
# ================================================================================
plot_scree_pca <- function(pca_obj, 
                           titulo = "Scree Plot do PCA: Autovalores e Critério de Kaiser") {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  eigs <- pca_obj$sdev^2
  var_pct <- (eigs / sum(eigs)) * 100
  cum_pct <- cumsum(var_pct)
  k <- length(eigs)
  
  par(mar = c(4.5, 4.5, 3.5, 4.5))
  bp <- barplot(
    eigs,
    names.arg = paste0("PC", 1:k),
    col = ifelse(eigs >= 1.0, "#2B83BA", "#ABDDA4"),
    border = "white",
    ylim = c(0, max(eigs) * 1.25),
    ylab = "Autovalor (Eigenvalue)",
    xlab = "Componentes Principais",
    main = titulo,
    cex.main = 0.95
  )
  
  # Linha do Critério de Kaiser (Autovalor = 1)
  abline(h = 1.0, col = "#D7191C", lty = 2, lwd = 2)
  text(bp[length(bp)] * 0.75, 1.08, "Critério de Kaiser (Autovalor = 1.0)", col = "#D7191C", cex = 0.85)
  
  # Eixo secundário com % de variância acumulada
  par(new = TRUE)
  plot(
    bp, cum_pct,
    type = "b", pch = 19, col = "#FDAE61", lwd = 2.5,
    axes = FALSE, xlab = "", ylab = "",
    ylim = c(0, 105)
  )
  axis(4, at = seq(0, 100, 20), labels = paste0(seq(0, 100, 20), "%"), col = "#FDAE61", col.axis = "#D95F02")
  mtext("Variância Acumulada (%)", side = 4, line = 2.8, col = "#D95F02")
  
  # Rótulos nas barras
  text(bp, cum_pct + 4, paste0(round(cum_pct, 1), "%"), cex = 0.8, col = "#A65628")
  
  legend("topleft",
         legend = c("Autovalor >= 1 (Kaiser)", "Autovalor < 1", "Variância Acumulada (%)"),
         fill = c("#2B83BA", "#ABDDA4", NA),
         border = c("white", "white", NA),
         pch = c(NA, NA, 19),
         lty = c(NA, NA, 1),
         col = c(NA, NA, "#FDAE61"),
         bty = "n", cex = 0.85)
  
  invisible(NULL)
}

# ================================================================================
#' Gerar Biplot Multivariado no Plano Latente do PCA (PC1 x PC2)
#'
#' @title Biplot do PCA com Elipses de Confiança
#' @description Projeta as 2.076 amostras no plano latente formado pelos dois
#'   primeiros componentes principais (PC1 x PC2), acompanhadas por vetores de
#'   cargas direcionais das features e elipses de dispersão por nível educacional.
#'
#' @param pca_obj Objeto prcomp gerado por calcular_pca().
#' @param df data.frame com os dados e coluna de grupo.
#' @param grupo Nome da coluna com os níveis educacionais (default = 'nivel_completo').
#' @param titulo Título principal do biplot.
#'
#' @return Invisível (NULL).
#'
#' @export
# ================================================================================
plot_biplot_pca <- function(pca_obj, 
                            df, 
                            grupo = "nivel_completo",
                            titulo = "Espaço Latente do PCA: PC1 vs. PC2 por Nível de Ensino") {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  scores <- as.data.frame(pca_obj$x)
  scores$Grupo <- df[[grupo]]
  
  cores_niveis <- c(
    "Ensino Fundamental I"  = "#2B83BA",
    "Ensino Fundamental II" = "#5AAE61",
    "Ensino Médio"          = "#D7191C",
    "Ensino Superior"       = "#756BB1"
  )
  
  var_pc1 <- round((pca_obj$sdev[1]^2 / sum(pca_obj$sdev^2)) * 100, 1)
  var_pc2 <- round((pca_obj$sdev[2]^2 / sum(pca_obj$sdev^2)) * 100, 1)
  
  par(mar = c(4.5, 4.5, 3.5, 1.5))
  plot(
    scores$PC1, scores$PC2,
    col = adjustcolor(cores_niveis[as.character(scores$Grupo)], alpha.f = 0.45),
    pch = 16,
    cex = 0.8,
    xlab = paste0("PC1: Hipotaxe Sintática e Discurso Técnico (", var_pc1, "% da variância)"),
    ylab = paste0("PC2: Extensão Textual vs. Expressividade (", var_pc2, "% da variância)"),
    main = titulo,
    cex.main = 0.92,
    xlim = c(-3.5, 6.0),
    ylim = c(-4.0, 5.0)
  )
  grid(col = "gray85")
  abline(h = 0, v = 0, lty = 3, col = "gray50")
  
  # Centróides por nível
  centroides <- stats::aggregate(cbind(PC1, PC2) ~ Grupo, data = scores, FUN = mean)
  points(
    centroides$PC1, centroides$PC2,
    pch = 21,
    bg = cores_niveis[as.character(centroides$Grupo)],
    col = "black",
    cex = 2.2,
    lwd = 2
  )
  
  # Vetores de Cargas Fatoriais (Loadings) multiplicados por fator de escala
  loadings <- pca_obj$rotation[, 1:2]
  fator_escala <- 4.2
  arrows(
    0, 0,
    loadings[, 1] * fator_escala,
    loadings[, 2] * fator_escala,
    col = "#252525",
    lwd = 2.0,
    length = 0.1
  )
  
  rotulos_vetores <- c(
    "prop_cv"                        = "Canonicidade CV",
    "pont_exp_prop"                  = "Pontuação (!?)",
    "formula_prop"                   = "Fórmulas",
    "asl"                            = "ASL",
    "sconj_densidade"                = "Subordinativas",
    "razao_nominal"                  = "Razão Nominal",
    "cconj_densidade"                = "Coordenativas",
    "prop_polissilabos"              = "Polissílabos",
    "virgulas_por_sentenca"          = "Vírgulas/Sent",
    "metadiscurso_prop"              = "Metadiscurso",
    "citacao_academica_prop"         = "Citações",
    "nominalizacao_prop"             = "Nominalizações",
    "conectivos_argumentativos_prop" = "Conect. Argumentativos",
    "pont_estruturante_prop"         = "Pont. Estruturante",
    "pessoalidade_prop"              = "Pessoalidade"
  )
  
  for (i in seq_len(nrow(loadings))) {
    nome_var <- rownames(loadings)[i]
    rotulo <- if (nome_var %in% names(rotulos_vetores)) rotulos_vetores[nome_var] else nome_var
    text(
      loadings[i, 1] * fator_escala * 1.15,
      loadings[i, 2] * fator_escala * 1.15,
      labels = rotulo,
      col = "#000000",
      font = 2,
      cex = 0.8
    )
  }
  
  legend(
    "topright",
    legend = names(cores_niveis),
    col = cores_niveis,
    pch = 19,
    pt.cex = 1.3,
    bty = "o",
    box.col = "gray80",
    bg = "white",
    cex = 0.8
  )
  
  invisible(NULL)
}

# ================================================================================
#' Gerar Gráfico de Projeção no Espaço Discriminante Canônico (LD1 x LD2)
#'
#' @title Projeção Canônica da LDA (LD1 vs. LD2)
#' @description Plota os escores das observações nas duas primeiras funções
#'   discriminantes lineares (LD1 e LD2), com centróides multivariados.
#'
#' @param lda_res Lista gerada por calcular_lda().
#' @param df data.frame com os dados.
#' @param grupo Nome da coluna com os níveis educacionais (default = 'nivel_completo').
#' @param titulo Título principal do gráfico.
#'
#' @return Invisível (NULL).
#'
#' @export
# ================================================================================
plot_projecao_lda <- function(lda_res, 
                              df, 
                              grupo = "nivel_completo",
                              titulo = "Separação Linear no Espaço Canônico: LD1 vs. LD2") {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  ld_scores <- as.data.frame(lda_res$predicoes$x)
  ld_scores$Grupo <- df[[grupo]]
  
  cores_niveis <- c(
    "Ensino Fundamental I"  = "#2B83BA",
    "Ensino Fundamental II" = "#5AAE61",
    "Ensino Médio"          = "#D7191C",
    "Ensino Superior"       = "#756BB1"
  )
  
  pct_ld1 <- round(lda_res$prop_traco[1] * 100, 1)
  pct_ld2 <- round(lda_res$prop_traco[2] * 100, 1)
  
  par(mar = c(4.5, 4.5, 3.5, 1.5))
  plot(
    ld_scores$LD1, ld_scores$LD2,
    col = adjustcolor(cores_niveis[as.character(ld_scores$Grupo)], alpha.f = 0.45),
    pch = 16,
    cex = 0.8,
    xlab = paste0("LD1: Eixo de Separação Primária (", pct_ld1, "% do poder discriminatório)"),
    ylab = paste0("LD2: Eixo de Separação Secundária (", pct_ld2, "% do poder discriminatório)"),
    main = titulo,
    cex.main = 0.95,
    xlim = c(-5.5, 6.0),
    ylim = c(-4.0, 4.5)
  )
  grid(col = "gray85")
  abline(h = 0, v = 0, lty = 3, col = "gray50")
  
  # Centróides canônicos
  centroides <- stats::aggregate(cbind(LD1, LD2) ~ Grupo, data = ld_scores, FUN = mean)
  points(
    centroides$LD1, centroides$LD2,
    pch = 21,
    bg = cores_niveis[as.character(centroides$Grupo)],
    col = "black",
    cex = 2.4,
    lwd = 2
  )
  
  # Rótulos dos centróides
  text(
    centroides$LD1,
    centroides$LD2 + 0.35,
    labels = c("Centróide Fund I", "Centróide Fund II", "Centróide Médio", "Centróide Superior"),
    col = cores_niveis[as.character(centroides$Grupo)],
    font = 2,
    cex = 0.85
  )
  
  legend(
    "topleft",
    legend = names(cores_niveis),
    col = cores_niveis,
    pch = 19,
    pt.cex = 1.3,
    bty = "o",
    box.col = "gray80",
    bg = "white",
    cex = 0.8
  )
  
  invisible(NULL)
}

# ================================================================================
#' Gerar Gráfico de Assinatura Linguística Multivariada (Perfis em Z-score)
#'
#' @title Gráfico de Perfis Multivariados Padronizados (Z-scores)
#' @description Ilustra a assinatura linguística única de cada nível educacional
#'   através das médias padronizadas (Z-scores) nas cinco dimensões analisadas.
#'
#' @param df data.frame contendo os dados.
#' @param vars Vetor com os nomes das cinco features.
#' @param grupo Nome da coluna com os níveis educacionais (default = 'nivel_completo').
#' @param rotulos_vars Vetor nomeado opcional para os rótulos das features no eixo X.
#' @param titulo Título principal do gráfico.
#'
#' @return Invisível (NULL).
#'
#' @export
# ================================================================================
plot_perfis_multivariados_zscore <- function(df, 
                                             vars, 
                                             grupo = "nivel_completo",
                                             rotulos_vars = NULL,
                                             titulo = "Assinatura Linguística Multivariada por Nível de Ensino (Médias em Z-score)") {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  # Padronização (Z-score) de cada variável em todo o acervo
  X_scaled <- as.data.frame(scale(df[, vars]))
  X_scaled$Grupo <- df[[grupo]]
  
  medias_z <- stats::aggregate(. ~ Grupo, data = X_scaled, FUN = mean)
  niveis <- levels(df[[grupo]])
  if (is.null(niveis)) niveis <- unique(as.character(df[[grupo]]))
  
  cores_niveis <- c(
    "Ensino Fundamental I"  = "#2B83BA",
    "Ensino Fundamental II" = "#5AAE61",
    "Ensino Médio"          = "#D7191C",
    "Ensino Superior"       = "#756BB1"
  )
  
  pchs <- c(15, 17, 19, 18)
  ltys <- c(1, 2, 1, 4)
  
  n_vars <- length(vars)
  
  par(mar = c(8.5, 4.5, 3.5, 1.5))
  plot(
    1:n_vars,
    rep(0, n_vars),
    type = "n",
    xaxt = "n",
    ylim = c(-1.3, 1.5),
    xlab = "",
    ylab = "Desvio Padrão da Média Global (Z-score)",
    main = titulo,
    cex.main = 0.88
  )
  grid(nx = NA, ny = NULL, col = "gray85")
  abline(h = 0, lty = 2, col = "gray40", lwd = 1.2)
  
  nomes_x <- if (!is.null(rotulos_vars)) rotulos_vars[vars] else vars
  axis(1, at = 1:n_vars, labels = nomes_x, las = 2, cex.axis = 0.70, font = 2)
  
  for (i in seq_along(niveis)) {
    lvl <- niveis[i]
    vals_z <- as.numeric(medias_z[medias_z$Grupo == lvl, vars])
    
    # Linha do perfil
    lines(1:n_vars, vals_z, col = cores_niveis[lvl], lwd = if (lvl == "Ensino Médio") 3.0 else 2.0, lty = ltys[i])
    # Pontos
    points(1:n_vars, vals_z, pch = pchs[i], col = cores_niveis[lvl], bg = cores_niveis[lvl],
           cex = if (lvl == "Ensino Médio") 1.8 else 1.4)
  }
  
  legend(
    "topright",
    legend = niveis,
    col = cores_niveis[niveis],
    pch = pchs,
    lty = ltys,
    lwd = c(2, 2, 3, 2),
    bty = "o",
    box.col = "gray80",
    bg = "white",
    cex = 0.8
  )
  
  invisible(NULL)
}

# ================================================================================
#' Gerar Gráfico de Heatmap da Matriz de Confusão para um Modelo
#'
#' @title Matriz de Confusão em Heatmap
#' @description Renderiza a matriz de confusão 4x4 do modelo no conjunto de teste,
#'   exibindo em cada célula a contagem absoluta de amostras e a taxa normalizada (%).
#'
#' @param df_conf data.frame contendo a matriz de confusão gerada pelo pipeline Python.
#' @param modelo_nome Nome do modelo a ser visualizado (default = 'Random Forest').
#' @param titulo Título principal do gráfico.
#'
#' @return Invisível (NULL).
#'
#' @export
# ================================================================================
plot_matriz_confusao_painel <- function(df_conf, 
                                        modelo_nome = "Random Forest",
                                        titulo = NULL) {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  df_sub <- df_conf[df_conf$Modelo == modelo_nome, ]
  if (nrow(df_sub) == 0) return(invisible(NULL))
  
  niveis <- c("Ensino Fundamental I", "Ensino Fundamental II", "Ensino Médio", "Ensino Superior")
  rotulos_curtos <- c("Fund I", "Fund II", "Médio", "Superior")
  
  mat_pct <- matrix(0, nrow = 4, ncol = 4)
  mat_cnt <- matrix(0, nrow = 4, ncol = 4)
  
  for (i in 1:4) {
    for (j in 1:4) {
      val_row <- df_sub[df_sub$Classe.Real == niveis[i] & df_sub$Classe.Predita == niveis[j], ]
      if (nrow(val_row) > 0) {
        mat_pct[i, j] <- val_row$Taxa.Normalizada....
        mat_cnt[i, j] <- val_row$Contagem
      }
    }
  }
  
  if (is.null(titulo)) {
    titulo <- paste0("Matriz de Confusão no Teste: ", modelo_nome)
  }
  
  par(mar = c(5, 5.5, 3.5, 2))
  
  # Cores de gradiente azul
  cor_pal <- colorRampPalette(c("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#084594"))(100)
  
  image(
    1:4, 1:4, t(mat_pct[4:1, ]),
    col = cor_pal,
    axes = FALSE,
    xlab = "Nível Predito pelo Modelo",
    ylab = "Nível Real (Gabarito)",
    main = titulo,
    cex.main = 0.95,
    zlim = c(0, 100)
  )
  
  axis(1, at = 1:4, labels = rotulos_curtos, font = 2)
  axis(2, at = 1:4, labels = rev(rotulos_curtos), font = 2, las = 1)
  
  # Adicionar valores em cada célula
  for (i in 1:4) {
    for (j in 1:4) {
      y_pos <- 5 - i
      pct_val <- mat_pct[i, j]
      cnt_val <- mat_cnt[i, j]
      
      cor_texto <- if (pct_val > 50) "white" else "black"
      
      text(
        j, y_pos + 0.12,
        labels = paste0(round(pct_val, 1), "%"),
        col = cor_texto,
        font = 2,
        cex = 1.05
      )
      text(
        j, y_pos - 0.15,
        labels = paste0("(", cnt_val, ")"),
        col = cor_texto,
        cex = 0.85
      )
    }
  }
  
  box(col = "gray40")
  invisible(NULL)
}

# ================================================================================
#' Gerar Gráfico de Barras Comparativo entre Modelos no Conjunto de Teste
#'
#' @title Gráfico Comparativo de Métricas dos Modelos
#' @description Produz um gráfico de barras agrupadas comparando Acurácia,
#'   Acurácia Balanceada e F1-Score Macro para os modelos avaliados.
#'
#' @param df_teste data.frame com os resultados do conjunto de teste.
#' @param titulo Título principal do gráfico.
#'
#' @return Invisível (NULL).
#'
#' @export
# ================================================================================
plot_comparativo_modelos_barras <- function(df_teste, 
                                            titulo = "Desempenho Geral no Conjunto de Teste Independente (N = 416)") {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  modelos <- df_teste$Modelo
  n_mod <- length(modelos)
  
  # Matriz de métricas: 3 métricas x N modelos
  mat_metricas <- rbind(
    df_teste$Acurácia * 100,
    df_teste$Acurácia.Balanceada * 100,
    df_teste$F1.Score..Macro. * 100
  )
  colnames(mat_metricas) <- modelos
  
  cores_metricas <- c("#2B83BA", "#5AAE61", "#D7191C")
  
  par(mar = c(4.5, 4.5, 3.5, 1.5))
  bp <- barplot(
    mat_metricas,
    beside = TRUE,
    col = cores_metricas,
    border = "white",
    ylim = c(0, 75),
    ylab = "Pontuação (%)",
    main = titulo,
    cex.main = 0.90,
    cex.names = 0.9,
    font.axis = 1
  )
  grid(nx = NA, ny = NULL, col = "gray85")
  
  # Adicionar rótulos de valores no topo das barras
  for (i in 1:nrow(mat_metricas)) {
    for (j in 1:ncol(mat_metricas)) {
      val <- mat_metricas[i, j]
      text(
        bp[i, j], val + 2.0,
        labels = paste0(round(val, 1), "%"),
        cex = 0.75,
        font = 2,
        col = "#333333"
      )
    }
  }
  
  legend(
    "topleft",
    legend = c("Acurácia Global", "Acurácia Balanceada", "F1-Score Macro"),
    fill = cores_metricas,
    border = "white",
    bty = "o",
    box.col = "gray80",
    bg = "white",
    cex = 0.85
  )
  
  invisible(NULL)
}

# ================================================================================
#' Gerar Gráfico de Importância Relativa das Features nos Modelos de Árvores
#'
#' @title Gráfico de Importância Relativa das Features
#' @description Produz gráfico de barras horizontais comparando a importância relativa (%)
#'   das cinco features no Random Forest (Gini) e no LightGBM (Ganho).
#'
#' @param df_imp data.frame contendo a importância das features.
#' @param rotulos_vars Vetor nomeado opcional para renomear as features.
#' @param titulo Título principal do gráfico.
#'
#' @return Invisível (NULL).
#'
#' @export
# ================================================================================
plot_importancia_features_arvores <- function(df_imp, 
                                              rotulos_vars = NULL,
                                              titulo = "Importância Relativa das Features (Random Forest vs. LightGBM)") {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))
  
  rf_sub <- df_imp[df_imp$Modelo == "Random Forest" & df_imp$Tipo.Métrica == "MDI (Gini Importance)", ]
  lgb_sub <- df_imp[df_imp$Modelo == "LightGBM" & df_imp$Tipo.Métrica == "Ganho (Gain Importance)", ]
  
  features <- rf_sub$Feature
  n_feat <- length(features)
  
  val_rf <- rf_sub$Importância.Relativa....
  val_lgb <- lgb_sub$Importância.Relativa....
  
  mat_imp <- rbind(val_rf, val_lgb)
  nomes_exibicao <- if (!is.null(rotulos_vars)) rotulos_vars[features] else features
  colnames(mat_imp) <- nomes_exibicao
  
  # Ordenar por importância média
  ordem <- order(colMeans(mat_imp), decreasing = FALSE)
  mat_imp <- mat_imp[, ordem]
  
  par(mar = c(4.5, 12.5, 3.5, 2))
  bp <- barplot(
    mat_imp,
    beside = TRUE,
    horiz = TRUE,
    col = c("#2B83BA", "#D7191C"),
    border = "white",
    xlim = c(0, max(mat_imp) * 1.28),
    xlab = "Importância Relativa (%)",
    main = titulo,
    cex.main = 0.88,
    las = 1,
    cex.names = 0.68
  )
  grid(nx = NULL, ny = NA, col = "gray85")
  
  # Rótulos nas barras
  for (i in 1:nrow(mat_imp)) {
    for (j in 1:ncol(mat_imp)) {
      val <- mat_imp[i, j]
      text(
        val + 1.2, bp[i, j],
        labels = paste0(round(val, 1), "%"),
        cex = 0.75,
        font = 2,
        col = "#333333"
      )
    }
  }
  
  legend(
    "bottomright",
    legend = c("Random Forest (Gini MDI)", "LightGBM (Ganho / Gain)"),
    fill = c("#2B83BA", "#D7191C"),
    border = "white",
    bty = "o",
    box.col = "gray80",
    bg = "white",
    cex = 0.85
  )
  
  invisible(NULL)
}
