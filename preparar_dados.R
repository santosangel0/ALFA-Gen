# ==============================================================================
# Projeto ALFA-Gen: Preparação de Amostras para Experimento de Níveis de Ensino
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

preparar_dados_experimento <- function(seed = 42) {
  set.seed(seed)
  
  # Localizar arquivo .zip em dados/
  zip_files <- list.files("dados", pattern = "\\.zip$", full.names = TRUE)
  if (length(zip_files) == 0) {
    stop("Nenhum arquivo .zip encontrado na pasta 'dados/'.")
  }
  zip_path <- zip_files[1]
  cat(sprintf("[INFO] Utilizando arquivo zip: %s\n", zip_path))
  
  # Listar arquivos no zip
  files_df <- utils::unzip(zip_path, list = TRUE)
  
  # Filtrar apenas os .txt de dataset_caed, excluindo metadados __MACOSX e .DS_Store
  txt_files <- files_df$Name[
    !grepl("^__MACOSX", files_df$Name) & 
    grepl("\\.txt$", files_df$Name) &
    grepl("dataset_caed/", files_df$Name)
  ]
  
  cat(sprintf("[INFO] Total de arquivos .txt válidos encontrados: %d\n", length(txt_files)))
  
  # Mapeamento de pastas para rótulos legíveis
  # 1_Ensino_Fundamental_I -> Ensino Fundamental I
  # 2_Ensino_Fundamental_II -> Ensino Fundamental II
  # 3_Ensino_Medio -> Ensino Médio
  # 4_Ensino_Superior -> Ensino Superior
  
  extrair_nivel <- function(path) {
    if (grepl("1_Ensino_Fundamental_I", path)) return("Ensino Fundamental I")
    if (grepl("2_Ensino_Fundamental_II", path)) return("Ensino Fundamental II")
    if (grepl("3_Ensino_Medio", path)) return("Ensino Médio")
    if (grepl("4_Ensino_Superior", path)) return("Ensino Superior")
    return("Outro")
  }
  
  extrair_nivel_simplificado <- function(path) {
    if (grepl("1_Ensino_Fundamental_I", path)) return("Fundamental I")
    if (grepl("2_Ensino_Fundamental_II", path)) return("Fundamental II")
    if (grepl("3_Ensino_Medio", path)) return("Médio")
    if (grepl("4_Ensino_Superior", path)) return("Superior")
    return("Outro")
  }
  
  df_todos <- data.frame(
    path = txt_files,
    id_original = basename(txt_files),
    nivel_completo = sapply(txt_files, extrair_nivel, USE.NAMES = FALSE),
    nivel = sapply(txt_files, extrair_nivel_simplificado, USE.NAMES = FALSE),
    stringsAsFactors = FALSE
  )
  
  # Tabela de frequências
  tab_freq <- df_todos %>%
    group_by(Nivel = nivel_completo) %>%
    summarise(
      `Frequência Absoluta` = n(),
      .groups = "drop"
    ) %>%
    mutate(
      `Frequência Relativa (%)` = round((`Frequência Absoluta` / sum(`Frequência Absoluta`)) * 100, 2)
    )
  
  # Ordem pedagógica correta
  ordem_niveis <- c("Ensino Fundamental I", "Ensino Fundamental II", "Ensino Médio", "Ensino Superior")
  tab_freq <- tab_freq %>%
    arrange(match(Nivel, ordem_niveis))
  
  # Adicionar linha de total
  tab_freq_total <- tab_freq %>%
    bind_rows(
      tibble(
        Nivel = "Total",
        `Frequência Absoluta` = sum(tab_freq$`Frequência Absoluta`),
        `Frequência Relativa (%)` = 100.00
      )
    )
  
  # Amostragem das 10 frases conhecidas por nível (40 frases no total)
  amostras_conhecidas_list <- list()
  for (niv in ordem_niveis) {
    sub_df <- df_todos %>% filter(nivel_completo == niv)
    amostra_niv <- sub_df %>% sample_n(10)
    amostras_conhecidas_list[[niv]] <- amostra_niv
  }
  df_conhecidas <- bind_rows(amostras_conhecidas_list)
  
  # Amostragem das 10 frases de nível desconhecido (sem sobreposição com as conhecidas)
  # Selecionamos uma amostra estratificada (3 Fund I, 2 Fund II, 3 Médio, 2 Superior = 10)
  # e depois embaralhamos completamente para que a ordem não revele o nível!
  df_restante <- df_todos %>% filter(!path %in% df_conhecidas$path)
  
  # Estratificação para garantir representatividade de todos os níveis no teste cego
  amostra_desc_f1 <- df_restante %>% filter(nivel_completo == "Ensino Fundamental I") %>% sample_n(3)
  amostra_desc_f2 <- df_restante %>% filter(nivel_completo == "Ensino Fundamental II") %>% sample_n(2)
  amostra_desc_em <- df_restante %>% filter(nivel_completo == "Ensino Médio") %>% sample_n(3)
  amostra_desc_es <- df_restante %>% filter(nivel_completo == "Ensino Superior") %>% sample_n(2)
  
  df_desconhecidas <- bind_rows(amostra_desc_f1, amostra_desc_f2, amostra_desc_em, amostra_desc_es)
  
  # Embaralhar completamente a ordem das 10 frases desconhecidas
  df_desconhecidas <- df_desconhecidas %>% sample_frac(1)
  
  # Atribuir IDs mascarados e anônimos (ID_01 a ID_10) para não permitir trapaça
  df_desconhecidas$id_anonimo <- sprintf("ID_%02d", 1:nrow(df_desconhecidas))
  
  # Função auxiliar para ler texto do zip
  ler_conteudo <- function(caminhos) {
    sapply(caminhos, function(p) {
      con <- unz(zip_path, p, encoding = "UTF-8")
      linhas <- readLines(con, encoding = "UTF-8", warn = FALSE)
      close(con)
      paste(linhas, collapse = "\n")
    }, USE.NAMES = FALSE)
  }
  
  cat("[INFO] Lendo conteúdos de texto das amostras conhecidas...\n")
  df_conhecidas$texto <- ler_conteudo(df_conhecidas$path)
  df_conhecidas$n_caracteres <- nchar(df_conhecidas$texto)
  df_conhecidas$n_palavras <- sapply(strsplit(df_conhecidas$texto, "\\s+"), length)
  
  cat("[INFO] Lendo conteúdos de texto das amostras desconhecidas...\n")
  df_desconhecidas$texto <- ler_conteudo(df_desconhecidas$path)
  df_desconhecidas$n_caracteres <- nchar(df_desconhecidas$texto)
  df_desconhecidas$n_palavras <- sapply(strsplit(df_desconhecidas$texto, "\\s+"), length)
  
  # Criar gabarito estruturado
  df_gabarito <- df_desconhecidas %>%
    select(
      `ID da Frase` = id_anonimo,
      `Nível Verdadeiro` = nivel_completo,
      `Nível Simplificado` = nivel,
      `ID Original` = id_original,
      `Nº Palavras` = n_palavras,
      `Nº Caracteres` = n_caracteres,
      texto
    ) %>%
    mutate(
      `Início do Texto` = paste0(substr(texto, 1, 60), "...")
    )
  
  resultado <- list(
    zip_path = zip_path,
    frequencias = tab_freq_total,
    frequencias_sem_total = tab_freq,
    amostras_conhecidas = df_conhecidas,
    amostras_desconhecidas = df_desconhecidas,
    gabarito = df_gabarito
  )
  
  # Salvar arquivo .rds com todos os dados processados para carregamento instantâneo
  saveRDS(resultado, "dados/amostras_experimento.rds")
  cat("[SUCESSO] Objeto salvo em: dados/amostras_experimento.rds\n")
  
  # Salvar cópias em CSV para referência fácil
  write_csv(tab_freq_total, "dados/frequencias_niveis.csv")
  write_csv(df_conhecidas %>% select(-texto), "dados/amostras_conhecidas_meta.csv")
  write_csv(df_desconhecidas %>% select(-texto), "dados/amostras_desconhecidas_meta.csv")
  write_csv(df_gabarito %>% select(-texto), "dados/gabarito_respostas.csv")
  cat("[SUCESSO] Tabelas CSV auxiliares salvas em dados/\n")
  
  return(resultado)
}

if (!interactive()) {
  preparar_dados_experimento()
}
