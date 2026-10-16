#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto ALFA-Gen: Extração da Feature de Canonicidade Silábica (CV)
"""

import os
import glob
import re
import zipfile
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Garantir que o tokenizador do NLTK está pronto
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Definição dos conjuntos fonológicos de vogais e consoantes em língua portuguesa
VOGAIS = set("aeiouáéíóúâêôãõàü")
CONSOANTES = set("bcçdfghjklmnpqrstvwxyz")

def classificar_canonicidade(palavra: str) -> bool:
    """
    Classifica se uma palavra é canônica (todas as sílabas do tipo CV).
    Em português, uma palavra tem todas as sílabas estritamente CV se e somente se
    é composta pela alternância estrita de consoante e vogal (CV)+:
    - Inicia com consoante;
    - Termina com vogal;
    - Comprimento par >= 2;
    - Posições pares (0, 2, 4...) são consoantes;
    - Posições ímpares (1, 3, 5...) são vogais.
    
    Exemplos canônicos: cabeça, canela, janela, luta, sopa, menino, boneca.
    Exemplos não canônicos: abajur, lápis, cansado, asa, alheio, porta, prato.
    """
    termo = palavra.lower().strip()
    # Limpar qualquer pontuação agregada
    termo = re.sub(r"[^a-záéíóúâêôãõàüç]", "", termo)
    
    # Palavras com menos de 2 letras não podem ser CV (ex: monossílabos 'a', 'o', 'e')
    # Além disso, para ser (CV)+ o tamanho deve ser necessariamente par
    if len(termo) < 2 or len(termo) % 2 != 0:
        return False
    
    for i, char in enumerate(termo):
        if i % 2 == 0:
            if char not in CONSOANTES:
                return False
        else:
            if char not in VOGAIS:
                return False
    return True

def extrair_nivel(caminho: str) -> tuple:
    """Retorna (nivel_ordem, nivel_simplificado, nivel_completo) a partir do caminho."""
    if "1_Ensino_Fundamental_I" in caminho:
        return (1, "Fundamental I", "Ensino Fundamental I")
    elif "2_Ensino_Fundamental_II" in caminho:
        return (2, "Fundamental II", "Ensino Fundamental II")
    elif "3_Ensino_Medio" in caminho:
        return (3, "Ensino Médio", "Ensino Médio")
    elif "4_Ensino_Superior" in caminho:
        return (4, "Ensino Superior", "Ensino Superior")
    return (0, "Desconhecido", "Desconhecido")

def processar_dataset():
    # 1. Encontrar o arquivo zip
    zips = glob.glob("dados/*.zip")
    if not zips:
        raise FileNotFoundError("Nenhum arquivo .zip encontrado em dados/")
    zip_path = zips[0]
    print(f"[INFO] Processando arquivo zip: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Filtrar arquivos txt válidos
        arquivos_txt = [
            f for f in z.namelist() 
            if not f.startswith("__MACOSX") and f.endswith(".txt") and "dataset_caed/" in f
        ]
        
        print(f"[INFO] Total de textos a processar: {len(arquivos_txt)}")
        
        registros = []
        
        for idx, arq in enumerate(arquivos_txt, start=1):
            if idx % 500 == 0 or idx == len(arquivos_txt):
                print(f"  -> Processando texto {idx}/{len(arquivos_txt)}...")
                
            conteudo_raw = z.read(arq)
            try:
                texto = conteudo_raw.decode("utf-8")
            except UnicodeDecodeError:
                texto = conteudo_raw.decode("latin-1", errors="ignore")
                
            ordem, nivel_simp, nivel_comp = extrair_nivel(arq)
            
            # Tokenização com NLTK
            tokens = word_tokenize(texto, language='portuguese')
            
            # Filtrar apenas tokens alfabéticos (palavras reais, ignorando números e pontuação)
            palavras = [t.lower() for t in tokens if re.search(r"^[a-záéíóúâêôãõàüç]+$", t.lower())]
            
            total_palavras = len(palavras)
            canonicas = [p for p in palavras if classificar_canonicidade(p)]
            nao_canonicas = [p for p in palavras if not classificar_canonicidade(p)]
            
            n_canonicas = len(canonicas)
            n_nao_canonicas = len(nao_canonicas)
            
            prop_canonicas = (n_canonicas / total_palavras) if total_palavras > 0 else 0.0
            pct_canonicas = prop_canonicas * 100.0
            
            registros.append({
                "id_arquivo": os.path.basename(arq),
                "caminho_relativo": arq,
                "nivel_ordem": ordem,
                "nivel": nivel_simp,
                "nivel_completo": nivel_comp,
                "total_palavras": total_palavras,
                "palavras_cv": n_canonicas,
                "palavras_nao_cv": n_nao_canonicas,
                "prop_cv": prop_canonicas,
                "pct_cv": pct_canonicas,
                "amostra_cv": ", ".join(list(set(canonicas))[:5])
            })
            
    df = pd.DataFrame(registros)
    
    # Salvar em CSV
    output_csv = "dados/features_canonicidade.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[SUCESSO] Dataset de features salvo com sucesso em: {output_csv}")
    print(f"Total de registros: {len(df)}")
    
    # Mostrar resumo rápido por nível
    resumo = df.groupby("nivel_completo")[["total_palavras", "palavras_cv", "prop_cv"]].mean()
    print("\nResumo das médias por nível de ensino:")
    print(resumo)
    
    return df

if __name__ == "__main__":
    processar_dataset()
