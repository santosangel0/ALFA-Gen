"""
================================================================================
PROJETO ALFA-GEN: EXTRATOR UNIFICADO DE FEATURES LINGUÍSTICAS (15 FEATURES)
================================================================================
Extrai o vetor exato de 15 atributos morfossintáticos, lexicais e estruturais
necessários para inferência nos modelos ONNX (LightGBM, Random Forest, Regressão Logística).
================================================================================
"""

import os
import re
import unicodedata
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import spacy
import pyphen
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Garantir recursos do NLTK
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Definições Fonológicas para Canonicidade (CV)
VOGAIS = set("aeiouáéíóúâêôãõàü")
CONSOANTES = set("bcçdfghjklmnpqrstvwxyz")

# Termos de Metadiscurso Acadêmico (normalizados)
METADISCURSO_TERMOS = {
    "autor", "autores", "autora", "autoras",
    "estudo", "estudos", "pesquisa", "pesquisas", "pesquisador", "pesquisadores",
    "metodologia", "metodologias", "metodologico", "metodologica", "metodologicos", "metodologicas",
    "dado", "dados", "analise", "analises", "analisar", "analisou", "analisando",
    "conclusao", "conclusoes", "concluir", "conclui-se", "concluiu-se",
    "tabela", "tabelas", "figura", "figuras", "grafico", "graficos", "quadro", "quadros",
    "capitulo", "capitulos", "artigo", "artigos", "secao", "secoes",
    "evidencia", "evidencias", "hipotese", "hipoteses",
    "resultado", "resultados", "investigacao", "investigacoes",
    "conceito", "conceitos", "teoria", "teorias", "teorico", "teorica", "teoricos", "teoricas",
    "bibliografia", "referencia", "referencias", "citacao", "citacoes",
    "observa-se", "constata-se", "verifica-se", "discute-se", "infere-se", "nota-se",
    "proposta", "propostas", "experimento", "experimentos", "amostra", "amostras",
    "cientifico", "cientifica", "cientificos", "cientificas", "academico", "academica"
}

# Regex Citações Acadêmicas
RE_CITACAO_PARENTETICA = re.compile(r"\([A-ZÀ-Ú][a-zà-úA-ZÀ-Ú\s\.,;&\-]+,?\s*(?:19|20)\d{2}[a-z]?\)")
RE_CITACAO_TERMOS = re.compile(r"\b(?:et\s+al\.?|apud|ibid\.?|op\.\s*cit\.?|in\s+memoriam)\b", re.IGNORECASE)
RE_PAGINACAO = re.compile(r"\b(?:p\.|pp\.|págs?\.|p[aá]gina)\s*\d+\b", re.IGNORECASE)
RE_SECAO_NUMERADA = re.compile(r"^(?:[1-9]\.[0-9](?:\.[0-9])?)\s+[A-ZÀ-Ú]", re.MULTILINE)

# Regex Nominalizações
RE_NOMINALIZACAO = re.compile(
    r"\b[a-z]{3,}(?:cao|coes|sao|soes|xao|xoes|mento|mentos|dade|dades|ismo|ismos|encia|encias|ancia|ancias|tude|tudes)\b"
)

# Conectivos Argumentativos Formais
CONECTIVOS_EXPRESSOES = [
    "portanto", "por conseguinte", "por consequencia", "dessa forma", "desse modo",
    "deste modo", "desta forma", "em suma", "assim sendo", "haja vista",
    "diante disso", "ante o exposto", "em virtude disso", "sendo assim",
    "ademais", "outrossim", "alem disso", "alem do mais", "somado a isso",
    "contudo", "todavia", "entretanto", "no entanto", "nao obstante",
    "faz-se necessario", "faz-se mister", "torna-se evidente", "torna-se mister",
    "cabe ao", "cabe a", "e primordial", "e mister", "deve-se", "urgente que"
]

# Pronomes de 1ª pessoa
PRONOMES_1P = {
    "eu", "me", "mim", "comigo", "meu", "meus", "minha", "minhas",
    "nos", "conosco", "nosso", "nossos", "nossa", "nossas"
}

# Ordem canônica rigorosa das 15 features esperadas pelos modelos ONNX
NOMES_FEATURES_ORDENADAS = [
    "prop_cv",
    "pont_exp_prop",
    "formula_prop",
    "asl",
    "sconj_densidade",
    "razao_nominal",
    "cconj_densidade",
    "prop_polissilabos",
    "virgulas_por_sentenca",
    "metadiscurso_prop",
    "citacao_academica_prop",
    "nominalizacao_prop",
    "conectivos_argumentativos_prop",
    "pont_estruturante_prop",
    "pessoalidade_prop"
]

ROTULOS_FEATURES = {
    "prop_cv": "Canonicidade Silábica (prop_cv)",
    "pont_exp_prop": "Pontuação Expressiva (pont_exp_prop)",
    "formula_prop": "Fórmulas Matemáticas (formula_prop)",
    "asl": "Comprimento Sentença (asl)",
    "sconj_densidade": "Conjunções Subordinativas (sconj_densidade)",
    "razao_nominal": "Razão Nominal (razao_nominal)",
    "cconj_densidade": "Conjunções Coordenativas (cconj_densidade)",
    "prop_polissilabos": "Palavras Polissílabas (prop_polissilabos)",
    "virgulas_por_sentenca": "Vírgulas por Sentença (virgulas_por_sentenca)",
    "metadiscurso_prop": "Metadiscurso Acadêmico (metadiscurso_prop)",
    "citacao_academica_prop": "Citações Acadêmicas (citacao_academica_prop)",
    "nominalizacao_prop": "Nominalizações Abstratas (nominalizacao_prop)",
    "conectivos_argumentativos_prop": "Conectivos Argumentativos (conectivos_argumentativos_prop)",
    "pont_estruturante_prop": "Pontuação Estruturante (pont_estruturante_prop)",
    "pessoalidade_prop": "Densidade de Pessoalidade (pessoalidade_prop)"
}


class ExtratorFeatures:
    def __init__(self):
        # Carregar spaCy com componentes necessários e auto-download resiliente
        try:
            self.nlp = spacy.load("pt_core_news_sm", disable=["ner"])
        except OSError:
            import spacy.cli
            spacy.cli.download("pt_core_news_sm")
            self.nlp = spacy.load("pt_core_news_sm", disable=["ner"])
        self.dic_pyphen = pyphen.Pyphen(lang="pt_BR")

    @staticmethod
    def normalizar_texto(texto: str) -> str:
        texto_norm = unicodedata.normalize("NFKD", texto)
        return "".join(c for c in texto_norm if not unicodedata.combining(c)).lower()

    @staticmethod
    def classificar_canonicidade(palavra: str) -> bool:
        termo = palavra.lower().strip()
        termo = re.sub(r"[^a-záéíóúâêôãõàüç]", "", termo)
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

    def extrair_features(self, texto: str) -> Dict[str, float]:
        """
        Extrai as 15 features linguísticas para um texto fornecido.
        Retorna dicionário {nome_feature: valor_float}.
        """
        texto_raw = texto if texto else ""
        texto_norm = self.normalizar_texto(texto_raw)

        # 1. Processamento spaCy
        doc = self.nlp(texto_raw)
        
        # Palavras alfabéticas via spaCy
        tokens_spacy = [t for t in doc if t.is_alpha and not t.is_space]
        total_palavras = max(len(tokens_spacy), 1)

        # Sentenças
        sentencas = list(doc.sents)
        total_sentencas = max(len(sentencas), 1)

        # --- Feature 1: Canonicidade CV ---
        # Calculado sobre tokens alfabéticos
        palavras_cv = [t.text for t in tokens_spacy if self.classificar_canonicidade(t.text)]
        prop_cv = len(palavras_cv) / total_palavras

        # --- Feature 2: Pontuação Expressiva (! e ?) ---
        clusters_exp = re.findall(r"[?!]+", texto_raw)
        n_excl = sum(1 for c in clusters_exp if "!" in c)
        n_interr = sum(1 for c in clusters_exp if "?" in c)
        pont_exp_prop = (n_excl + n_interr) / total_palavras

        # --- Feature 3: Fórmulas Matemáticas ---
        matches_formula = re.findall(r"\b[fF][óÓoO]rmula\w*", texto_raw)
        formula_prop = len(matches_formula) / total_palavras

        # --- Feature 4: ASL (Average Sentence Length via NLTK sent_tokenize) ---
        sentencas_nltk = sent_tokenize(texto_raw, language='portuguese')
        total_sentencas_nltk = max(len(sentencas_nltk), 1)
        asl = total_palavras / total_sentencas_nltk

        # --- Feature 5: Conjunções Subordinativas ---
        sconjs = [t for t in doc if t.pos_ == "SCONJ"]
        sconj_densidade = len(sconjs) / total_palavras

        # --- Feature 6: Razão Nominal (Substantivos / Verbos) ---
        subst = [t for t in doc if t.pos_ == "NOUN"]
        verbos = [t for t in doc if t.pos_ in ("VERB", "AUX")]
        razao_nominal = len(subst) / max(len(verbos), 1)

        # --- Feature 7: Conjunções Coordenativas ---
        cconjs = [t for t in doc if t.pos_ == "CCONJ"]
        cconj_densidade = len(cconjs) / total_palavras

        # --- Feature 8: Proporção de Polissílabos ---
        polissilabos = 0
        for t in tokens_spacy:
            pal_lower = t.text.lower()
            hif = self.dic_pyphen.inserted(pal_lower)
            n_sil = len(hif.split("-")) if hif else 1
            if n_sil >= 4:
                polissilabos += 1
        prop_polissilabos = polissilabos / total_palavras

        # --- Feature 9: Vírgulas por Sentença ---
        virgulas_qtd = texto_raw.count(",")
        virgulas_por_sentenca = virgulas_qtd / total_sentencas

        # --- Feature 10: Metadiscurso Acadêmico ---
        metadiscurso_qtd = 0
        tokens_norm = [self.normalizar_texto(t.text) for t in tokens_spacy]
        for tok in tokens_norm:
            if tok in METADISCURSO_TERMOS:
                metadiscurso_qtd += 1
        metadiscurso_prop = metadiscurso_qtd / total_palavras

        # --- Feature 11: Citações Acadêmicas (ABNT) ---
        cit_parenteses = len(RE_CITACAO_PARENTETICA.findall(texto_raw))
        cit_termos = len(RE_CITACAO_TERMOS.findall(texto_raw))
        cit_paginacao = len(RE_PAGINACAO.findall(texto_raw))
        cit_secao = len(RE_SECAO_NUMERADA.findall(texto_raw))
        anos_parenteses = len(re.findall(r"\((?:19|20)\d{2}[a-z]?\)", texto_raw))
        total_citacoes = cit_parenteses + cit_termos + cit_paginacao + cit_secao + anos_parenteses
        citacao_academica_prop = total_citacoes / total_palavras

        # --- Feature 12: Nominalizações Abstratas ---
        nominalizacoes = [t for t in tokens_norm if RE_NOMINALIZACAO.match(t)]
        nominalizacao_prop = len(nominalizacoes) / total_palavras

        # --- Feature 13: Conectivos Argumentativos ---
        total_conectivos = 0
        for expr in CONECTIVOS_EXPRESSOES:
            padrao = r"\b" + re.escape(expr) + r"\b"
            total_conectivos += len(re.findall(padrao, texto_norm))
        conectivos_argumentativos_prop = total_conectivos / total_palavras

        # --- Feature 14: Pontuação Estruturante Complexa ---
        qtd_ponto_virgula = texto_raw.count(";")
        qtd_dois_pontos = texto_raw.count(":")
        qtd_parenteses = texto_raw.count("(")
        qtd_travessoes = len(re.findall(r"(?:—|--)", texto_raw))
        total_estruturante = qtd_ponto_virgula + qtd_dois_pontos + qtd_parenteses + qtd_travessoes
        pont_estruturante_prop = total_estruturante / total_palavras

        # --- Feature 15: Densidade de Pessoalidade ---
        tokens_1p = [t for t in tokens_norm if t in PRONOMES_1P]
        pessoalidade_prop = len(tokens_1p) / total_palavras

        return {
            "prop_cv": float(prop_cv),
            "pont_exp_prop": float(pont_exp_prop),
            "formula_prop": float(formula_prop),
            "asl": float(asl),
            "sconj_densidade": float(sconj_densidade),
            "razao_nominal": float(razao_nominal),
            "cconj_densidade": float(cconj_densidade),
            "prop_polissilabos": float(prop_polissilabos),
            "virgulas_por_sentenca": float(virgulas_por_sentenca),
            "metadiscurso_prop": float(metadiscurso_prop),
            "citacao_academica_prop": float(citacao_academica_prop),
            "nominalizacao_prop": float(nominalizacao_prop),
            "conectivos_argumentativos_prop": float(conectivos_argumentativos_prop),
            "pont_estruturante_prop": float(pont_estruturante_prop),
            "pessoalidade_prop": float(pessoalidade_prop),
        }

    def para_vetor_numpy(self, features_dict: Dict[str, float]) -> np.ndarray:
        """Converte dicionário em array float32 de formato (1, 15) na ordem canônica."""
        vetor = [features_dict[col] for col in NOMES_FEATURES_ORDENADAS]
        return np.array([vetor], dtype=np.float32)
