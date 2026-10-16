"""
================================================================================
PROJETO ALFA-GEN: INFERÊNCIA E CLASSIFICAÇÃO COM MODELOS ONNX
================================================================================
Carrega os modelos treinados (LightGBM, Random Forest, Regressão Logística)
e executa a predição probabilística das 4 classes de nível de ensino.
================================================================================
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import onnxruntime as rt
from extrator_features import NOMES_FEATURES_ORDENADAS

MAPA_CLASSES = {
    0: "Ensino Fundamental I",
    1: "Ensino Fundamental II",
    2: "Ensino Médio",
    3: "Ensino Superior"
}

ARQUIVOS_MODELOS = {
    "LightGBM": "modelos/lightgbm.onnx",
    "Random Forest": "modelos/random_forest.onnx",
    "Regressão Logística": "modelos/regressao_logistica.onnx"
}


class ClassificadorONNX:
    def __init__(self, diretorio_modelos: str = "modelos"):
        if not os.path.isabs(diretorio_modelos) and not os.path.exists(diretorio_modelos):
            raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidato = os.path.join(raiz, diretorio_modelos)
            if os.path.exists(candidato):
                diretorio_modelos = candidato
        self.diretorio = diretorio_modelos
        self.sessoes: Dict[str, rt.InferenceSession] = {}
        self._carregar_modelos()

    def _carregar_modelos(self):
        for nome, rel_path in ARQUIVOS_MODELOS.items():
            caminho = os.path.join(self.diretorio, os.path.basename(rel_path))
            if os.path.exists(caminho):
                self.sessoes[nome] = rt.InferenceSession(caminho)

    def obter_modelos_disponiveis(self) -> list:
        return list(self.sessoes.keys())

    def prever(self, features_dict: Dict[str, float], nome_modelo: str = "LightGBM") -> Dict[str, Any]:
        """
        Executa predição do modelo ONNX sobre o dicionário de 15 features.
        Retorna dicionário estruturado com predição, probabilidades e confiança.
        """
        if nome_modelo not in self.sessoes:
            disponiveis = list(self.sessoes.keys())
            if not disponiveis:
                raise FileNotFoundError("Nenhum modelo ONNX foi encontrado em 'modelos/'.")
            nome_modelo = disponiveis[0]

        sess = self.sessoes[nome_modelo]
        in_name = sess.get_inputs()[0].name

        # Montar array (1, 15) rigorosamente na ordem
        vetor = [features_dict[col] for col in NOMES_FEATURES_ORDENADAS]
        x_input = np.array([vetor], dtype=np.float32)

        out = sess.run(None, {in_name: x_input})

        # Classe predita
        pred_raw = out[0]
        if isinstance(pred_raw, (np.ndarray, list)):
            classe_id = int(pred_raw[0])
        else:
            classe_id = int(pred_raw)

        # Probabilidades
        prob_dict = {}
        raw_probs = out[1]
        if isinstance(raw_probs, list) and isinstance(raw_probs[0], dict):
            # Formato [{0: p0, 1: p1, 2: p2, 3: p3}]
            for c_id in range(4):
                prob_dict[MAPA_CLASSES[c_id]] = float(raw_probs[0].get(c_id, 0.0))
        elif isinstance(raw_probs, np.ndarray):
            # Formato array (1, 4)
            for c_id in range(4):
                prob_dict[MAPA_CLASSES[c_id]] = float(raw_probs[0][c_id])
        else:
            # Fallback
            for c_id in range(4):
                prob_dict[MAPA_CLASSES[c_id]] = 1.0 if c_id == classe_id else 0.0

        confianca = prob_dict[MAPA_CLASSES[classe_id]]

        return {
            "modelo": nome_modelo,
            "classe_id": classe_id,
            "classe_nome": MAPA_CLASSES[classe_id],
            "confianca": confianca,
            "probabilidades": prob_dict,
            "vetor_features": features_dict
        }
