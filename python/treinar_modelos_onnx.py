"""
================================================================================
PROJETO ALFA-GEN: TREINAMENTO, AVALIAÇÃO E EXPORTAÇÃO ONNX DOS MODELOS
================================================================================
Modelos:
  1. Regressão Logística Multinomial (com StandardScaler em Pipeline)
  2. Random Forest Classifier
  3. LightGBM Classifier

Protocolo Anti-Vazamento (No Data Leakage):
  - Holdout Test Set estratificado (20% dos dados, N = 416).
  - Validação Cruzada Estratificada 5-Fold (StratifiedKFold) no conjunto de treino (80%, N = 1.660).
  - Pré-processamento (StandardScaler) encapsulado estritamente via Pipeline no Treino de cada fold.
  - Conjunto de teste isolado e avaliado apenas uma única vez ao final.

Exportação ONNX e Paridade Numérica:
  - Exportação dos três modelos treinados para o diretório modelos/*.onnx
  - Verificação de inferência e paridade numérica via onnxruntime.
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance

from lightgbm import LGBMClassifier

import onnx
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMLFloatTensorType


# ================================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ================================================================================
def carregar_dados(caminho_csv: str = "dados/dataset_multivariado_features.csv"):
    df = pd.read_csv(caminho_csv)
    
    features = [
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
    
    # Mapeamento pedagógico ordenado das classes: 0 a 3
    # 0 = Fundamental I, 1 = Fundamental II, 2 = Ensino Médio, 3 = Ensino Superior
    label_map = {
        1: "Ensino Fundamental I",
        2: "Ensino Fundamental II",
        3: "Ensino Médio",
        4: "Ensino Superior"
    }
    
    # Ajustar para 0-indexado
    df["target"] = df["nivel_ordem"] - 1
    
    X = df[features].copy()
    y = df["target"].values
    
    class_names = [label_map[i + 1] for i in range(4)]
    
    return df, X, y, features, class_names


# ================================================================================
# 2. DEFINIÇÃO DOS MODELOS
# ================================================================================
def criar_modelos() -> Dict[str, Any]:
    modelos = {
        "Regressão Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                C=1.0,
                solver="lbfgs"
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.04,
            num_leaves=25,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
    }
    return modelos


# ================================================================================
# 3. VALIDAÇÃO CRUZADA ESTRATIFICADA (5-FOLD)
# ================================================================================
def executar_validacao_cruzada(X_train: pd.DataFrame, y_train: np.ndarray, n_splits: int = 5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    modelos = criar_modelos()
    
    resultados_cv_detalhes = []
    
    print("\n" + "=" * 80)
    print(f"INICIANDO VALIDAÇÃO CRUZADA ESTRATIFICADA ({n_splits}-FOLD) - SEM VAZAMENTO")
    print("=" * 80)
    
    for nome_modelo, modelo in modelos.items():
        print(f"\nAvaliando Modelo: {nome_modelo}...")
        fold = 1
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Ajuste estrito no fold de treino
            modelo.fit(X_tr, y_tr)
            
            y_pred = modelo.predict(X_val)
            y_prob = modelo.predict_proba(X_val)
            
            acc = accuracy_score(y_val, y_pred)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            f1_macro = f1_score(y_val, y_pred, average="macro")
            f1_weighted = f1_score(y_val, y_pred, average="weighted")
            prec_macro = precision_score(y_val, y_pred, average="macro", zero_division=0)
            rec_macro = recall_score(y_val, y_pred, average="macro")
            auc_ovr = roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro")
            loss = log_loss(y_val, y_prob)
            
            resultados_cv_detalhes.append({
                "Modelo": nome_modelo,
                "Fold": fold,
                "Acurácia": acc,
                "Acurácia Balanceada": bal_acc,
                "F1-Score (Macro)": f1_macro,
                "F1-Score (Ponderado)": f1_weighted,
                "Precisão (Macro)": prec_macro,
                "Revocação (Macro)": rec_macro,
                "ROC-AUC (OvR Macro)": auc_ovr,
                "Log Loss": loss
            })
            fold += 1
            
    df_cv = pd.DataFrame(resultados_cv_detalhes)
    
    # Resumo agregado (Média e Desvio Padrão)
    resumo_cv = df_cv.groupby("Modelo").agg({
        "Acurácia": ["mean", "std"],
        "Acurácia Balanceada": ["mean", "std"],
        "F1-Score (Macro)": ["mean", "std"],
        "F1-Score (Ponderado)": ["mean", "std"],
        "Precisão (Macro)": ["mean", "std"],
        "Revocação (Macro)": ["mean", "std"],
        "ROC-AUC (OvR Macro)": ["mean", "std"],
        "Log Loss": ["mean", "std"]
    })
    
    print("\nResumo da Validação Cruzada (Média ± Desvio Padrão):")
    print(resumo_cv)
    
    return df_cv, resumo_cv


# ================================================================================
# 4. TREINAMENTO FINAL E AVALIAÇÃO NO CONJUNTO DE TESTE (HOLDOUT)
# ================================================================================
def treinar_e_avaliar_teste(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list
):
    modelos = criar_modelos()
    
    resultados_teste = []
    metricas_classes = []
    matrizes_confusao = []
    modelos_treinados = {}
    
    print("\n" + "=" * 80)
    print("TREINANDO MODELOS FINAIS NO CONJUNTO DE TREINO (N = 1.660)")
    print("AVALIANDO NO CONJUNTO DE TESTE INDEPENDENTE (N = 416)")
    print("=" * 80)
    
    for nome_modelo, modelo in modelos.items():
        print(f"\nAjustando {nome_modelo}...")
        modelo.fit(X_train, y_train)
        modelos_treinados[nome_modelo] = modelo
        
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro")
        auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        loss = log_loss(y_test, y_prob)
        
        resultados_teste.append({
            "Modelo": nome_modelo,
            "Acurácia": acc,
            "Acurácia Balanceada": bal_acc,
            "F1-Score (Macro)": f1_macro,
            "F1-Score (Ponderado)": f1_weighted,
            "Precisão (Macro)": prec_macro,
            "Revocação (Macro)": rec_macro,
            "ROC-AUC (OvR Macro)": auc_ovr,
            "Log Loss": loss
        })
        
        # Métricas por classe
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        for c_idx, c_name in enumerate(class_names):
            metricas_classes.append({
                "Modelo": nome_modelo,
                "Classe": c_name,
                "Precisão": report[c_name]["precision"],
                "Revocação (Recall)": report[c_name]["recall"],
                "F1-Score": report[c_name]["f1-score"],
                "Suporte (N Teste)": int(report[c_name]["support"])
            })
            
        # Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
        
        for i, c_real in enumerate(class_names):
            for j, c_pred in enumerate(class_names):
                matrizes_confusao.append({
                    "Modelo": nome_modelo,
                    "Classe Real": c_real,
                    "Classe Predita": c_pred,
                    "Contagem": int(cm[i, j]),
                    "Taxa Normalizada (%)": float(cm_norm[i, j] * 100)
                })
                
    df_res_teste = pd.DataFrame(resultados_teste)
    df_classes = pd.DataFrame(metricas_classes)
    df_conf = pd.DataFrame(matrizes_confusao)
    
    print("\nResultados no Conjunto de Teste:")
    print(df_res_teste.to_string(index=False))
    
    return modelos_treinados, df_res_teste, df_classes, df_conf


# ================================================================================
# 5. IMPORTÂNCIA DAS FEATURES
# ================================================================================
def extrair_importancia_features(
    modelos_treinados: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    features: list,
    class_names: list
) -> pd.DataFrame:
    registros = []
    
    # 1. Regressão Logística (Coeficientes por classe)
    lr_pipe = modelos_treinados["Regressão Logística"]
    lr_clf = lr_pipe.named_steps["classifier"]
    coefs = lr_clf.coef_  # shape: (n_classes, n_features)
    
    for i, c_name in enumerate(class_names):
        for j, feat in enumerate(features):
            registros.append({
                "Modelo": "Regressão Logística",
                "Tipo Métrica": f"Coeficiente Padronizado ({c_name})",
                "Feature": feat,
                "Valor": coefs[i, j],
                "Importância Relativa (%)": np.nan
            })
            
    # Magnitude média absoluta dos coeficientes da LR
    lr_mag = np.mean(np.abs(coefs), axis=0)
    lr_mag_pct = (lr_mag / np.sum(lr_mag)) * 100
    for j, feat in enumerate(features):
        registros.append({
            "Modelo": "Regressão Logística",
            "Tipo Métrica": "Magnitude Média Coeficientes (|Beta|)",
            "Feature": feat,
            "Valor": lr_mag[j],
            "Importância Relativa (%)": lr_mag_pct[j]
        })
        
    # 2. Random Forest (MDI Gini & Permutation Importance)
    rf = modelos_treinados["Random Forest"]
    rf_gini = rf.feature_importances_
    rf_gini_pct = (rf_gini / np.sum(rf_gini)) * 100
    
    perm_rf = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    for j, feat in enumerate(features):
        registros.append({
            "Modelo": "Random Forest",
            "Tipo Métrica": "MDI (Gini Importance)",
            "Feature": feat,
            "Valor": rf_gini[j],
            "Importância Relativa (%)": rf_gini_pct[j]
        })
        registros.append({
            "Modelo": "Random Forest",
            "Tipo Métrica": "Permutation Importance (Teste)",
            "Feature": feat,
            "Valor": perm_rf.importances_mean[j],
            "Importância Relativa (%)": np.nan
        })
        
    # 3. LightGBM (Gain & Split Importance)
    lgb = modelos_treinados["LightGBM"]
    lgb_gain = lgb.booster_.feature_importance(importance_type="gain")
    lgb_gain_pct = (lgb_gain / np.sum(lgb_gain)) * 100
    lgb_split = lgb.booster_.feature_importance(importance_type="split")
    lgb_split_pct = (lgb_split / np.sum(lgb_split)) * 100
    
    for j, feat in enumerate(features):
        registros.append({
            "Modelo": "LightGBM",
            "Tipo Métrica": "Ganho (Gain Importance)",
            "Feature": feat,
            "Valor": lgb_gain[j],
            "Importância Relativa (%)": lgb_gain_pct[j]
        })
        registros.append({
            "Modelo": "LightGBM",
            "Tipo Métrica": "Divisões (Split Frequency)",
            "Feature": feat,
            "Valor": float(lgb_split[j]),
            "Importância Relativa (%)": lgb_split_pct[j]
        })
        
    df_imp = pd.DataFrame(registros)
    return df_imp


# ================================================================================
# 6. EXPORTAÇÃO ONNX E VERIFICAÇÃO DE PARIDADE NUMÉRICA
# ================================================================================
def exportar_e_validar_onnx(
    modelos_treinados: Dict[str, Any],
    X_test: pd.DataFrame,
    features: list,
    diretorio_modelos: str = "modelos"
) -> pd.DataFrame:
    os.makedirs(diretorio_modelos, exist_ok=True)
    
    n_features = len(features)
    X_test_arr = X_test.values.astype(np.float32)
    
    validacoes = []
    
    print("\n" + "=" * 80)
    print("EXPORTAÇÃO DOS MODELOS PARA O FORMATO ONNX E TESTES DE PARIDADE")
    print("=" * 80)
    
    # 1. Regressão Logística
    print("\n[1/3] Exportando Regressão Logística para ONNX...")
    caminho_lr = os.path.join(diretorio_modelos, "regressao_logistica.onnx")
    initial_type_sk = [("float_input", SklearnFloatTensorType([None, n_features]))]
    onnx_lr = convert_sklearn(
        modelos_treinados["Regressão Logística"],
        initial_types=initial_type_sk,
        target_opset=15
    )
    onnx.checker.check_model(onnx_lr)
    with open(caminho_lr, "wb") as f:
        f.write(onnx_lr.SerializeToString())
    tamanho_lr_kb = os.path.getsize(caminho_lr) / 1024
    print(f"Salvo em: {caminho_lr} ({tamanho_lr_kb:.1f} KB)")
    
    # Validação ONNX Runtime vs Sklearn
    sess_lr = rt.InferenceSession(caminho_lr)
    in_name_lr = sess_lr.get_inputs()[0].name
    out_lr = sess_lr.run(None, {in_name_lr: X_test_arr})
    pred_onnx_lr = out_lr[0]
    
    # Extrair probabilidades
    if isinstance(out_lr[1], list) and isinstance(out_lr[1][0], dict):
        prob_onnx_lr = np.array([[row[c] for c in range(4)] for row in out_lr[1]])
    else:
        prob_onnx_lr = np.array(out_lr[1])
        
    pred_sk_lr = modelos_treinados["Regressão Logística"].predict(X_test)
    prob_sk_lr = modelos_treinados["Regressão Logística"].predict_proba(X_test)
    
    match_lr = np.mean(pred_onnx_lr == pred_sk_lr) * 100
    max_diff_prob_lr = float(np.max(np.abs(prob_onnx_lr - prob_sk_lr)))
    
    validacoes.append({
        "Modelo": "Regressão Logística",
        "Arquivo ONNX": "modelos/regressao_logistica.onnx",
        "Tamanho (KB)": round(tamanho_lr_kb, 1),
        "Opset ONNX": 15,
        "Acurácia de Paridade (%)": match_lr,
        "Diferença Máxima Probabilidades": max_diff_prob_lr,
        "Status Verificação": "Aprovado (100% Idêntico)" if match_lr == 100.0 and max_diff_prob_lr < 1e-4 else "Atenção"
    })
    
    # 2. Random Forest
    print("\n[2/3] Exportando Random Forest para ONNX...")
    caminho_rf = os.path.join(diretorio_modelos, "random_forest.onnx")
    onnx_rf = convert_sklearn(
        modelos_treinados["Random Forest"],
        initial_types=initial_type_sk,
        target_opset=15
    )
    onnx.checker.check_model(onnx_rf)
    with open(caminho_rf, "wb") as f:
        f.write(onnx_rf.SerializeToString())
    tamanho_rf_kb = os.path.getsize(caminho_rf) / 1024
    print(f"Salvo em: {caminho_rf} ({tamanho_rf_kb:.1f} KB)")
    
    sess_rf = rt.InferenceSession(caminho_rf)
    in_name_rf = sess_rf.get_inputs()[0].name
    out_rf = sess_rf.run(None, {in_name_rf: X_test_arr})
    pred_onnx_rf = out_rf[0]
    
    if isinstance(out_rf[1], list) and isinstance(out_rf[1][0], dict):
        prob_onnx_rf = np.array([[row[c] for c in range(4)] for row in out_rf[1]])
    else:
        prob_onnx_rf = np.array(out_rf[1])
        
    pred_sk_rf = modelos_treinados["Random Forest"].predict(X_test)
    prob_sk_rf = modelos_treinados["Random Forest"].predict_proba(X_test)
    
    match_rf = np.mean(pred_onnx_rf == pred_sk_rf) * 100
    max_diff_prob_rf = float(np.max(np.abs(prob_onnx_rf - prob_sk_rf)))
    
    validacoes.append({
        "Modelo": "Random Forest",
        "Arquivo ONNX": "modelos/random_forest.onnx",
        "Tamanho (KB)": round(tamanho_rf_kb, 1),
        "Opset ONNX": 15,
        "Acurácia de Paridade (%)": match_rf,
        "Diferença Máxima Probabilidades": max_diff_prob_rf,
        "Status Verificação": "Aprovado (100% Idêntico)" if match_rf == 100.0 and max_diff_prob_rf < 1e-4 else "Atenção"
    })
    
    # 3. LightGBM
    print("\n[3/3] Exportando LightGBM para ONNX...")
    caminho_lgb = os.path.join(diretorio_modelos, "lightgbm.onnx")
    initial_type_lgb = [("float_input", OnnxMLFloatTensorType([None, n_features]))]
    onnx_lgb = onnxmltools.convert_lightgbm(
        modelos_treinados["LightGBM"],
        initial_types=initial_type_lgb,
        target_opset=15
    )
    onnx.checker.check_model(onnx_lgb)
    with open(caminho_lgb, "wb") as f:
        f.write(onnx_lgb.SerializeToString())
    tamanho_lgb_kb = os.path.getsize(caminho_lgb) / 1024
    print(f"Salvo em: {caminho_lgb} ({tamanho_lgb_kb:.1f} KB)")
    
    sess_lgb = rt.InferenceSession(caminho_lgb)
    in_name_lgb = sess_lgb.get_inputs()[0].name
    out_lgb = sess_lgb.run(None, {in_name_lgb: X_test_arr})
    pred_onnx_lgb = out_lgb[0]
    
    if isinstance(out_lgb[1], list) and isinstance(out_lgb[1][0], dict):
        prob_onnx_lgb = np.array([[row[c] for c in range(4)] for row in out_lgb[1]])
    else:
        prob_onnx_lgb = np.array(out_lgb[1])
        
    pred_sk_lgb = modelos_treinados["LightGBM"].predict(X_test)
    prob_sk_lgb = modelos_treinados["LightGBM"].predict_proba(X_test)
    
    match_lgb = np.mean(pred_onnx_lgb == pred_sk_lgb) * 100
    max_diff_prob_lgb = float(np.max(np.abs(prob_onnx_lgb - prob_sk_lgb)))
    
    validacoes.append({
        "Modelo": "LightGBM",
        "Arquivo ONNX": "modelos/lightgbm.onnx",
        "Tamanho (KB)": round(tamanho_lgb_kb, 1),
        "Opset ONNX": 15,
        "Acurácia de Paridade (%)": match_lgb,
        "Diferença Máxima Probabilidades": max_diff_prob_lgb,
        "Status Verificação": "Aprovado (100% Idêntico)" if match_lgb == 100.0 and max_diff_prob_lgb < 1e-4 else "Atenção"
    })
    
    df_val = pd.DataFrame(validacoes)
    print("\nTabela de Verificação de Paridade ONNX:")
    print(df_val.to_string(index=False))
    
    return df_val


# ================================================================================
# 7. EXECUÇÃO PRINCIPAL
# ================================================================================
def main():
    print("=" * 80)
    print("PROJETO ALFA-GEN: PIPELINE COMPLETO DE MODELAGEM E EXPORTAÇÃO ONNX")
    print("=" * 80)
    
    # 1. Carregar dados
    df, X, y, features, class_names = carregar_dados()
    print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features.")
    print("Features:", features)
    print("Classes:", class_names)
    
    # 2. Divisão Treino/Teste Estratificada (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\nDivisão realizada:")
    print(f"  Treino (80%): {len(X_train)} amostras")
    print(f"  Teste (20%):  {len(X_test)} amostras (isoladas estritamente)")
    
    # Salvar metadados da divisão
    splits_info = pd.DataFrame({
        "Conjunto": ["Treino", "Teste", "Total"],
        "Total Amostras": [len(X_train), len(X_test), len(df)],
        "Fundamental I": [sum(y_train == 0), sum(y_test == 0), sum(y == 0)],
        "Fundamental II": [sum(y_train == 1), sum(y_test == 1), sum(y == 1)],
        "Ensino Médio": [sum(y_train == 2), sum(y_test == 2), sum(y == 2)],
        "Ensino Superior": [sum(y_train == 3), sum(y_test == 3), sum(y == 3)]
    })
    splits_info.to_csv("dados/splits_info.csv", index=False)
    
    # 3. Validação Cruzada 5-Fold no Treino
    df_cv_detalhes, df_cv_resumo = executar_validacao_cruzada(X_train, y_train, n_splits=5)
    df_cv_detalhes.to_csv("dados/resultados_cv_modelos.csv", index=False)
    
    # 4. Treinamento Final e Avaliação no Teste
    modelos_treinados, df_res_teste, df_classes, df_conf = treinar_e_avaliar_teste(
        X_train, y_train, X_test, y_test, class_names
    )
    df_res_teste.to_csv("dados/resultados_teste_modelos.csv", index=False)
    df_classes.to_csv("dados/metricas_por_classe.csv", index=False)
    df_conf.to_csv("dados/matrizes_confusao_teste.csv", index=False)
    
    # 5. Importância das Features
    df_imp = extrair_importancia_features(modelos_treinados, X_test, y_test, features, class_names)
    df_imp.to_csv("dados/importancia_features.csv", index=False)
    
    # 6. Exportação ONNX e Verificação
    df_onnx = exportar_e_validar_onnx(modelos_treinados, X_test, features)
    df_onnx.to_csv("dados/verificacao_onnx.csv", index=False)
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTADO COM SUCESSO! ARQUIVOS SALVOS EM dados/ E modelos/")
    print("=" * 80)


if __name__ == "__main__":
    main()
