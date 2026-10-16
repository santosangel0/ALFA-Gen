# Teste de Conversão ONNX para Regressão Logística, Random Forest e LightGBM
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

import onnx
import onnxruntime as rt
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools

# Dados sintéticos de teste (5 features, 4 classes)
np.random.seed(42)
X_dummy = np.random.randn(100, 5).astype(np.float32)
y_dummy = np.random.randint(0, 4, size=100)

initial_type = [("float_input", FloatTensorType([None, 5]))]

# 1. Regressão Logística
print("--- Testando Regressão Logística ONNX ---")
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=500, random_state=42))
])
pipe_lr.fit(X_dummy, y_dummy)
onnx_lr = convert_sklearn(pipe_lr, initial_types=initial_type, target_opset=15)
onnx.checker.check_model(onnx_lr)
sess_lr = rt.InferenceSession(onnx_lr.SerializeToString())
input_name = sess_lr.get_inputs()[0].name
pred_onx_lr = sess_lr.run(None, {input_name: X_dummy[:5]})
print("Predições ONNX LR:", pred_onx_lr[0])
print("Predições Sklearn LR:", pipe_lr.predict(X_dummy[:5]))

# 2. Random Forest
print("\n--- Testando Random Forest ONNX ---")
rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
rf.fit(X_dummy, y_dummy)
onnx_rf = convert_sklearn(rf, initial_types=initial_type, target_opset=15)
onnx.checker.check_model(onnx_rf)
sess_rf = rt.InferenceSession(onnx_rf.SerializeToString())
pred_onx_rf = sess_rf.run(None, {sess_rf.get_inputs()[0].name: X_dummy[:5]})
print("Predições ONNX RF:", pred_onx_rf[0])
print("Predições Sklearn RF:", rf.predict(X_dummy[:5]))

# 3. LightGBM
print("\n--- Testando LightGBM ONNX ---")
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMLFloatTensorType
initial_type_lgb = [("float_input", OnnxMLFloatTensorType([None, 5]))]

lgb = LGBMClassifier(n_estimators=20, max_depth=4, random_state=42, verbose=-1)
lgb.fit(X_dummy, y_dummy)
onnx_lgb = onnxmltools.convert_lightgbm(lgb, initial_types=initial_type_lgb, target_opset=15)
onnx.checker.check_model(onnx_lgb)
sess_lgb = rt.InferenceSession(onnx_lgb.SerializeToString())
pred_onx_lgb = sess_lgb.run(None, {sess_lgb.get_inputs()[0].name: X_dummy[:5]})
print("Predições ONNX LGB:", pred_onx_lgb[0])
print("Predições Sklearn LGB:", lgb.predict(X_dummy[:5]))

print("\nTODOS OS 3 MODELOS FORAM CONVERTIDOS E VALIDADOS COM SUCESSO!")
