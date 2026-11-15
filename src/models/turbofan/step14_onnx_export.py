# Step 14: export the trained LSTM to ONNX and check it gives the same answers.
#
# What ONNX is: a standard file format for a trained network. Once exported,
# the model can be run by "ONNX Runtime" (or TensorRT on an NVIDIA GPU, or a
# phone, or a web server) without PyTorch or any of our training code.
# This is the step a real deployment would need.
#
# What this script checks:
#   1. the exported file loads and runs in ONNX Runtime
#   2. its predictions on the 100 test engines match PyTorch's (tiny float
#      differences are normal; anything above ~0.01 cycles would be a bug)
#   3. how fast each one is on this CPU (just for interest - not a benchmark)
#
# Run step 17 first: it saves the tuned LSTM (seed 0) as models/turbofan/lstm_tuned_seed0.pt.
# The tuned model reads 70-cycle sequences; its shape (64 units, 2 layers) is the
# same as step 12's, and dropout only matters during training, so RULModel's
# defaults fit it. (To export the untuned step-12 model instead, set
# MODEL_NAME = "lstm_seed0" and EXPORT_SEQ_LEN = 30.)

import os
import time
import numpy as np
import torch
import onnx
import onnxruntime
from common import load_normalized, get_sensor_columns, score_all, MODELS_DIR, RUL_CAP
from lstm_model import RULModel, make_last_sequences

MODEL_NAME = "lstm_tuned_seed0"
EXPORT_SEQ_LEN = 70
onnx_path = os.path.join(MODELS_DIR, MODEL_NAME + ".onnx")

train_data, test_data = load_normalized()
feature_columns = get_sensor_columns(train_data)
x_test, y_test, test_engines = make_last_sequences(test_data, feature_columns, EXPORT_SEQ_LEN)

model = RULModel(len(feature_columns))
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, MODEL_NAME + ".pt")))
model.eval()

# ---- export: one example input tells the exporter the shape (batch, 70, 16) ----
example_input = torch.zeros(1, EXPORT_SEQ_LEN, len(feature_columns))
torch.onnx.export(model, (example_input,), onnx_path,
                  input_names=["sensor_window"], output_names=["rul_scaled"],
                  dynamic_axes={"sensor_window": {0: "batch"}, "rul_scaled": {0: "batch"}},
                  verbose=False)   # its progress messages contain emoji that crash some Windows consoles
# Newer PyTorch writes the weights to a separate ".onnx.data" file. Fold them
# back into the one .onnx file so the model is a single file to copy around.
onnx_model = onnx.load(onnx_path)
onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
if os.path.exists(onnx_path + ".data"):
    os.remove(onnx_path + ".data")
print("Exported to", onnx_path, "(%.1f KB)" % (os.path.getsize(onnx_path) / 1024))

# ---- PyTorch predictions ----
with torch.no_grad():
    torch_out = model(torch.tensor(x_test)).numpy()

# ---- ONNX Runtime predictions ----
session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
onnx_out = session.run(None, {"sensor_window": x_test})[0]

difference_in_cycles = np.abs(torch_out - onnx_out) * RUL_CAP
print("Largest difference between PyTorch and ONNX: %.6f cycles" % difference_in_cycles.max())

torch_cycles = np.clip(torch_out * RUL_CAP, 0, RUL_CAP)
onnx_cycles = np.clip(onnx_out * RUL_CAP, 0, RUL_CAP)
print("PyTorch scores: RMSE %.3f  MAE %.3f  PHM08 %.2f" % score_all(y_test, torch_cycles))
print("ONNX    scores: RMSE %.3f  MAE %.3f  PHM08 %.2f" % score_all(y_test, onnx_cycles))

# ---- rough speed check: predict all 100 engines, 200 times ----
repeats = 200
start = time.time()
with torch.no_grad():
    for i in range(repeats):
        model(torch.tensor(x_test))
torch_ms = (time.time() - start) / repeats * 1000

start = time.time()
for i in range(repeats):
    session.run(None, {"sensor_window": x_test})
onnx_ms = (time.time() - start) / repeats * 1000
print("Time for 100 engines: PyTorch %.2f ms, ONNX Runtime %.2f ms (CPU, rough)" % (torch_ms, onnx_ms))
