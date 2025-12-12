import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path

# ======================
# LOAD CONFIG
# ======================
with open("config/config.yaml", "r") as f:
    import yaml
    CONFIG = yaml.safe_load(f)

# Helper: load class names
def get_classes(dataset):
    return CONFIG["datasets"][dataset]["classes"]

# Helper: load model
def load_model(arch, dataset):
    model_path = Path(f"models/{arch}_{dataset}.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Helper: preprocess image
def preprocess(img, dataset):
    # MNIST & Fashion
    if dataset in ["mnist", "fashion"]:
        img = img.convert("L")
        img = img.resize((28, 28))
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, 28 * 28)       # 784
        return arr

    # CIFAR10 - your model was trained on GRAYSCALE (1024 dims)
    elif dataset == "cifar10":
        img = img.convert("L")              # convert to grayscale, not RGB
        img = img.resize((32, 32))
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, 32 * 32)       # 1024
        return arr


# Helper: plotting
def plot_probs(class_names, probs):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_names, probs)
    ax.set_title("Model Confidence Across All Classes")
    ax.set_ylabel("Probability")
    plt.xticks(rotation=45)
    return fig

# =========================
# MAIN PREDICTION FUNCTION
# =========================
def predict(img, model_name, dataset_name):
    classes = get_classes(dataset_name)
    model = load_model(model_name, dataset_name)
    x = preprocess(img, dataset_name)

    logits = model.forward(x)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    probs = probs[0]

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]

    fig = plot_probs(classes, probs)

    return pred_class, fig


# ======================
# GRADIO UI
# ======================
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(["simple_dnn", "medium_dnn", "deep_dnn"], label="Select Model"),
        gr.Dropdown(["mnist", "fashion", "cifar10"], label="Select Dataset")
    ],
    outputs=[
        gr.Textbox(label="Predicted Class"),
        gr.Plot(label="Confidence Chart")
    ],
    title="DNN Image Classifier",
    description="Upload an image and view prediction + confidence plot from your MNIST/Fashion/CIFAR10 DNN models."
)

demo.launch()
