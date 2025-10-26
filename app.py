import gradio as gr
import numpy as np
from model_utils import load_model_bundle

bundle = load_model_bundle()
MODEL = bundle["model"]
TARGET_NAMES = bundle.get("target_names", ["setosa", "versicolor", "virginica"])

def predict(sepal_length, sepal_width, petal_length, petal_width):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    y = MODEL.predict(X)[0]
    return TARGET_NAMES[int(y)]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="sepal_length"),
        gr.Number(label="sepal_width"),
        gr.Number(label="petal_length"),
        gr.Number(label="petal_width"),
    ],
    outputs=gr.Text(label="prediction"),
    title="ML CI/CT/CD Quickstart (Iris)",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
