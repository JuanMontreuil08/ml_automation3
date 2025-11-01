import gradio as gr
import numpy as np
from model_utils import load_model_bundle

bundle = load_model_bundle()
MODEL = bundle
TARGET_NAMES = ["no spam", "spam"]  # etiquetas de salida

# --- Función de predicción ---
def predict(email_text):
    """
    Recibe el texto de un email y devuelve 'spam' o 'no spam'.
    """
    # En tu pipeline, el modelo debe poder recibir texto directamente.
    # Si usas un vectorizer (CountVectorizer, TfidfVectorizer, etc.), 
    # asegúrate de haberlo guardado y cargarlo también.
    if hasattr(MODEL, "predict"):
        X = [email_text]
        y = MODEL.predict(X)[0]
        return TARGET_NAMES[int(y)]
    else:
        return "Error: el modelo no tiene método 'predict'."

# --- Interfaz Gradio ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Escribe el email", lines=8, placeholder="Pega aquí el contenido del correo..."),
    outputs=gr.Textbox(label="Predicción"),
    title="Clasificador de Spam con XGBoost",
    description="Introduce el texto de un correo electrónico y el modelo predecirá si es spam o no.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
