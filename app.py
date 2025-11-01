import gradio as gr
import joblib
from pathlib import Path

# --- Directorio del modelo ---
MODELS_DIR = Path(__file__).parent / "models"
BUNDLE_PATH = MODELS_DIR / "xgb_model.pkl"

def load_model_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError("models/xgb_model.pkl no encontrado. Entrena el modelo primero.")
    return joblib.load(BUNDLE_PATH)

# --- Cargar bundle ---
bundle = joblib.load(BUNDLE_PATH)  # esto es un dict
MODEL = bundle["model"]
VECTORIZER = bundle["vectorizer"]
TARGET_NAMES = bundle.get("target_names", ["no spam", "spam"])

# --- Función de predicción ---
def predict(email_text):
    """
    Recibe el texto de un email y devuelve 'spam' o 'no spam'.
    """
    X = VECTORIZER.transform([email_text])  # convierte texto a features numéricas
    y = MODEL.predict(X)[0]
    return TARGET_NAMES[int(y)]

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

