import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from model_utils import save_model_bundle  # igual que en tu archivo original


# Asegúrate de tener descargados estos recursos de NLTK:
# nltk.download('punkt')
# nltk.download('stopwords')

def transform_text(text):
    ps = PorterStemmer()
    # lowercase
    text = text.lower()
    # tokenization
    text = nltk.word_tokenize(text)
    # eliminar caracteres no alfanuméricos
    y = [i for i in text if i.isalnum()]
    # eliminar stopwords y puntuación
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    # stemming
    y = [ps.stem(i) for i in y]
    return " ".join(y)


def train_and_eval():
    # === 1. Cargar y limpiar datos ===
    df = pd.read_csv("data/spam.csv", encoding="ISO-8859-1")
    df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    df = df.rename(columns={"v1": "Categoria", "v2": "SMS"})
    df = df.drop_duplicates()

    # === 2. Codificar etiquetas ===
    encoder = LabelEncoder()
    df["Categoria"] = encoder.fit_transform(df["Categoria"])

    # === 3. Preprocesar textos ===
    df["transformed_sms"] = df["SMS"].apply(transform_text)

    # === 4. Vectorización ===
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["transformed_sms"]).toarray()
    y = df["Categoria"].values

    # === 5. División de datos ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # === 6. Entrenamiento del modelo ===
    clf = XGBClassifier(objective="binary:logistic", n_estimators=50, seed=123, use_label_encoder=False, eval_metric="logloss")
    clf.fit(X_train, y_train)

    # === 7. Evaluación ===
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "classes": list(encoder.classes_)
    }

    # === 8. Guardar modelo y vectorizador ===
    bundle = {
        "model": clf,
        "vectorizer": tfidf,
        "label_encoder": encoder,
        "target_names": list(encoder.classes_)
    }
    save_model_bundle(bundle)

    # === 9. Imprimir resultados ===
    print("Entrenamiento completado con éxito")
    print("Métricas:", metrics)
    return metrics


if __name__ == "__main__":
    train_and_eval()

