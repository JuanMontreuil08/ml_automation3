from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from model_utils import save_model_bundle

def train_and_eval():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, yhat)),
        "f1_macro": float(f1_score(yte, yhat, average="macro")),
        "classes": list(iris.target_names)
    }
    bundle = {"model": clf, "target_names": iris.target_names.tolist()}
    save_model_bundle(bundle)
    print("Train OK:", metrics)
    return metrics

if __name__ == "__main__":
    train_and_eval()
