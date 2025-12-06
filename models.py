# modules/models.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from modules.utils import prepare_data

def evaluer(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred).tolist()
    }

def entrainer_et_tester(path="data/etudiants.csv"):
    X_train, X_test, y_train, y_test, le = prepare_data(path)
    
    results = {}
    
    # 1. Logistic Regression
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)
    p = log.predict(X_test)
    results['logistic'] = evaluer(y_test, p)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    p = rf.predict(X_test)
    results['random_forest'] = evaluer(y_test, p)
    
    # 3. SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    p = svm.predict(X_test)
    results['svm'] = evaluer(y_test, p)
    
    # 4. KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    p = knn.predict(X_test)
    results['knn'] = evaluer(y_test, p)
    
    # Sauvegarde des modèles (au moins le meilleur - ici sauvegarde RF)
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(log, "models/logistic.pkl")
    joblib.dump(svm, "models/svm.pkl")
    joblib.dump(knn, "models/knn.pkl")
    
    # Sauvegarde des encodeurs
    joblib.dump(le, "models/label_encoders.pkl")
    
    return results

if __name__ == "__main__":
    res = entrainer_et_tester()
    import json
    print(json.dumps(res, indent=2))
