# modules/utils.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def prepare_data(path="data/etudiants.csv", target="reussite", test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    df = df.dropna()  # simple clean; adapte si besoin
    
    # Encodage simple
    le = {}
    for col in ["sexe","type_bac","logement","transport"]:
        df[col] = df[col].astype(str)
        le[col] = LabelEncoder()
        df[col] = le[col].fit_transform(df[col])
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test, le
