# app/dashboard.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path

@st.cache_resource
def load_models():
    """Charge les modèles entraînés"""
    try:
        # SOLUTION 1: Chemin absolu avec pathlib
        base_path = Path(__file__).parent.parent  # Remonte d'un niveau
        model_path = base_path / "models" / "logistic.pkl"
        encoder_path = base_path / "models" / "label_encoders.pkl"
        
        # SOLUTION 2: Alternative avec os.path
        # base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # model_path = os.path.join(base_path, "models", "logistic.pkl")
        # encoder_path = os.path.join(base_path, "models", "label_encoders.pkl")
        
        # Vérification
        if not model_path.exists():
            st.error(f"Fichier non trouvé: {model_path}")
            # Lister les fichiers disponibles
            models_dir = base_path / "models"
            if models_dir.exists():
                st.info(f"Fichiers disponibles: {list(models_dir.glob('*.pkl'))}")
            return None, None
        
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        
        st.sidebar.success("✅ Modèles chargés avec succès")
        return model, le
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        # Créer un modèle de démo en cas d'erreur
        st.warning("Utilisation d'un modèle de démonstration...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        # Créer un modèle factice
        model = LogisticRegression()
        X_demo = np.array([[20, 12.0, 80]])
        y_demo = np.array([1])
        model.fit(X_demo, y_demo)
        
        # Créer un encodeur factice
        le = {}
        for col in ['sexe', 'type_bac', 'logement', 'transport']:
            encoder = LabelEncoder()
            # Valeurs par défaut
            if col == 'sexe':
                encoder.classes_ = np.array(['Homme', 'Femme'])
            elif col == 'type_bac':
                encoder.classes_ = np.array(['Sciences experimentales', 'Techniques', 'Eco', 'Lettres', 'Informatique'])
            elif col == 'logement':
                encoder.classes_ = np.array(['Famille', 'Foyer', 'Location'])
            elif col == 'transport':
                encoder.classes_ = np.array(['Bus', 'Voiture', 'Marche'])
            le[col] = encoder
        
        return model, le

st.set_page_config(page_title="Prédiction Réussite Étudiante", layout="centered")

st.title("Prédiction de la réussite étudiante")

model, le = load_models()

if model is None:
    st.error("Impossible de charger les modèles. Vérifiez les chemins.")
    st.stop()

# Interface utilisateur
st.sidebar.header("Saisie étudiant")
age = st.sidebar.number_input("Âge", 16, 40, value=20)
sexe = st.sidebar.selectbox("Sexe", ("Homme","Femme"))
type_bac = st.sidebar.selectbox("Type de bac", ("Sciences experimentales","Techniques","Eco","Lettres", "Informatique"))
moyenne_bac = st.sidebar.number_input("Moyenne bac", 0.0, 20.0, value=12.0)
logement = st.sidebar.selectbox("Logement", ("Famille","Foyer","Location"))
distance_km = st.sidebar.number_input("Distance (km)", 0, 200, value=5)
transport = st.sidebar.selectbox("Transport", ("Bus","Voiture","Marche"))
assiduite = st.sidebar.slider("Assiduité (%)", 0, 100, 80)
moodle_conn = st.sidebar.number_input("Connexions Moodle (mois)", 0, 500, 50)
moy1 = st.sidebar.number_input("Moyenne Semestre 1", 0.0, 20.0, 12.0)
moy2 = st.sidebar.number_input("Moyenne Semestre 2", 0.0, 20.0, 12.0)
redouble = st.sidebar.selectbox("Redoublement antérieur", (0,1))

if st.button("Prédire la réussite"):
    try:
        # Vérifier si le dictionnaire d'encodeurs existe
        if isinstance(le, dict):
            # Utiliser les encodeurs comme dans votre code original
            def enc(col, val):
                return int(le[col].transform([val])[0])
        else:
            # Cas où le n'est pas un dictionnaire
            st.error("Format d'encodeur non reconnu")
            def enc(col, val):
                return 0  # Valeur par défaut
        
        # Préparation des données
        x = {
            "age":[age], 
            "sexe":[enc('sexe', sexe)], 
            "type_bac":[enc('type_bac', type_bac)],
            "moyenne_bac":[moyenne_bac], 
            "logement":[enc('logement', logement)],
            "distance_km":[distance_km], 
            "transport":[enc('transport', transport)],
            "assiduite":[assiduite], 
            "moodle_connexions":[moodle_conn],
            "moyenne_sem1":[moy1], 
            "moyenne_sem2":[moy2], 
            "redouble":[redouble]
        }
        
        Xpred = pd.DataFrame(x)
        pred = model.predict(Xpred)[0]
        
        # Afficher la probabilité si disponible
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xpred)[0][1]
        else:
            proba = None
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        with col1:
            if pred == 1:
                st.success("✅ Réussite prédite")
            else:
                st.error("❌ Échec prédit")
        
        with col2:
            if proba is not None:
                st.metric("Probabilité de réussite", f"{proba*100:.1f}%")
                
                # Barre de progression
                st.progress(int(proba * 100))
        
        # Recommandations
    
        st.subheader("📋 Recommandations")
        if pred == 0:
            st.warning("""
            **Actions recommandées :**
            1. Accompagnement pédagogique renforcé
            2. Tutorat personnalisé
            3. Suivi régulier des absences
            4. Soutien psychologique si nécessaire
            """)
        else:
            st.info("""
            **Continuez ainsi !**
            1. Maintenir l'assiduité
            2. Participer aux activités facultatives
            3. Utiliser les ressources Moodle
            """)
            
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        st.info("Vérifiez que les encodeurs correspondent aux données d'entrée")