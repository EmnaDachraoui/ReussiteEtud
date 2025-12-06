# modules/collect.py
import pandas as pd
import random
import os

def generer_dataset(nb=500, path="data/etudiants.csv"):
    sexes = ["Homme", "Femme"]
    type_bac = ["Sciences experimentales", "Techniques", "Eco", "Lettres", "Informatique"]
    logement = ["Famille", "Foyer", "Location"]
    transport = ["Bus", "Voiture", "Marche"]
    
    rows = []
    for _ in range(nb):
        age = random.randint(18, 25)
        sexe = random.choice(sexes)
        bac = random.choice(type_bac)
        moy_bac = round(random.uniform(9, 18), 2)
        log = random.choice(logement)
        dist = random.randint(0, 50)
        trans = random.choice(transport)
        assid = random.randint(40, 100)
        moodle = random.randint(0, 120)
        moy1 = round(random.uniform(6, 18), 2)
        moy2 = round(random.uniform(6, 18), 2)
        redouble = random.choice([0, 1]) if random.random() < 0.1 else 0
        # cible : réussite si moyenne semestres >= 10
        reussite = 1 if (moy1 + moy2)/2 >= 10 else 0
        
        rows.append({
            "age": age, "sexe": sexe, "type_bac": bac, "moyenne_bac": moy_bac,
            "logement": log, "distance_km": dist, "transport": trans,
            "assiduite": assid, "moodle_connexions": moodle,
            "moyenne_sem1": moy1, "moyenne_sem2": moy2,
            "redouble": redouble, "reussite": reussite
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset généré : {path} ({nb} lignes)")
    return df

if __name__ == "__main__":
    generer_dataset()
