# modules/analyse.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_data(path="data/etudiants.csv"):
    return pd.read_csv(path)

def stats_generales(df):
    print("Shape:", df.shape)
    print(df.describe(include='all'))
    print("\nValeurs manquantes par colonne:\n", df.isnull().sum())

def indicateurs(df):
    print("\nTaux de réussite global:")
    print(df['reussite'].value_counts(normalize=True))
    print("\nMoyenne par type de bac:")
    print(df.groupby('type_bac')[['moyenne_sem1','moyenne_sem2']].mean())

def plot_all(df, outdir="output_plots"):
    os.makedirs(outdir, exist_ok=True)
    sns.countplot(x='type_bac', data=df)
    plt.title("Répartition par type de bac")
    plt.savefig(f"{outdir}/type_bac_count.png")
    plt.clf()
    
    sns.boxplot(x='reussite', y='moyenne_sem1', data=df)
    plt.title("Moyenne sem1 vs réussite")
    plt.savefig(f"{outdir}/box_moy1_reussite.png")
    plt.clf()
    
    sns.scatterplot(x='assiduite', y='moyenne_sem1', hue='reussite', data=df)
    plt.title("Assiduité vs Moyenne Sem1")
    plt.savefig(f"{outdir}/assiduite_moy1.png")
    plt.clf()
    
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True)
    plt.title("Heatmap corrélations")
    plt.savefig(f"{outdir}/heatmap_corr.png")
    plt.clf()
    print("Plots sauvegardés dans", outdir)

if __name__ == "__main__":
    df = load_data()
    stats_generales(df)
    indicateurs(df)
    plot_all(df)
