import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from flask import render_template

def process_clustering(filepath):
    df = pd.read_csv(filepath)
    df['Num_Artists'] = df['Artists'].apply(lambda x: len(str(x).split(',')))
    features = ["Popularity", "Num_Artists"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    inertia = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    elbow_path_opt = os.path.join('static', 'elbow_plot_opt.png')
    plt.figure(figsize=(8,6))
    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Jumlah Cluster (K)')
    plt.ylabel('Inertia')
    plt.title('Metode Elbow untuk Menentukan K Optimal')
    plt.savefig(elbow_path_opt)
    plt.close()

    # Silhouette
    silhouette_scores = []
    silhouette_range = range(2, 10)
    for k in silhouette_range:
        kmeans_sil = KMeans(n_clusters=k, random_state=42)
        labels = kmeans_sil.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    silhouette_path = os.path.join('static', 'silhouette_plot.png')
    plt.figure(figsize=(8,6))
    plt.plot(silhouette_range, silhouette_scores, marker='o', color='orange')
    plt.xlabel('Jumlah Cluster (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Metode Silhouette Score')
    plt.savefig(silhouette_path)
    plt.close()

    # Clustering akhir
    model = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)

    cluster_plot_path = os.path.join('static', 'cluster_plot.png')
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=df["Popularity"],
        y=df["Num_Artists"],
        hue=df["Cluster"],
        palette="viridis"
    )
    plt.title("Clustering Lagu Hindi Populer")
    plt.xlabel("Popularity")
    plt.ylabel("Number of Artists")
    plt.savefig(cluster_plot_path)
    plt.close()

    return render_template('index.html',
                           tables=[df.to_html(classes='data table table-bordered table-striped', header="true")],
                           elbow_plot_opt=elbow_path_opt,
                           silhouette_plot=silhouette_path,
                           cluster_plot=cluster_plot_path)
