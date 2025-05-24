from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = None
scaler = None
df_clusters = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, scaler, df_clusters

    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

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

            plt.figure(figsize=(8,6))
            plt.plot(k_range, inertia, marker='o')
            plt.xlabel('Jumlah Cluster (K)')
            plt.ylabel('Inertia')
            plt.title('Metode Elbow untuk Menentukan K Optimal')
            elbow_path_opt = os.path.join('static', 'elbow_plot_opt.png')
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

            plt.figure(figsize=(8,6))
            plt.plot(silhouette_range, silhouette_scores, marker='o', color='orange')
            plt.xlabel('Jumlah Cluster (K)')
            plt.ylabel('Silhouette Score')
            plt.title('Metode Silhouette Score')
            silhouette_path = os.path.join('static', 'silhouette_plot.png')
            plt.savefig(silhouette_path)
            plt.close()

            # Clustering akhir
            model = KMeans(n_clusters=3, random_state=42)
            df['Cluster'] = model.fit_predict(X_scaled)
            df_clusters = df.copy()

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
            cluster_plot_path = os.path.join('static', 'cluster_plot.png')
            plt.savefig(cluster_plot_path)
            plt.close()

            return render_template('index.html',
                                   tables=[df.to_html(classes='data table table-bordered table-striped', header="true")],
                                   elbow_plot_opt=elbow_path_opt,
                                   silhouette_plot=silhouette_path,
                                   cluster_plot=cluster_plot_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
