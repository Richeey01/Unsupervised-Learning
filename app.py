from flask import Flask, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and data
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('Country-data.csv')

features = ['gdpp', 'child_mort', 'income', 'life_expec', 'health']
X = df[features]
X_scaled = scaler.transform(X)
df['cluster'] = kmeans.predict(X_scaled)

# Best country per cluster
df['score'] = (df['income'] + df['gdpp']) / (df['child_mort'] + 1)

cluster_names = {
    0: 'Developing',
    1: 'Developed',
    2: 'Underdeveloped'
}

best_per_cluster = []
for cluster_id in range(3):
    group = df[df['cluster'] == cluster_id]
    best = group.loc[group['score'].idxmax()]
    best_per_cluster.append({
        'cluster': cluster_id,
        'label': cluster_names[cluster_id],
        'country': best['country'],
        'child_mort': best['child_mort'],
        'income': int(best['income']),
        'gdpp': int(best['gdpp']),
        'life_expec': best['life_expec'],
    })

best_overall = df.loc[df['score'].idxmax()]
df.drop(columns=['score'], inplace=True)

# 10 samples
samples = df[['country', 'child_mort', 'income', 'gdpp', 'life_expec', 'health', 'cluster']].head(10).to_dict(orient='records')


@app.route('/')
def index():
    return render_template('index.html',
                           best_per_cluster=best_per_cluster,
                           best_overall=best_overall,
                           cluster_names=cluster_names,
                           samples=samples)


if __name__ == '__main__':
    app.run(debug=True)