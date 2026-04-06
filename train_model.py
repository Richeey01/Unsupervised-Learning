import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib
import os

# Load data
df = pd.read_csv('Country-data.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

features = ['gdpp', 'child_mort', 'income', 'life_expec', 'health']
X = df[features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df['cluster'] = kmeans.labels_




colors = ['#e74c3c', '#2ecc71', '#3498db']
labels = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}

plt.figure(figsize=(10, 6))
for cluster_id in range(3):
    mask = df['cluster'] == cluster_id
    plt.scatter(df[mask]['gdpp'], df[mask]['child_mort'],
                c=colors[cluster_id], label=f'Cluster {cluster_id}',
                alpha=0.75, s=60, edgecolors='white', linewidth=0.5)

plt.xlabel('GDP per Capita (USD)', fontsize=12)
plt.ylabel('Child Mortality (per 1000 births)', fontsize=12)
plt.title('K-Means Clustering of Countries\n(GDP per Capita vs Child Mortality)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_plot.png', dpi=150)
print("Figure saved: cluster_plot.png")


print("\nCluster summary:")
print(df.groupby('cluster')[['gdpp', 'child_mort', 'life_expec', 'income']].mean().round(1))

print("\n10 sample rows:")
print(df[['country', 'gdpp', 'child_mort', 'life_expec', 'income', 'cluster']].head(10).to_string(index=False))



df['score'] = (df['income'] + df['gdpp']) / (df['child_mort'] + 1)

print("\n" + "=" * 55)
print("  BEST COUNTRY PER CLUSTER")
print("  (lowest child mortality + highest income & GDP)")
print("=" * 55)
for cluster_id in range(3):
    group = df[df['cluster'] == cluster_id]
    best = group.loc[group['score'].idxmax()]
    print(f"\nCluster {cluster_id}:")
    print(f"  Country      : {best['country']}")
    print(f"  Child Mort.  : {best['child_mort']} per 1000")
    print(f"  Income       : ${best['income']:,.0f}")
    print(f"  GDP/capita   : ${best['gdpp']:,.0f}")
    print(f"  Life Expect. : {best['life_expec']} years")

print("\n>>> Overall best country across all clusters:")
best_overall = df.loc[df['score'].idxmax()]
print(f"    {best_overall['country']} "
      f"(child_mort={best_overall['child_mort']}, "
      f"income=${best_overall['income']:,.0f}, "
      f"gdpp=${best_overall['gdpp']:,.0f})")

df.drop(columns=['score'], inplace=True)


joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved: kmeans_model.pkl, scaler.pkl")