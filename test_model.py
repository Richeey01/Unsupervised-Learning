import joblib
import numpy as np

# Load saved model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

cluster_names = {
    0: "Developed (high GDP, low child mortality)",
    1: "Underdeveloped (low GDP, high child mortality)",
    2: "Developing (mid GDP, moderate child mortality)"
}

print("=" * 50)
print("  Country Cluster Predictor (KMeans)")
print("=" * 50)
print("Enter country socioeconomic indicators:\n")

def get_float(prompt, example):
    while True:
        try:
            val = input(f"  {prompt} (e.g. {example}): ")
            return float(val)
        except ValueError:
            print("  Please enter a number.")

gdpp       = get_float("GDP per capita (USD)", "5000")
child_mort = get_float("Child mortality (per 1000 births)", "25.0")
income     = get_float("Net income per person (USD)", "8000")
life_expec = get_float("Life expectancy (years)", "70.0")
health     = get_float("Health spending (% of GDP)", "5.0")

X = np.array([[gdpp, child_mort, income, life_expec, health]])
X_scaled = scaler.transform(X)
cluster = kmeans.predict(X_scaled)[0]

print("\n" + "=" * 50)
print(f"  Predicted Cluster: {cluster}")
print(f"  Category: {cluster_names[cluster]}")
print("=" * 50)