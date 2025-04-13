import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar el dataset
try:
    df = pd.read_csv("data/transporte_data.csv")
except FileNotFoundError:
    raise FileNotFoundError("El archivo 'transporte_masivo_dataset.csv' no se encuentra. Verifica la ruta.")


# Seleccionar características numéricas para el clustering
if not {"num_pasajeros", "distancia_estaciones"}.issubset(df.columns):
    raise ValueError("El dataset no contiene las columnas necesarias: 'num_pasajeros' y 'distancia_estaciones'.")

X = df[["num_pasajeros", "distancia_estaciones"]]


# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df["grupo"] = kmeans.fit_predict(X)

# Visualizar los clusters
plt.scatter(df["num_pasajeros"], df["distancia_estaciones"], c=df["grupo"], cmap="viridis")
plt.xlabel("Número de Pasajeros")
plt.ylabel("Distancia entre Estaciones")
plt.title("Agrupamiento de Estaciones de Transporte Masivo")
plt.show()

# Guardar resultados
df.to_csv("data/transporte_data.csv", index=False)
print("Resultados guardados en 'transporte_masivo_resultados.csv'.")