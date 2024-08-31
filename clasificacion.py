import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('cuestionario.csv')

# Convertir la variable categórica 'categoria_riesgo' a valores numéricos
label_encoder = LabelEncoder()
df['categoria_riesgo'] = label_encoder.fit_transform(df['categoria_riesgo'])

# Separar las variables predictoras (ansiedad, depresion, estres) y la variable objetivo (categoria_riesgo)
X = df.drop('categoria_riesgo', axis=1)
y = df['categoria_riesgo']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar un modelo de Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar el modelo y mostrar un reporte de clasificación
print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=label_encoder.classes_))
