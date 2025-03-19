# Instalar librerías necesarias (Solo ejecutar si no están instaladas)
from IPython.utils import io
import tqdm.notebook

total = 100
with tqdm.notebook.tqdm(total=total) as pbar:
    with io.capture_output() as captured:
        # Instalar rdkit
        !pip -q install rdkit.pypi==2021.9.4
        pbar.update(20)
        # Instalar Pillow
        !pip -q install Pillow
        pbar.update(40)
        # Instalar molplotly
        !pip install molplotly
        pbar.update(60)
        # Instalar jupyter-dash
        !pip install jupyter-dash
        pbar.update(80)
        # Instalar el diseño de aplicación dash
        !pip install dash-bootstrap-components
        pbar.update(100)


# Importar librerías
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Leer bases de datos
datos = "/content/drive/MyDrive/INGER/ETAPA 2024/Version Final/0. Data Set/Concatenadas.csv"
datos = pd.read_csv(datos)


# Agregar una columna de índices
datos['Index'] = datos.index


# Información del dataset
datos.info()


# Separar las las columnas de las variables predictorias (X) de la columna que contiene la variable a predecir (Y)
# PODEMOS UTILIZAR UNICAMENTE COLUMNAS (DESCRIPTORES) DE INTERES PARA ENTRENAR EL MODELO Y NO CAER EN SOBREAJUSTE.
columnas_interes = [1, 3, 5, 6, 7, 8, 20]
X = datos.iloc[:, columnas_interes]
y = datos.iloc[:, 44]



# Separar datos en conjuntos de entrenamiento y prueba

#indicamos que se partan los datos en un 80% para la entrenamiento y 20% para prueba,
#el valor de random_state=42 nos indica que el algoritmo utilizará los mismos datos de entrenamiento y prueba lo que nos dara reproducibilidad

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# Creación del modelo de Árbol de Decisión Binario
arbol = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42)


# Entrenar el modelo
arbol_geros = arbol.fit(X_train, y_train)


# Predecir la respuesta en base a las etiquetas para todos los datos
y_pred_all = arbol_geros.predict(X)




#Predicciones

# Cargar la base de datos COCONUT
coconut_data = pd.read_csv("/content/drive/MyDrive/INGER/ETAPA 2024/DATA /COCONUT_DesMol.csv")

coconut_data.dropna(inplace=True)

coconut_data.info()

# Preprocesar los datos de COCONUT
# Asegúrate de que coconut_data tenga las mismas columnas que usaste para entrenar el modelo
coconut_features = coconut_data.iloc[:, [5, 8, 10, 11, 12, 13, 25]]

# Utilizar el modelo para hacer predicciones
coconut_predictions = arbol_geros.predict(coconut_features)

# Añadir las predicciones a tu DataFrame de COCONUT
coconut_data['Prediccion'] = coconut_predictions
coconut_data


# Analizar los resultados
# Contar cuántos compuestos se predicen como activos o inactivos
print(coconut_data['Prediccion'].value_counts())

# Filtrar los compuestos predichos como activos
compuestos_activos = coconut_data[coconut_data['Prediccion'] == 1]


# Mostrar los primeros compuestos predichos como activos
compuestos_activos.head(10)



#CURVA ROC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Obtener las probabilidades de predicción para la clase positiva
y_pred_proba = arbol_geros.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calcular el área bajo la curva ROC
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Imprimir el AUC
print('AUC: %.2f' % roc_auc)