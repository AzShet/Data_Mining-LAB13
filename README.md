# Proyecto 13: Clasificación de Productividad de Empleados con Polars y Scikit-learn

## Descripción General

Este proyecto consiste en un flujo de trabajo completo de Machine Learning para predecir la productividad de los empleados de una fábrica de ropa. El objetivo es determinar si la productividad de un trabajador será "Baja" (menor a 0.5) o "Alta" (mayor o igual a 0.5) basándose en diversos atributos del proceso de producción.

Este laboratorio se distingue por el uso de **Polars**, una librería de manipulación de DataFrames de alto rendimiento, como alternativa a Pandas. Además, se sigue una estructura de proyecto modular y profesional, con el código fuente, notebooks y pruebas unitarias organizados en directorios separados para facilitar su mantenimiento y escalabilidad.

---

## Información del Proyecto

Este repositorio representa el entregable del Laboratorio N°13 del curso de **Data Mining**.

* **Estudiante:** [César Diego Ruelas Flores](https://www.linkedin.com/in/diego-ruelas-flores/)
* **Carrera:** Big Data y Ciencia de Datos
* **Institución:** [TECSUP](https://www.tecsup.edu.pe/)
* **Curso:** Mineria de Datos
* **Fecha:** 11 de Junio de 2025

### Docente

> **[Luis Paraguay Arzapalo](https://github.com/luispar90)**
>
> Ingeniero de sistemas y magíster en Dirección de Tecnologías de la información por la ESAN y La Salle de España. Especialista en Big Data, Business Intelligence, Machine Learning, Cloud, SQL, Modelamiento de Datos y Agilidad.

---

## Características del Proyecto

* **Manipulación de Datos Eficiente:** Uso de la librería **Polars** para todas las operaciones de carga, limpieza y transformación de datos.
* **Carga de Datos Robusta:** Integración con la librería oficial `ucimlrepo` para obtener el conjunto de datos directamente desde el Repositorio de Machine Learning de UCI.
* **Preprocesamiento Exhaustivo:**
    * Imputación de valores faltantes utilizando la mediana.
    * Detección y eliminación de outliers multivariados con el algoritmo `IsolationForest`.
    * Codificación de variables categóricas a numéricas mediante One-Hot Encoding (`to_dummies`).
    * Escalado de características numéricas con `StandardScaler`.
* **Modelado Comparativo:** Entrenamiento y evaluación de 6 modelos de clasificación diferentes para establecer una línea base de rendimiento:
    1.  Regresión Logística
    2.  K-Nearest Neighbors (k-NN)
    3.  Support Vector Machine (SVM)
    4.  Árbol de Clasificación
    5.  Random Forest
    6.  Naive Bayes
* **Optimización de Modelo:** Búsqueda de hiperparámetros con `RandomizedSearchCV` para encontrar la mejor configuración del modelo con mayor rendimiento (Random Forest).
* **Pruebas Unitarias:** Implementación de una suite de pruebas con **Pytest** y **Pytest-Mock** para garantizar la fiabilidad y correctitud de cada función del módulo de utilidades.

---

## Estructura del Repositorio

El proyecto está organizado siguiendo las mejores prácticas para mantener un código limpio y modular.

```
Data_Mining-LAB13/
├── notebooks/
│   └── LAB13-RUELAS.ipynb      # Notebook principal con el flujo de trabajo y visualizaciones.
├── src/
│   ├── __init__.py             # Inicializador del paquete 'src'.
│   └── utils.py                # Módulo con todas las funciones reutilizables.
├── tests/
│   ├── __init__.py             # Inicializador del paquete 'tests'.
│   └── test_utils.py           # Pruebas unitarias para el módulo utils.py.
├── .gitignore                  # Archivo para ignorar archivos no deseados (ej. __pycache__).
├── requirements.txt            # Lista de dependencias del proyecto.
└── README.md                   # Este archivo.
```

---

## Flujo de Trabajo (Workflow)

El proceso completo se detalla en el notebook `notebooks/LAB13-RUELAS.ipynb` y se puede resumir en los siguientes pasos:

### 1. Carga y Preparación de Datos
En lugar de depender de una URL estática, se utiliza la librería `ucimlrepo` para obtener el dataset por su ID (`597`). La librería devuelve las características (`X`) y la variable objetivo (`y`) como DataFrames de Pandas, los cuales son inmediatamente convertidos a Polars y concatenados. Posteriormente, se elimina la columna `date` y se binariza la variable objetivo `actual_productivity` (0 si es < 0.5, 1 si es ≥ 0.5).

### 2. Preprocesamiento de Datos
Se aplica una serie de transformaciones para limpiar y preparar los datos para el modelado:
* **Manejo de Valores Faltantes:** Se imputan los valores nulos en columnas numéricas (como `wip`) con la mediana de la columna, una medida robusta a outliers.
* **Tratamiento de Outliers:** Se utiliza el algoritmo `IsolationForest` para identificar y eliminar filas que se comportan como anomalías multivariadas, mejorando la calidad del conjunto de entrenamiento.
* **Codificación de Variables Categóricas:** Las columnas con texto (`day`, `department`, `quarter`) y `team` se convierten en formato numérico usando One-Hot Encoding (`to_dummies` con `drop_first=True`), creando una representación binaria que los modelos pueden interpretar.
* **Escalado de Características:** Todas las características numéricas son estandarizadas con `StandardScaler`. El escalador se ajusta *únicamente* con los datos de entrenamiento para evitar fuga de datos (data leakage) y luego se aplica para transformar tanto el conjunto de entrenamiento como el de prueba.

### 3. Modelado y Evaluación
Los datos procesados se dividen en un 80% para entrenamiento y un 20% para prueba. Se entrenan 6 modelos de clasificación para obtener una visión comparativa de su rendimiento inicial, utilizando la métrica de **Accuracy**.

### 4. Optimización de Hiperparámetros
El modelo **Random Forest**, por su buen desempeño inicial y robustez, es seleccionado para una optimización más profunda. Se emplea `RandomizedSearchCV` para explorar de manera eficiente un amplio espacio de hiperparámetros (`n_estimators`, `max_depth`, etc.) y encontrar la combinación que maximiza el rendimiento mediante validación cruzada.

---

## Entorno y Ejecución

Para replicar este proyecto, sigue los siguientes pasos.

### 1. Prerrequisitos
* Python 3.9 o superior
* `pip` (manejador de paquetes de Python)
* `git` (para clonar el repositorio)

### 2. Clonar el Repositorio
```bash
git clone <URL_DE_TU_REPOSITORIO_GIT>
cd Data_Mining-LAB13
```

### 3. Crear un Entorno Virtual (Recomendado)
Es una buena práctica aislar las dependencias del proyecto para evitar conflictos.
```bash
# Crear el entorno virtual
python -m venv .venv

# Activar el entorno
# En Windows:
.\.venv\Scripts\activate
# En macOS/Linux:
source .venv/bin/activate
```

### 4. Instalación de Dependencias
Instala todas las librerías necesarias ejecutando el siguiente comando:
```bash
pip install -r requirements.txt
```

### 5. Ejecutar las Pruebas
Para verificar que todo el código funciona correctamente, ejecuta la suite de pruebas desde el directorio raíz del proyecto:
```bash
pytest
```
Deberías ver que las 6 pruebas pasan exitosamente.

### 6. Ejecutar el Notebook
Para explorar el análisis y los resultados, inicia Jupyter Lab o Jupyter Notebook:
```bash
jupyter lab
```
Luego, navega a la carpeta `notebooks/` y abre el archivo `LAB13-RUELAS.ipynb`.

---

## Contenido del `requirements.txt`
Este archivo contiene las librerías exactas para replicar el entorno:
```
ucimlrepo==0.0.7
polars==1.0.0
scikit-learn==1.5.1
pytest==8.4.0
pytest-mock==3.14.1
matplotlib==3.9.1
seaborn==0.13.2
pandas==2.2.2
lxml==5.2.2
```

---

## Autor y Agradecimientos

* **Autor:** César Diego Ruelas Flores
* **Agradecimientos:** Un especial agradecimiento al docente **Luis Paraguay** por su guía y mentoría a lo largo del curso de Data Mining, fomentando el uso de herramientas modernas y buenas prácticas en la ciencia de datos en pro de nuestro desarrollo profesional. Muchas Gracias.
