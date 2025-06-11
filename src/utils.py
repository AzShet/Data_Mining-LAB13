import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Any
from ucimlrepo import fetch_ucirepo
import polars.selectors as cs

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

def load_and_prepare_data(dataset_id: int, target: str, threshold: float) -> pl.DataFrame:
    """Obtiene un dataset desde UCI, lo combina y realiza un preprocesamiento inicial.

    Esta función se conecta al Repositorio de Machine Learning de UCI para
    descargar un dataset por su ID. Combina los DataFrames de características (X)
    y objetivo (y) en un único DataFrame de Polars. Finalmente, elimina la
    columna 'date' y binariza la variable objetivo según un umbral.

    Args:
        dataset_id (int): El ID numérico del dataset en el Repositorio de UCI.
        target (str): El nombre de la columna objetivo que será binarizada.
        threshold (float): El punto de corte para la binarización. Los valores
            mayores o iguales al umbral se convierten en 1, y los menores en 0.

    Returns:
        pl.DataFrame: Un DataFrame de Polars que contiene los datos combinados
            y preparados para las siguientes etapas de preprocesamiento.
    """
    repo = fetch_ucirepo(id=dataset_id)
    X_pd = repo.data.features
    y_pd = repo.data.targets

    X_pl = pl.from_pandas(X_pd)
    y_pl = pl.from_pandas(y_pd)
    df = pl.concat([X_pl, y_pl], how="horizontal")

    df = df.drop("date")
    df = df.with_columns(
        pl.when(pl.col(target) >= threshold)
        .then(pl.lit(1, dtype=pl.Int32))
        .otherwise(pl.lit(0, dtype=pl.Int32))
        .alias(target)
    )
    return df

def handle_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """Imputa valores faltantes en columnas de tipo Float64 usando su mediana.

    Itera sobre todas las columnas de tipo Float64 en el DataFrame. Si una
    columna contiene valores nulos, calcula su mediana (ignorando los nulos)
    y la utiliza para rellenar dichos espacios vacíos.

    Args:
        df (pl.DataFrame): El DataFrame de Polars de entrada, posiblemente con
            valores nulos.

    Returns:
        pl.DataFrame: Un DataFrame de Polars sin valores nulos en sus columnas
            de tipo Float64.
    """
    for col_name in df.select(pl.col(pl.Float64)).columns:
        if df[col_name].is_null().any():
            median_val = df[col_name].median()
            df = df.with_columns(pl.col(col_name).fill_null(median_val))
    return df

def handle_multivariate_outliers(df: pl.DataFrame, contamination: float = 0.05) -> pl.DataFrame:
    """Identifica y elimina outliers multivariados usando Isolation Forest.

    Este método aplica el algoritmo Isolation Forest a las columnas numéricas
    del DataFrame para detectar observaciones que son anómalas en un contexto
    multivariado. Las filas identificadas como outliers son eliminadas.

    Args:
        df (pl.DataFrame): El DataFrame de Polars de entrada.
        contamination (float, optional): La proporción esperada de outliers
            en el conjunto de datos. Debe estar entre 0 y 0.5. Por defecto es 0.05.

    Returns:
        pl.DataFrame: Un nuevo DataFrame de Polars sin las filas que fueron
            clasificadas como outliers.
    """
    numeric_cols = df.select(cs.numeric()).columns
    numeric_data_np = df.select(numeric_cols).to_numpy()

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(numeric_data_np)

    df_with_preds = df.with_columns(pl.Series("outlier", outliers))
    return df_with_preds.filter(pl.col("outlier") == 1).drop("outlier")

def create_dummies(df: pl.DataFrame, cat_features: List[str]) -> pl.DataFrame:
    """Convierte columnas categóricas a variables dummy (One-Hot Encoding).

    Toma una lista de nombres de columnas y las transforma en formato numérico
    mediante one-hot encoding. Para evitar la multicolinealidad, se elimina
    la primera categoría de cada variable (`drop_first=True`).

    Args:
        df (pl.DataFrame): El DataFrame de Polars de entrada.
        cat_features (List[str]): Una lista con los nombres de las columnas
            categóricas que se van a convertir.

    Returns:
        pl.DataFrame: Un nuevo DataFrame donde las columnas categóricas han sido
            reemplazadas por sus representaciones numéricas dummy.
    """
    df = df.with_columns(pl.col('team').cast(pl.Utf8))
    return df.to_dummies(columns=cat_features, drop_first=True)

def split_data(df: pl.DataFrame, target: str, test_size: float, random_state: int) -> Tuple:
    """Divide los datos en conjuntos de entrenamiento y prueba.

    Primero, separa las características (X) de la variable objetivo (y).
    Luego, convierte estos DataFrames de Polars en arrays de NumPy y finalmente
    los divide en subconjuntos de entrenamiento y prueba.

    Args:
        df (pl.DataFrame): El DataFrame procesado y listo para la división.
        target (str): El nombre de la columna que contiene la variable objetivo.
        test_size (float): La proporción del dataset que se asignará al conjunto
            de prueba (ej. 0.2 para un 20%).
        random_state (int): La semilla para el generador de números aleatorios,
            asegurando la reproducibilidad de la división.

    Returns:
        Tuple: Una tupla de cuatro arrays de NumPy:
        (X_train, X_test, y_train, y_test).
    """
    X = df.drop(target).to_numpy()
    y = df.select(pl.col(target)).to_numpy().ravel()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Estandariza las características numéricas usando StandardScaler.

    Ajusta el escalador (`fit`) utilizando únicamente los datos de entrenamiento
    para aprender la media y la desviación estándar. Luego, aplica la
    transformación a ambos conjuntos, entrenamiento y prueba.

    Args:
        X_train (np.ndarray): El array de características de entrenamiento.
        X_test (np.ndarray): El array de características de prueba.

    Returns:
        Tuple: Una tupla con los siguientes elementos:
            - X_train_scaled (np.ndarray): Los datos de entrenamiento escalados.
            - X_test_scaled (np.ndarray): Los datos de prueba escalados.
            - scaler (StandardScaler): El objeto scaler ya ajustado.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_and_evaluate_models(models: Dict, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Entrena y evalúa un diccionario de modelos de clasificación.

    Recorre un diccionario de modelos, entrena cada uno con los datos de
    entrenamiento y calcula su métrica de `accuracy` en los datos de prueba.

    Args:
        models (Dict): Un diccionario donde las claves son los nombres de los
            modelos (str) y los valores son las instancias de los modelos de
            scikit-learn sin entrenar.
        X_train (np.ndarray): Array de características de entrenamiento escaladas.
        y_train (np.ndarray): Array de la variable objetivo de entrenamiento.
        X_test (np.ndarray): Array de características de prueba escaladas.
        y_test (np.ndarray): Array de la variable objetivo de prueba.

    Returns:
        Dict[str, float]: Un diccionario con los nombres de los modelos como
            claves y sus puntuaciones de `accuracy` como valores.
    """
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = accuracy_score(y_test, y_pred)
    return scores

def find_best_model(model: Any, params: Dict, X_train: np.ndarray, y_train: np.ndarray, search_type: str = 'grid', n_iter: int = 10) -> Tuple:
    """Encuentra los mejores hiperparámetros para un modelo dado.

    Utiliza validación cruzada para buscar la mejor combinación de
    hiperparámetros, ya sea a través de una búsqueda exhaustiva en cuadrícula
    (`GridSearchCV`) o una búsqueda aleatoria (`RandomizedSearchCV`).

    Args:
        model (Any): Una instancia del estimador de scikit-learn (ej. RandomForestClassifier()).
        params (Dict): Un diccionario definiendo el espacio de búsqueda de
            hiperparámetros.
        X_train (np.ndarray): Array de características de entrenamiento escaladas.
        y_train (np.ndarray): Array de la variable objetivo de entrenamiento.
        search_type (str, optional): El tipo de búsqueda: 'grid' o 'random'.
            Por defecto es 'grid'.
        n_iter (int, optional): Número de combinaciones a probar en la búsqueda
            aleatoria. Se ignora si search_type es 'grid'. Por defecto es 10.

    Raises:
        ValueError: Si `search_type` no es ni 'grid' ni 'random'.

    Returns:
        Tuple: Una tupla con los siguientes elementos:
            - best_estimator_ (Any): El mejor modelo encontrado, ya entrenado.
            - best_params_ (Dict): El diccionario con la mejor combinación de hiperparámetros.
            - best_score_ (float): El score de validación cruzada del mejor modelo.
    """
    if search_type == 'grid':
        search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    else:
        raise ValueError("search_type debe ser 'grid' o 'random'")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
