import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Any
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

def load_and_prepare_data(dataset_id: int, target: str, threshold: float) -> pl.DataFrame:
    """
    Obtiene un dataset desde UCI ML Repo, lo combina y prepara para el análisis.

    Args:
        dataset_id (int): El ID del dataset en el repositorio UCI.
        target (str): El nombre de la columna objetivo.
        threshold (float): El umbral para binarizar la variable objetivo.

    Returns:
        pl.DataFrame: Un DataFrame de Polars combinado y preprocesado.
    """
    # Obtener el dataset usando la librería oficial
    repo = fetch_ucirepo(id=dataset_id)
    X_pd = repo.data.features
    y_pd = repo.data.targets

    # Convertir a Polars y combinar features y target
    X_pl = pl.from_pandas(X_pd)
    y_pl = pl.from_pandas(y_pd)
    df = pl.concat([X_pl, y_pl], how="horizontal")

    # Realizar el preprocesamiento inicial
    df = df.drop("date")
    df = df.with_columns(
        pl.when(pl.col(target) >= threshold)
        .then(pl.lit(1, dtype=pl.Int32))
        .otherwise(pl.lit(0, dtype=pl.Int32))
        .alias(target)
    )
    return df

# --- El resto de las funciones permanecen sin cambios ---

def handle_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """Rellena valores faltantes en columnas de tipo float con su mediana."""
    for col_name in df.select(pl.col(pl.Float64)).columns:
        if df[col_name].is_null().any():
            median_val = df[col_name].median()
            df = df.with_columns(pl.col(col_name).fill_null(median_val))
    return df

def handle_multivariate_outliers(df: pl.DataFrame, contamination: float = 0.05) -> pl.DataFrame:
    """Detecta y elimina outliers multivariados usando Isolation Forest."""
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    numeric_data_np = df.select(numeric_cols).to_numpy()
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(numeric_data_np)
    df_with_preds = df.with_columns(pl.Series("outlier", outliers))
    return df_with_preds.filter(pl.col("outlier") == 1).drop("outlier")

def create_dummies(df: pl.DataFrame, cat_features: List[str]) -> pl.DataFrame:
    """Convierte variables categóricas a dummies, incluyendo 'team'."""
    df = df.with_columns(pl.col('team').cast(pl.Utf8))
    return df.to_dummies(columns=cat_features, drop_first=True)

def split_data(df: pl.DataFrame, target: str, test_size: float, random_state: int) -> Tuple:
    """Convierte un DataFrame de Polars a NumPy y lo divide para scikit-learn."""
    X = df.drop(target).to_numpy()
    y = df.select(pl.col(target)).to_numpy().ravel()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Escala features (arrays de NumPy) usando StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_and_evaluate_models(models: Dict, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Entrena y evalúa múltiples modelos, retornando sus accuracies."""
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = accuracy_score(y_test, y_pred)
    return scores

def find_best_model(model: Any, params: Dict, X_train: np.ndarray, y_train: np.ndarray, search_type: str = 'grid', n_iter: int = 10) -> Tuple:
    """Realiza búsqueda de hiperparámetros (Grid o Random Search)."""
    if search_type == 'grid':
        search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    else:
        raise ValueError("search_type debe ser 'grid' o 'random'")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
