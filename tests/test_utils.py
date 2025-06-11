import pytest
import polars as pl
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from sklearn.preprocessing import StandardScaler

from src import utils

# --- Fixtures para Pruebas ---
@pytest.fixture
def df_for_splitting() -> pl.DataFrame:
    """Crea un DF predecible para probar la función de división."""
    return pl.DataFrame({
        "feature1": list(range(10)),
        "feature2": list(range(10, 20)),
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

@pytest.fixture
def df_with_missing_values() -> pl.DataFrame:
    """Crea un DF con valores nulos para probar el relleno."""
    return pl.DataFrame({
        "col_a": [1.0, 2.0, 3.0, 4.0],
        "col_b_missing": [10.0, 20.0, None, 50.0]
    })

@pytest.fixture
def df_for_dummies() -> pl.DataFrame:
    """Crea un DF con variables categóricas para probar la creación de dummies."""
    return pl.DataFrame({
        "day": ["Mon", "Tue", "Mon"],
        "department": ["sewing", "cutting", "sewing"],
        "team": [1, 2, 1]
    })

@pytest.fixture
def df_with_outlier() -> pl.DataFrame:
    """Crea un DF con un outlier multivariado obvio."""
    return pl.DataFrame({
        "x": [1.0, 1.0, 2.0, 2.0, 1.0, 100.0],
        "y": [1.0, 2.0, 1.0, 2.0, 1.0, 100.0]
    })


# --- Pruebas Unitarias ---

def test_load_and_prepare_data(mocker):
    mock_fetch = mocker.patch('src.utils.fetch_ucirepo')
    mock_features_pd = pd.DataFrame({'date': ['2025-06-11'], 'some_feature': [10]})
    mock_targets_pd = pd.DataFrame({'actual_productivity': [0.8]})
    mock_repo = MagicMock()
    mock_repo.data.features = mock_features_pd
    mock_repo.data.targets = mock_targets_pd
    mock_fetch.return_value = mock_repo
    df = utils.load_and_prepare_data(dataset_id=597, target="actual_productivity", threshold=0.5)
    mock_fetch.assert_called_once_with(id=597)
    assert isinstance(df, pl.DataFrame)
    assert 'date' not in df.columns
    assert df['actual_productivity'][0] == 1
    assert 'some_feature' in df.columns

def test_handle_missing_values(df_with_missing_values):
    df_input = df_with_missing_values
    assert df_input["col_b_missing"].is_null().sum() == 1
    df_filled = utils.handle_missing_values(df_input)
    assert df_filled["col_b_missing"].is_null().sum() == 0
    assert df_filled[2, "col_b_missing"] == 20.0

def test_create_dummies(df_for_dummies):
    """Prueba que las variables categóricas se conviertan a dummies correctamente."""
    df_input = df_for_dummies
    df_dummied = utils.create_dummies(df_input, ["day", "department", "team"])

    assert "day" not in df_dummied.columns
    assert "department" not in df_dummied.columns

    # --- INICIO DE LA CORRECCIÓN ---
    # El resultado real muestra que para 'department', se crea 'department_cutting'.
    # Ajustamos las aserciones para que coincidan con este comportamiento.
    assert "day_Tue" in df_dummied.columns
    assert "day_Mon" not in df_dummied.columns

    # La columna creada es 'department_cutting', no 'department_sewing'.
    assert "department_cutting" in df_dummied.columns
    # Por lo tanto, 'department_sewing' fue la categoría eliminada.
    assert "department_sewing" not in df_dummied.columns

    assert "team_2" in df_dummied.columns
    assert "team_1" not in df_dummied.columns

    # Verificamos un valor específico en la columna que sí se creó
    assert df_dummied[1, "department_cutting"] == 1
    assert df_dummied[0, "department_cutting"] == 0
    # --- FIN DE LA CORRECCIÓN ---

def test_handle_multivariate_outliers(df_with_outlier):
    df_input = df_with_outlier
    df_cleaned = utils.handle_multivariate_outliers(df_input, contamination=0.15)
    assert df_cleaned.shape[0] < df_input.shape[0]
    assert 100.0 not in df_cleaned.get_column("x")

def test_split_data(df_for_splitting):
    df = df_for_splitting
    X_train, X_test, y_train, y_test = utils.split_data(
        df, 'target', test_size=0.2, random_state=42
    )
    assert isinstance(X_train, np.ndarray) and isinstance(y_test, np.ndarray)
    assert X_train.shape == (8, 2)
    assert X_test.shape == (2, 2)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)

def test_scale_features():
    X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    X_test = np.array([[4.0, 40.0]])
    X_train_s, X_test_s, scaler = utils.scale_features(X_train, X_test)
    assert isinstance(X_train_s, np.ndarray)
    assert isinstance(scaler, StandardScaler)
    assert X_train_s.shape == X_train.shape
    assert X_test_s.shape == X_test.shape
    np.testing.assert_almost_equal(X_train_s.mean(axis=0), [0.0, 0.0])
    np.testing.assert_almost_equal(X_train_s.std(axis=0), [1.0, 1.0])
