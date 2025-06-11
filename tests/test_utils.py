import pytest
import polars as pl
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

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

# --- Pruebas Unitarias Actualizadas ---
def test_load_and_prepare_data(mocker):
    """
    Prueba la carga de datos simulando la respuesta de fetch_ucirepo.
    Verifica que los dataframes de pandas se unan y procesen correctamente.
    """
    # 1. Simular (mock) la función fetch_ucirepo para que no haga la llamada real
    mock_fetch = mocker.patch('src.utils.fetch_ucirepo')

    # 2. Definir los datos falsos que devolverá la función simulada
    mock_features_pd = pd.DataFrame({'date': ['2025-06-11'], 'some_feature': [10]})
    mock_targets_pd = pd.DataFrame({'actual_productivity': [0.8]})
    mock_repo = MagicMock()
    mock_repo.data.features = mock_features_pd
    mock_repo.data.targets = mock_targets_pd
    mock_fetch.return_value = mock_repo

    # 3. Ejecutar la función que estamos probando
    df = utils.load_and_prepare_data(dataset_id=597, target="actual_productivity", threshold=0.5)

    # 4. Verificar que todo funcionó como se esperaba
    mock_fetch.assert_called_once_with(id=597)  # Se llamó a la función con el ID correcto
    assert isinstance(df, pl.DataFrame)         # El resultado es un DataFrame de Polars
    assert 'date' not in df.columns             # La columna 'date' fue eliminada
    assert df['actual_productivity'][0] == 1    # El target fue binarizado correctamente
    assert 'some_feature' in df.columns         # La feature de prueba está presente

def test_split_data(df_for_splitting):
    """Prueba unitaria para split_data, verificando tipos y dimensiones de salida."""
    df = df_for_splitting
    X_train, X_test, y_train, y_test = utils.split_data(
        df, 'target', test_size=0.2, random_state=42
    )
    assert isinstance(X_train, np.ndarray) and isinstance(y_test, np.ndarray)
    assert X_train.shape == (8, 2)
    assert X_test.shape == (2, 2)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)
