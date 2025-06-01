# data_loader.py

import pandas as pd
import numpy as np

def load_data(file_path, delimiter=','):
    """
    Carga un archivo CSV y devuelve un DataFrame.
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return None

def split_inputs_outputs(df, input_cols, output_cols):
    """
    Separa inputs y outputs desde un DataFrame según las columnas indicadas.
    Args:
        df: DataFrame original
        input_cols: lista de nombres de columnas para inputs
        output_cols: lista de nombres de columnas para outputs
    Returns:
        inputs, outputs: arrays numpy
    """
    try:
        inputs = df[input_cols].to_numpy(dtype=float)
        outputs = df[output_cols].to_numpy(dtype=float)
        return inputs, outputs
    except Exception as e:
        print(f"❌ Error al separar inputs y outputs: {e}")
        return None, None

def validate_input_output_shape(inputs, outputs):
    """
    Verifica que inputs y outputs sean arrays 2D con el mismo número de filas (DMUs).
    """
    if inputs is None or outputs is None:
        return False
    if len(inputs.shape) != 2 or len(outputs.shape) != 2:
        return False
    if inputs.shape[0] != outputs.shape[0]:
        return False
    return True

def get_column_suggestions(df):
    """
    Retorna dos listas de sugerencias automáticas:
    primeras n columnas como inputs, últimas m como outputs.
    """
    total_cols = df.columns.tolist()
    if len(total_cols) < 3:
        return total_cols[:1], total_cols[1:]
    midpoint = len(total_cols) // 2
    return total_cols[:midpoint], total_cols[midpoint:]
