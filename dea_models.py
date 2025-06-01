# dea_models.py

import numpy as np
from scipy.optimize import linprog

def dea_ccr(inputs, outputs, orientation='output'):
    """
    Modelo DEA CCR (rendimientos constantes a escala).
    Args:
        inputs: np.ndarray, shape (n_dmus, n_inputs)
        outputs: np.ndarray, shape (n_dmus, n_outputs)
        orientation: 'input' or 'output'
    Returns:
        List of efficiency scores
    """
    n_dmus = inputs.shape[0]
    scores = []

    for i in range(n_dmus):
        x0 = inputs[i, :]
        y0 = outputs[i, :]

        # Objetivo
        if orientation == 'output':
            c = np.concatenate([np.zeros(n_dmus), [-1]])
        else:  # input-oriented
            c = np.concatenate([np.zeros(n_dmus), [1]])

        # Restricciones
        A_eq = []
        b_eq = []

        # Restricciones de inputs
        for j in range(inputs.shape[1]):
            row = list(inputs[:, j]) + [0]
            A_eq.append(row)
            b_eq.append(x0[j])

        # Restricciones de outputs
        for j in range(outputs.shape[1]):
            row = list(-outputs[:, j]) + [1] if orientation == 'output' else list(-outputs[:, j]) + [0]
            A_eq.append(row)
            b_eq.append(-y0[j])

        # lambda ≥ 0, θ libre
        bounds = [(0, None)] * n_dmus + [(None, None)]

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            efficiency = result.x[-1] if orientation == 'output' else 1 / result.x[-1]
            scores.append(round(efficiency, 4))
        else:
            scores.append(None)

    return scores

def dea_bcc(inputs, outputs, orientation='output'):
    """
    Modelo DEA BCC (rendimientos variables a escala).
    Args:
        inputs: np.ndarray, shape (n_dmus, n_inputs)
        outputs: np.ndarray, shape (n_dmus, n_outputs)
        orientation: 'input' or 'output'
    Returns:
        List of efficiency scores
    """
    n_dmus = inputs.shape[0]
    scores = []

    for i in range(n_dmus):
        x0 = inputs[i, :]
        y0 = outputs[i, :]

        # Objetivo
        if orientation == 'output':
            c = np.concatenate([np.zeros(n_dmus), [-1], [0]])
        else:
            c = np.concatenate([np.zeros(n_dmus), [1], [0]])

        # Restricciones
        A_eq = []
        b_eq = []

        # Inputs
        for j in range(inputs.shape[1]):
            row = list(inputs[:, j]) + [0, 0]
            A_eq.append(row)
            b_eq.append(x0[j])

        # Outputs
        for j in range(outputs.shape[1]):
            row = list(-outputs[:, j]) + [1 if orientation == 'output' else 0, 0]
            A_eq.append(row)
            b_eq.append(-y0[j])

        # Restricción de suma de lambdas = 1
        A_eq.append([1.0] * n_dmus + [0, 1.0])
        b_eq.append(1.0)

        bounds = [(0, None)] * n_dmus + [(None, None), (None, None)]

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            efficiency = result.x[-2] if orientation == 'output' else 1 / result.x[-2]
            scores.append(round(efficiency, 4))
        else:
            scores.append(None)

    return scores
