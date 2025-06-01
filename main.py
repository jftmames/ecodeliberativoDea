# main.py

import streamlit as st
import pandas as pd
from io import StringIO
from dea_models import dea_ccr, dea_bcc
from data_loader import load_data, split_inputs_outputs, validate_input_output_shape, get_column_suggestions

st.set_page_config(page_title="Simulador DEA", layout="wide")
st.title("📊 Simulador DEA (Análisis Envolvente de Datos)")

# --- Subida de archivo ---
st.sidebar.header("Carga de datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer contenido
    df = load_data(uploaded_file)
    if df is not None:
        st.success("✅ Archivo cargado correctamente")
        st.dataframe(df)

        # Sugerencias automáticas
        suggested_inputs, suggested_outputs = get_column_suggestions(df)

        st.sidebar.markdown("### Selección de variables")
        input_cols = st.sidebar.multiselect("Variables de input", df.columns.tolist(), default=suggested_inputs)
        output_cols = st.sidebar.multiselect("Variables de output", df.columns.tolist(), default=suggested_outputs)

        if input_cols and output_cols:
            inputs, outputs = split_inputs_outputs(df, input_cols, output_cols)
            if validate_input_output_shape(inputs, outputs):
                st.sidebar.markdown("### Modelo DEA")
                model_type = st.sidebar.selectbox("Tipo de modelo", ["CCR (constantes)", "BCC (variables)"])
                orientation = st.sidebar.radio("Orientación", ["input", "output"])

                if st.sidebar.button("Ejecutar análisis DEA"):
                    try:
                        if model_type.startswith("CCR"):
                            efficiency = dea_ccr(inputs, outputs, orientation=orientation)
                        else:
                            efficiency = dea_bcc(inputs, outputs, orientation=orientation)

                        df_result = df.copy()
                        df_result["Eficiencia DEA"] = efficiency
                        st.subheader("📈 Resultados de eficiencia")
                        st.dataframe(df_result)

                        csv_output = df_result.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Descargar resultados", data=csv_output, file_name="resultados_dea.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"❌ Error al ejecutar el modelo DEA: {e}")
            else:
                st.warning("⚠️ Inputs y outputs no válidos para el análisis DEA.")
else:
    st.info("📂 Sube un archivo CSV para comenzar.")
