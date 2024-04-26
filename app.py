import streamlit as st
import pandas as pd

dados = pd.read_csv('df_clean.csv')
st.write('# Simulador de avaliação de crédito')