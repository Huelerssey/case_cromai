import pandas as pd
import streamlit as st
import tensorflow as tf


# função que otimiza o carregamento dos dados
@st.cache_data
def carregar_dados():
    tabela = pd.read_csv("results/resultados.csv")
    return tabela

# função que otimiza o carregamento do modelo
def carregar_modelo():
    modelo =  tf.keras.models.load_model('modelo_dpl/my_model.h5')
    return modelo
