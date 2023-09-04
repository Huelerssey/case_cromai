import streamlit as st
import numpy as np
import time
from src.data_utility import carregar_modelo
from PIL import Image


# constroi a pagina 1
def pagina1():
    # carrega o modelo
    model = carregar_modelo()

    # titulo da pagina
    st.markdown(
        "<h2 style='text-align: center;'>🎰DogCatNator🎰</h2>", unsafe_allow_html=True
    )

    st.write("---")

    st.write("")
    st.markdown(
        "<h4 style='text-align: center;'>Saudações, meus futuros coleguinhas de trabalho 😄! Recebi a grandiosa missão, como cientista de dados, de desenvolver um algoritmo capaz de identificar gatinhos malvados que adoram destruir poltronas e os adoráveis cachorrinhos que espalham alegria balançando seus rabinhos. Inspirado pelo e-mail que recebi do futuro, que clamava por ajuda, decidi agir. Apresento-lhes minha contribuição para salvar nosso planeta! 🌍🐾</h5>",
        unsafe_allow_html=True,
    )

    # cria 3 colunas
    coluna1, coluna2, coluna3 = st.columns(3)

    # primeira coluna
    with coluna2:
        st.write("")
        st.write("")
        st.markdown(
            "<h5 style='text-align: center;'>🐶 Envie a foto de um Doguinho ou Gatito 🐱</h5>",
            unsafe_allow_html=True,
        )

    # segunda coluna
    with coluna2:
        st.write("")
        uploaded_file = st.file_uploader(
            "Escolha uma imagem...", type=["jpg", "png", "jpeg"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem carregada.", use_column_width=True)
            st.write("")
            progress_bar = st.progress(0)

            # Simulando um processo de classificação
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)

            # Transforma a imagem para o formato que o modelo espera
            img_array = np.array(image.resize((128, 128))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            class_names = ["Gato", "Cachorro"]

            # Mostra a classificação
            st.success(f"Esta imagem é um: {class_names[np.argmax(predictions[0])]}!")

    with coluna2:
        st.write("")
        st.write("")
        st.markdown(
            "<h5 style='text-align: center;'>Veja a mágica acontecer! 🌈🦄</h5>",
            unsafe_allow_html=True,
        )
