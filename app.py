import streamlit as st
import json
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import pages.separador.pagina_1 as PaginaUm
import pages.separador.pagina_2 as PaginaDois


# configurações da pagina
st.set_page_config(
    page_title='Doguinhos e Gatitos',
    page_icon='🌈',
    layout='wide'
)

#aplicar estilos de css a pagina
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# animações
with open("animacoes/animacao_lottie.json") as source:
    animacao = json.load(source)

# Menu de navegação lateral
with st.sidebar:

    # animação
    st_lottie(animacao, height=200, width=300)

    # badges
    st.markdown("""
    <div style="display: flex; justify-content: space-between;">
        <div>
            <a href="https://github.com/Huelerssey" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" width="100" />
            </a>
        </div>
        <div>
            <a href="https://www.linkedin.com/in/huelerssey-rodrigues-a3145a261/" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" width="100" />
            </a>
        </div>
        <div>
            <a href="https://api.whatsapp.com/send?phone=5584999306130" target="_blank">
                <img src="https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white" width="100" />
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")

    opcao_selecionada = option_menu(
        menu_title="Menu Inicial",
        menu_icon="justify",
        options=["Modelo Deep Learning", "Apresentando Resultados"],
        icons=['code-square', 'rocket-takeoff'],
        default_index=0,
        orientation='vertical',
    )


# Retorna a pagina 1
if opcao_selecionada == "Modelo Deep Learning":
    PaginaUm.pagina1()

# Retorna a pagina 2
elif opcao_selecionada == "Apresentando Resultados":
    PaginaDois.pagina2()
