import streamlit as st
from src.data_utility import carregar_dados


# constroi a pagina 2
def pagina2():
    # Carrega os dados da tabela
    df_resultados = carregar_dados()

    # titulo da pagina
    st.title("Tabela de Resultados 📈🚀")

    # Adiciona a coluna de epochs
    epochs = [f"{i}ª" for i in range(1, 11)]
    df_resultados["epochs"] = epochs * 4
    cols = df_resultados.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    df_resultados = df_resultados[cols]

    # Multiplica por 100 e arredonda para 2 casas decimais
    df_resultados["loss"] = df_resultados["loss"].apply(
        lambda x: round(x * 100, 2)
    )
    df_resultados["accuracy"] = df_resultados["accuracy"].apply(
        lambda x: round(x * 100, 2)
    )
    df_resultados["val_loss"] = df_resultados["val_loss"].apply(
        lambda x: round(x * 100, 2)
    )
    df_resultados["val_accuracy"] = df_resultados["val_accuracy"].apply(
        lambda x: round(x * 100, 2)
    )

    # Adiciona o símbolo de porcentagem
    df_resultados["loss"] = df_resultados["loss"].astype(str) + "%"
    df_resultados["accuracy"] = df_resultados["accuracy"].astype(str) + "%"
    df_resultados["val_loss"] = df_resultados["val_loss"].astype(str) + "%"
    df_resultados["val_accuracy"] = df_resultados["val_accuracy"].astype(str) + "%"

    # Exibe a tabela
    st.write(df_resultados.to_html(index=False), unsafe_allow_html=True)
