# DogCatNaitor: Classificador de Gatos e Cachorros 🐶🐱

## Descrição

DogCatNaitor é uma solução de deep learning desenvolvida para o desafio técnico da Cromai, com o objetivo de identificar gatinhos malvados destruidores de poltronas e cachorrinhos fofinhos que compartilham a alegria com o rabinho abanando.

O projeto foi estruturado de acordo com os commits do git. Cada versão do código gerou resultados que correspondem ao commit, com a convenção "V" para versão, indo da V1 (modelo inicial) até a versão final do código.

## Estrutura do Projeto

- **Resultados**: Os resultados solicitados pelo desafio estão na pasta "results", que contém um arquivo `.csv` com os valores e descrições das modificações feitas em cada versão do modelo.
- **Script de Treinamento**: O script executável com a versão final do modelo está na pasta "src", no arquivo `train.py`.
  
## Dependências

Para extrair todo o potencial do hardware, utilizou-se a versão `2.9.3` do TensorFlow com suporte a GPU.

Para manter o modelo em produção e apresentar os resultados de maneira mais intuitiva, foi necessário atualizar algumas bibliotecas para suas versões mais recentes. Isso levou à criação de dois arquivos `requirements.txt`:

- **requirements.txt**: Contém as dependências atualizadas necessárias para rodar o projeto em produção.
- **requirements_legacy.txt**: Contém as dependências originais que são compatíveis para executar o script de treinamento do modelo localmente.

## Execução

Para rodar o projeto localmente, você pode seguir os seguintes passos:

1. Clone o repositório.

2. Crie um ambiente virtual:

- **Windows**:

  ```
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **Linux/Mac**:

  ```
  python3 -m venv venv
  source venv/bin/activate
  ```

3. Instale as dependências:

- Para a versão de produção:

  ```
  pip install -r requirements.txt
  ```

- Para rodar o script de treinamento:

  ```
  pip install -r requirements_legacy.txt
  ```

4. Execute o script ou a aplicação:

- Para treinar o modelo localmente:

  ```
  python src/train.py
  ```

- Para executar a aplicação web em localhost:

  ```
  streamlit run app.py
  ```

---

Este projeto foi uma jornada de otimização e adaptação, equilibrando as necessidades de desenvolvimento e produção. Através de iterações cuidadosas e testes, consegui criar um modelo robusto e uma aplicação web para demonstrá-lo de forma eficaz.
