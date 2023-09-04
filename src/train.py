import os
import re
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers import Rescaling, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE


# Controla a versão do código
def get_latest_version(models_path):
    """Busca a última versão do modelo no diretório fornecido."""

    # Lista todos os arquivos no diretório de modelos
    files = os.listdir(models_path)
    
    # Extrai os números das versões dos nomes dos arquivos usando regex
    versions = [int(re.search(r'V(\d+)', file).group(1)) for file in files if re.search(r'V(\d+)', file)]
    
    # Retorna o número da última versão; se nenhuma versão for encontrada, retorna 0
    return max(versions, default=0)

def create_lenet(input_shape):
    """
    Cria uma mini arquitetura lenet

    Args:
        input_shape: Uma lista de três valores inteiros que definem a forma de\
                entrada da rede. Exemplo: [100, 100, 3]

    Returns:
        Um modelo sequencial, seguindo a arquitetura lenet
    """

    # Carrega o modelo MobileNetV2 sem as camadas superiores e com pesos pré-treinados do ImageNet
    baseModel = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

    # Definimos que estamos criando um modelo sequencial
    model = Sequential()

    # Adiciona o modelo base
    model.add(baseModel)

    # Primeira camada do modelo:
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Dropout(0.20))

    # Segunda camada do modelo:
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.20))

    # Terceira camada do modelo:
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.20))

    # Primeira camada fully connected
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # Classificador softmax
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model


if __name__ == "__main__":

    # Caminho para chegar no diretório que contém as imagens de treino
    train_path = "dataset/cats_and_dogs"

    # Diretório onde os modelos serão salvos
    models_path = "models"

    # Tamanho da largura da janela que será utilizada pelo modelo
    width = 128

    # Tamanho da altura da janela que será utilizada pelo modelo
    height = 128

    # Profundidade das janelas utilizadas pelo modelo (RGB: 3, escala de cinza: 1)
    depth = 3

    # Quantidade de classes que o modelo utilizará
    classes = 2

    # Quantidade de épocas (iterações do modelo durante o treinamento)
    epochs = 10

    # Taxa de aprendizado para o otimizador
    init_lr = 1e-4

    # Tamanho dos lotes utilizados por cada época
    batch_size = 32

    # Forma de entrada do modelo
    input_shape = (height, width, depth) 

    # Adiciona um versionador de modelo antes de salvar
    latest_version = get_latest_version(models_path) 

    # Salva o modelo versionado
    save_model = os.path.join(models_path, f"lenet-{{epoch:02d}}-{{accuracy:.3f}}-{{val_accuracy:.3f}}-V{latest_version + 1}.model") 

    # Seleciona o modo de cor em função da variável depth
    color_mode = {1: "grayscale", 3: "rgb"} 

    # Cria o diretório dos modelos, se ele não existir
    os.makedirs(models_path, exist_ok=True)

    # Definição de técnicas de augmentação de dados
    data_augmentation = tf.keras.models.Sequential([
                            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                            tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
    ])

    # Criação do conjunto de dados de treinamento
    train_ds = image_dataset_from_directory(
                            train_path,
                            seed=123,
                            label_mode='categorical',
                            validation_split=0.2,
                            subset="training",
                            color_mode=color_mode[depth],
                            image_size=(height, width),
                            batch_size=batch_size,
                            shuffle=True
    )

    # Criação do conjunto de dados de validação
    val_ds = image_dataset_from_directory(
                            train_path,
                            seed=123,
                            label_mode='categorical',
                            validation_split=0.2,
                            subset="validation",
                            color_mode=color_mode[depth],
                            image_size=(height, width),
                            batch_size=batch_size,
                            shuffle=True
    )

    # Camada de reescalonamento para normalizar os pixels das imagens
    rescaling_layer = Rescaling(1./255)

    # Aplica o reescalonamento aos conjuntos de treinamento e validação
    train_ds = train_ds.map(lambda x, y: (rescaling_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescaling_layer(x), y), num_parallel_calls=AUTOTUNE)

    # Criação do modelo
    model = create_lenet(input_shape)

    # Definição do otimizador
    opt = Adam(lr=init_lr, decay=init_lr / epochs)

    # Compilação do modelo
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    # Início do treinamento
    print("\n training network")

    # Definição de callbacks para o treinamento
    checkpoint1 = ModelCheckpoint(save_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(save_model, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    callbacks_list = [checkpoint1, checkpoint2, early_stopping]

    # Treinamento do modelo
    H = model.fit(train_ds,
                validation_data=val_ds,
                epochs=epochs, 
                verbose=1,
                callbacks=callbacks_list
    )

    # Verifica se já existe um arquivo salvo
    if os.path.exists('results/resultados.csv'):
        # Se o arquivo já existir, ele será lido
        old_results = pd.read_csv('results/resultados.csv')
    else:
        # Se o arquivo não existir, cria um DataFrame vazio com a mesma estrutura
        old_results = pd.DataFrame(columns=['loss', 'accuracy', 'val_loss', 'val_accuracy', 'experiment_description'])

    # cria o novo DataFrame com os resultados atuais
    results_df = pd.DataFrame(H.history)
    results_df['experiment_description'] = f"V{latest_version + 1} Adicao de Data Augmentation para enriquecer o conjunto de treinamento e melhorar a robustez do modelo. Aumento na resolucao da imagem para 128x128 para capturar mais detalhes e caracteristicas. Reducao na taxa de aprendizado para promover uma convergencia mais suave e evitar oscilacoes durante o treinamento. Integracao de Transfer Learning com MobileNetV2 para aproveitar caracteristicas aprendidas em conjuntos de dados maiores e melhorar a eficiencia. Remocao de camadas de MaxPooling devido ao uso de Transfer Learning e para evitar reducoes adicionais nas dimensoes da imagem. Inclusao de callback de Early Stopping para monitorar o treinamento e parar quando nao houver melhorias significativas."

    # Concatena os antigos resultados com os novos
    final_results = pd.concat([old_results, results_df], ignore_index=True)

    # Salva o DataFrame final no arquivo CSV
    final_results.to_csv('results/resultados.csv', index=False)
