import os
import re
import cv2 # OpenCV ou cv2 para tratamento de imagens;
import pandas as pd
import numpy as np # Numpy para trabalharmos com matrizes n-dimensionais
from keras.models import Sequential # Importando modelo sequencial
from keras.layers.convolutional import Conv2D, MaxPooling2D # Camada de convolução e max pooling
from keras.layers.core import Activation, Flatten, Dense # Camada da função de ativação, flatten, entre outros
from keras.layers import Rescaling, Dropout # Camada de escalonamento
from keras.optimizers import Adam # optimizador Adam
from keras.callbacks import ModelCheckpoint # Classe utilizada para acompanhamento durante o treinamento onde definimos os atributos que serão considerados para avaliação
from keras.preprocessing.image import ImageDataGenerator # Gerador de imagens
from keras.utils import image_dataset_from_directory # Função que carrega o dataset de um diretório
from tensorflow.data import AUTOTUNE


# controla a versão do código
def get_latest_version(models_path):
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
    # Definimos que estamos criando um modelo sequencial
    model = Sequential()

    # Primeira camada do modelo:
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))

    # Segunda camada do modelo:
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))

    # Terceira camada do modelo:
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.20))

    # Primeira camada fully connected
    model.add(Flatten())
    model.add(Dense(510))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # Classificador softmax
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model


if __name__ == "__main__":
    train_path = "dataset/cats_and_dogs" # Adicione aqui o caminho para chegar no diretório que contém as imagens de treino na sua maquina
    models_path = "models" # Defina aqui onde serão salvos os modelos na sua maquina
    width = 64 # Tamanho da largura da janela que será utilizada pelo modelo
    height = 64 # Tamanho da altura da janela que será utilizada pelo modelo
    depth = 3 # Profundidade das janelas utilizadas pelo modelo, caso seja RGB use 3, caso escala de cinza 1
    classes = 2 # Quantidade de classes que o modelo utilizará
    epochs = 10 # Quantidade de épocas (a quantidade de iterações que o modelo realizará durante o treinamento)
    init_lr = 1e-3 # Taxa de aprendizado a ser utilizado pelo optimizador
    batch_size = 32 # Tamanho dos lotes utilizados por cada epoca
    input_shape = (height, width, depth) # entrada do modelo
    latest_version = get_latest_version(models_path) # adiciona um versionador de modelo antes de salvar
    save_model = os.path.join(models_path, f"lenet-{{epoch:02d}}-{{accuracy:.3f}}-{{val_accuracy:.3f}}-V{latest_version + 1}.model") # salva o modelo versionado
    color_mode = {1:"grayscale", 3: "rgb"} # Usado para selecionar o colormode em função da variável depth

    os.makedirs(models_path, exist_ok=True)

    # gera mais imagens para implementar o dataset
    datagen = ImageDataGenerator(
                            rotation_range=20,
                            zoom_range=0.15,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode="nearest",
                            brightness_range=[0.5, 1.5]
    )

    train_ds = image_dataset_from_directory(
                            train_path,
                            seed=123,
                            label_mode='categorical',
                            validation_split=0.2,
                            subset="training",
                            color_mode=color_mode[depth],
                            image_size=(height, width),
                            batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
                            train_path,
                            seed=123,
                            label_mode='categorical',
                            validation_split=0.2,
                            subset="validation",
                            color_mode=color_mode[depth],
                            image_size=(height, width),
                            batch_size=batch_size
    )


    rescaling_layer = Rescaling(1./255)
    # pré-busca em buffer para que você possa produzir dados do disco sem que a E/S se torne um bloqueio
    train_ds = train_ds.map(lambda x, y: (rescaling_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescaling_layer(x), y), num_parallel_calls=AUTOTUNE)

    model = create_lenet(input_shape)

    opt = Adam(lr=init_lr, decay=init_lr / epochs)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    print("\n training network")

    checkpoint1 = ModelCheckpoint(save_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(save_model, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1,checkpoint2]

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
    results_df['experiment_description'] = "V3: Inclusao de camadas de dropout para mitigar o overfitting e promover a generalizacao. Aumento de dados com tecnicas como zoom e rotacao para enriquecer o conjunto de treinamento e melhorar a robustez. Reducao da resolucao da imagem para 64x64 para acelerar o treinamento e focar em caracteristicas mais genericas. Ajuste na divisao de treino/teste para 80/20, permitindo mais dados para treinamento."

    # Concatena os antigos resultados com os novos
    final_results = pd.concat([old_results, results_df], ignore_index=True)

    # Salva o DataFrame final no arquivo CSV
    final_results.to_csv('results/resultados.csv', index=False)
