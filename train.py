import os
import cv2 # OpenCV ou cv2 para tratamento de imagens;
import numpy as np # Numpy para trabalharmos com matrizes n-dimensionais
from keras.models import Sequential # Importando modelo sequencial
from keras.layers.convolutional import Conv2D, MaxPooling2D # Camada de convolução e max pooling
from keras.layers.core import Activation, Flatten, Dense # Camada da função de ativação, flatten, entre outros
from keras.layers import Rescaling # Camada de escalonamento
from keras.optimizers import Adam # optimizador Adam
from keras.callbacks import ModelCheckpoint # Classe utilizada para acompanhamento durante o treinamento onde definimos os atributos que serão considerados para avaliação
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import image_dataset_from_directory # Função que carrega o dataset de um diretório




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
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Segunda camada do modelo:
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Primeira camada fully connected
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # Classificador softmax
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model


if __name__ == "__main__":
    train_path = "./cats_and_dogs" # Adicione aqui o caminho para chegar no diretório que contém as imagens de treino na sua maquina
    models_path = "models" # Defina aqui onde serão salvos os modelos na sua maquina
    width = 100 # Tamanho da largura da janela que será utilizada pelo modelo
    height = 100 # Tamanho da altura da janela que será utilizada pelo modelo
    depth = 1 # Profundidade das janelas utilizadas pelo modelo, caso seja RGB use 3, caso escala de cinza 1
    classes = 2 # Quantidade de classes que o modelo utilizará
    epochs = 10 # Quantidade de épocas (a quantidade de iterações que o modelo realizará durante o treinamento)
    init_lr = 1e-3 # Taxa de aprendizado a ser utilizado pelo optimizador
    batch_size = 32 # Tamanho dos lotes utilizados por cada epoca
    input_shape = (height, width, depth) # entrada do modelo
    save_model = os.path.join(models_path, "lenet-{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}.model")
    color_mode = {1:"grayscale", 3: "rgb"} # Usado para selecionar o colormode em função da variável depth

    os.makedirs(models_path, exist_ok=True)

    train_ds = image_dataset_from_directory(
                            train_path,
                            seed=123,
                            label_mode='categorical',
                            validation_split=0.3,
                            subset="training",
                            color_mode=color_mode[depth],
                            image_size=(height, width),
                            batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
                            train_path,
                            seed=123,
                            label_mode='categorical',
                            validation_split=0.3,
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
