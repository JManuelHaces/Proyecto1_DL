# Archivo con funciones que serán útiles para el proyecto
# -------------------------------------------------------------------------------------------------
# Librerías
import os
import shutil
import random
import pathlib
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# -------------------------------------------------------------------------------------------------
# Funciones

# Función para ver los contenidos de las carpetas
def folders_content(path:str) -> None:
    # Listas con los nombres de las carpetas
    carpetas = os.listdir(path)
    # Lista con los paths a las carpetas
    list_paths = [carpeta for carpeta in carpetas if os.path.isdir(os.path.join(path, carpeta))]
    # ----------------------------------------------------------------------------------------------
    # Ciclo para movernos por folder
    for folder in list_paths:
        # Obteniendo el contenido del folder
        content = os.listdir(path + folder)
        # Imprimiendo el contenido del folder
        print(f'- Contenido de la carpeta {folder}:')
        # Si el contenido del folder es mayor a 5, mostramos los primeros 5 y cuántos faltan (solo carpetas) 
        if len(content) > 5:
                for arch in content[:5]:
                    if not '.' in arch:
                        print(f'\t- {arch}/')
                    else:
                        pass
                # Imprimimos cuántas carpetas más hay
                print(f'\t- Más {len(content) - 5} carpetas.')
                # En caso de tener archivos los mostramos
                if '.' in arch:
                    print(f'\t- {arch}')
                print()
        # En caso de tener menos de 5 carpetas mostramos como viene
        else:
            for arch in content:
                if not '.' in arch:
                        print(f'\t- {arch}/')
                else:
                    print(f'\t- {arch}')
            print()


# Función para realizar los cambios necesarios en la carpeta 'Val'
def directory_name_change(df_nombres: pd.DataFrame, df_data: pd.DataFrame, original_path:str, new_path:str):
    # Ciclo para movernos por las clases que hay
    for i in list(df_nombres['Clase']):
        # Tomando la fila con la clase que se está usando
        clase_actual = df_nombres[df_nombres['Clase'] == i].reset_index(drop=True)
        # Datos de las imágenes seleccionadas
        data_actual = df_data[df_data['Clase'] == i].reset_index(drop=True)
        # Tomando la lista con las imágenes
        list_imgs = clase_actual['Img_Name'][0]
        # Creando una nueva carpeta
        path_carpeta_nueva = f'{new_path}{i}/'
        os.makedirs(f'{path_carpeta_nueva}/images/', exist_ok=True)
        # Cambiando de lugar las carpetas
        for img in list_imgs:
            # Cambiando de lugar la carpeta
            shutil.copyfile(fr'{original_path}/{img}', f'{path_carpeta_nueva}/images/{img}')
        # Exportsando un CSV por carpeta
        data_actual.to_csv(f'{path_carpeta_nueva}/{i}_boxes.txt', sep='\t', index=False)


def data_generator(path:str, img_size: tuple = (32, 32)):
    # Parámetros de la generación de imágenes
    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    # Carga las imágenes de train utilizando el generador
    data_generator_dir = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical'
    )



    # Función para cargar una imagen y su anotación
    def load_img_and_annotation(img_path):
        img = Image.open(img_path)
        class_name = os.path.basename(os.path.dirname(img_path))
        annotation_path = os.path.join(os.path.dirname(img_path), class_name + '_boxex.txt')
        # Cargando anotaciones (.txt)
        with open(annotation_path, 'r') as f:
            annotation = f.read().strip()
        # Crear un diccionario con las anotaciones
        # Forma {NombreImagen: [bounds (len_4)]}
        anotaciones_dic = {}
        for linea in annotation.split('\n'):
            temp = linea.split('\t')
            anotaciones_dic[temp[0]] = [class_name] + temp[1:]

        return np.array(img), annotation

    # Función de generación de datos personalizada
    def generator(generator):
        while True:
            batch_x, batch_y = generator.next()
            batch_x_processed = []
            batch_y_processed = []
            for i in range(len(batch_x)):
                img, annotation = load_img_and_annotation(batch_x[i])
                batch_x_processed.append(img)
                batch_y_processed.append(annotation)
            yield np.array(batch_x_processed), np.array(batch_y_processed)

    # Retorna el generador personalizado de datos
    return generator(data_generator_dir)
