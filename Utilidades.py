# Archivo con funciones que serán útiles para el proyecto
# -------------------------------------------------------------------------------------------------
# Librerías
import os
import pickle
import shutil
import random
import pathlib
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import tensorflow.keras.backend as K
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


def data_generator(path:str, img_size: tuple = (64, 64, 3)):
    # Parámetros de la generación de imágenes
    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    # Carga las imágenes de train utilizando el generador
    data_generator_dir = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=128,
        class_mode=None
    )

    # Función para cargar una imagen y su anotación
    def load_img_and_annotation(img_path):
        with open(r"./Varios/Dict_Clases.pkl", "rb") as input_file:
             dict_clases = pickle.load(input_file)
        img = Image.open(img_path)
        img_path = img_path.replace('\\', '/')
        split_path = img_path.split('/')
        new_path = ('/').join(split_path[:-2])
        class_name = split_path[-3]
        annotation_path = os.path.join(new_path, class_name + '_boxes.txt')
        # Cargando anotaciones (.txt)
        with open(annotation_path, 'r') as f:
            annotations = f.read().strip()
        # Crear un diccionario con las anotaciones
        # Forma {NombreImagen: [class, bounds(len=4)]}
        anotaciones_dic = {}
        for linea in annotations.split('\n'):
            temp = linea.split('\t')
            clase = dict_clases.get(class_name)
            if clase == None:
                raise ValueError(f'La clase {class_name} no está en el diccionario de clases.')
            anotaciones_dic[temp[0]] = [clase] + temp[1:]
        annotation = anotaciones_dic[img_path.split('/')[-1]]

        return np.array(img), annotation

    # Función de generación de datos personalizada
    def generator(generator):
        while True:
            batch_x_processed = []
            batch_y_processed = []
            for i, filename in enumerate(generator.filenames):
                path_img = os.path.join(generator.directory, filename)
                img, annotation = load_img_and_annotation(path_img)
                batch_x_processed.append(img)
                batch_y_processed.append(annotation)
            yield np.array(batch_x_processed), np.array(batch_y_processed)

    # Retorna el generador personalizado de datos
    return generator(data_generator_dir)



# IoU Metric
def iou_metric(y_true, y_pred):
    """
    Función de métrica personalizada para la métrica de IoU (Intersection over Union).
    y_true: tensor de tamaño (batch_size, 4) que contiene las coordenadas de los bboxes verdaderos
    y_pred: tensor de tamaño (batch_size, 4) que contiene las coordenadas de los bboxes predichos
    """
    # Definir las coordenadas de los bboxes verdaderos y predichos
    xmin_true, ymin_true, xmax_true, ymax_true = K.split(y_true, 4)
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = K.split(y_pred, 4)
    # Calcular las coordenadas de la intersección
    xmin_inter = K.maximum(xmin_true, xmin_pred)
    ymin_inter = K.maximum(ymin_true, ymin_pred)
    xmax_inter = K.minimum(xmax_true, xmax_pred)
    ymax_inter = K.minimum(ymax_true, ymax_pred)
    # Calcular el área de la intersección
    w_inter = K.maximum(0.0, xmax_inter - xmin_inter + 1.0)
    h_inter = K.maximum(0.0, ymax_inter - ymin_inter + 1.0)
    area_inter = w_inter * h_inter
    # Calcular el área de las cajas verdaderas y predichas
    w_true = xmax_true - xmin_true + 1.0
    h_true = ymax_true - ymin_true + 1.0
    area_true = w_true * h_true
    w_pred = xmax_pred - xmin_pred + 1.0
    h_pred = ymax_pred - ymin_pred + 1.0
    area_pred = w_pred * h_pred
    # Calcular el área de la unión
    area_union = area_true + area_pred - area_inter
    # Calcular la métrica IoU
    iou = area_inter / K.maximum(area_union, 1e-7)
    # Retornar el promedio de la métrica IoU sobre todo el batch
    return K.mean(iou)

