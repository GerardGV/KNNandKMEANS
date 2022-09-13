__authors__ = ['1603392', '1605947', '1537514']
__group__ = 'DX.12/DJ.10'

import copy
import time
import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2
from collections import Counter

NUM_IMATGES = 150

#el numero de colores basicos es 11 por ello probamos k=11
KMAX = 11

#----------------------------------------------------
#           FUNCIONES DE ANALISIS CUALITATIVO
#----------------------------------------------------

def retrieval_by_color(imgs, labels, arr_str):
    labels_aux = []
    for x, i in enumerate(labels):
        for element in arr_str:
            if element in i:
                labels_aux.append(x)
                break

    return imgs[labels_aux]

def retrieval_by_shape(img, knn, str):
    labels_aux =[]
    for index, value in enumerate(knn):
        if value == str:
            labels_aux.append(index)

    return img[labels_aux]

#-----------------------------------------------------.
#           FUNCIONES DE ANALISIS CUANTITATIVO
#------------------------------------------------------
def get_shape_accurency (knn,gt):
    cont_prueba=0
    index_incorrecte =[]
    i = 0
    for prueba, correcte in zip(knn,gt):
        if correcte == prueba:
            cont_prueba += 1
        else:
            index_incorrecte.append(i)
        i+=1


    precission = (cont_prueba/gt.size) *100
    print("El nivel de precisio es:")
    print(precission)

def Kmean_statistic(kmeans, kmax):
    #Calcular tiempo de ejecución

    statistics = np.zeros(kmax-2)

    for img in kmeans:


        for k in range(2, kmax):
            time_init = time.time()
            it = img.fit()
            img.whitinClassDistance()
            time_final = time.time() - time_init

            statistics[k-2] += it/time_final


            # print("L'algoritme triga: " + str(time_final) + "s")
    statistics = statistics/len(kmeans)
    # print(statistics)
    return statistics

#------------------------------------------------------
# MAIN
#------------------------------------------------------

if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')
    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))


    #Declaració de variables pel funcionament de retrieval by color
    kmeansImatges=[]
    workImg = test_imgs[:NUM_IMATGES]
    k_colors = []


    #bucle para aplicar algoritmos y trabajar con el numero de imagenes que queramos en mejoras de Kmeans
    for it, imatge in enumerate(workImg):
        kmeansImatges.append(Kmeans.KMeans(imatge,KMAX))
        k_colors.append(kmeansImatges[it].colors)

#-----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
# TEST MILLORA DE OPTION INITS: RANDOM I COSTUME
    kmeansImatgesFirst = []
    kmeansImatgesRandom = []
    kmeansImatgesCustom = []
    diccFirst= {'km_init': 'first'}
    diccRandom = {'km_init': 'random'}
    diccCustom = {'km_init': 'custom'}

    for it, imatge in enumerate(workImg):
        #opcion first
        kmeansImatgesFirst.append(Kmeans.KMeans(imatge, KMAX,diccFirst))
        kmeansImatgesFirst[it].find_bestK(KMAX)
        # visualize_k_means(kmeansImatgesFirst[it], imatge.shape)

        #opcion random
        kmeansImatgesRandom.append(Kmeans.KMeans(imatge, KMAX, diccRandom))
        kmeansImatgesRandom[it].find_bestK(KMAX)
        # visualize_k_means(kmeansImatgesRandom[it], imatge.shape)

        #opcion custom
        kmeansImatgesCustom.append(Kmeans.KMeans(imatge, KMAX,diccCustom))
        kmeansImatgesCustom[it].find_bestK(KMAX)
        # visualize_k_means(kmeansImatgesCustom[it], imatge.shape)

    #Ejecutar statistics
    x_axis = range(2, KMAX)
    print("Statistics first:")
    stats_first = Kmean_statistic(kmeansImatgesFirst, KMAX)
    plt.plot(x_axis, stats_first, label="First")
    print("Statistics random:")
    stats_random = Kmean_statistic(kmeansImatgesRandom, KMAX)
    plt.plot(x_axis, stats_random, label="Random")
    print("Statistics custom:")
    stats_custom = Kmean_statistic(kmeansImatgesCustom, KMAX)
    plt.plot(x_axis, stats_custom, label="Custom")
    plt.title('Kmeans statistics')
    plt.xlabel('K value')
    plt.ylabel('iter/sec')
    plt.legend()
    plt.show()

    #Retrieval_by_color
    for kmeans in kmeansImatgesCustom:
        kmeans.get_labels()
        k_colors.append(Kmeans.get_colors(kmeans.centroids))
    colors_result = retrieval_by_color(workImg, k_colors, ['Red'])
    visualize_retrieval(colors_result, colors_result.shape[0], title = 'orange,black and white')


#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
# TEST MEJORAS DE LLINDARS
    kmeansImatges2 = copy.deepcopy(kmeansImatgesFirst)
    llindars=[10,30, 50, 70, 90]

    #seleccionamos un llindar
    for llindar in llindars:
        #aplicamos el llindar a todas las imagenes
        print("> LLindar "+str(llindar))
        k_colors = []
        for kmeans in kmeansImatges2:
            kmeans.find_bestK(KMAX,llindar)
            kmeans.get_labels()
            k_colors.append(Kmeans.get_colors(kmeans.centroids))

        # Retrieval_by_color
        colors_result = retrieval_by_color(workImg, k_colors, ['Red'])
        visualize_retrieval(colors_result, colors_result.shape[0], title = 'Red (llindar:'+str(llindar)+')')


#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
# TEST MEJORAS DE GREYS

    #ANALISIS QUALITATIU AMB retrieval_by_shape

    #Inicializar KNN_RGB
    knn_rgb = KNN.KNN(train_imgs, train_class_labels)
    knn_predict_rgb = knn_rgb.predict(test_imgs, 5)

    # Get_shape_accurency KNN_RGB
    get_shape_accurency(knn_predict_rgb, test_class_labels)

    #Retrieval_by_shape KNN_RGB
    shape_result_rgb = retrieval_by_shape(test_imgs, knn_predict_rgb, 'Shorts')
    iAnt=0
    for i in range(20, shape_result_rgb.shape[0], 20):
        visualize_retrieval(shape_result_rgb[iAnt:i], 20, title = 'Shorts')
        iAnt = i

    #Inicializar KNN_Grey
    knn_grey = KNN.KNN(train_imgs, train_class_labels, grey=True)
    knn_predict_grey = knn_grey.predict(test_imgs, 5)

    # Get_shape_accurency KNN_Grey
    get_shape_accurency(knn_predict_grey, test_class_labels)

    #Retrieval_by_shape KNN_Grey
    shape_result_grey = retrieval_by_shape(test_imgs, knn_predict_grey, 'Shorts')
    iAnt=0
    for i in range(20, shape_result_grey.shape[0], 20):
        visualize_retrieval(shape_result_grey[iAnt:i], 20, title = 'Shorts')
        iAnt = i

    #ANALISIS ESTADISTIC DE LA MILLORA GREY
    ks = [1, 2, 3, 4, 5, 10, 15]
    # KNN with RGB
    acc = []
    for k in ks:
        knn = KNN.KNN(train_imgs, train_class_labels)
        knn_predict = knn.predict(test_imgs, k)
        # Get_shape_accurency
        acc.append(get_shape_accurency(knn_predict, test_class_labels))

    plt.plot(ks, acc, label='RGB')

    # KNN with grey scale
    acc = []
    for k in ks:
        knn = KNN.KNN(train_imgs, train_class_labels, grey=True)
        knn_predict = knn.predict(test_imgs, k)
        # Get_shape_accurency
        acc.append(get_shape_accurency(knn_predict, test_class_labels))

    plt.plot(ks, acc, label='Grey Scale')

    plt.title('KNN accuracy')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.show()
#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
