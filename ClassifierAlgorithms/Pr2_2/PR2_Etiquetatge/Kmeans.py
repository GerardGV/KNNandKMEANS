__authors__ = ['1603392', '1605947', '1537514']
__group__ = 'DX.12/DJ.10'

import array
import copy
import math
import random
import numpy as np
import utils

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)# DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################
        self.Original_Matrix = X #guardamos el hipercubo para luego tratarlo en el init_centroid
        self._init_centroids()
        self.colors=get_colors(self.centroids)

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
        ##si es llista hay que comprobar que todos los elementos son float

        if X.dtype != 'float':
            X = np.array(X, dtype='float')

            # R2
        if X.ndim <= 3:
            N = X.shape[0] * X.shape[1]
            aux = X.reshape([N, 3])
            self.X = copy.deepcopy(aux)
        else:
            shape = X.shape
            N=shape[len(shape)-2]
            aux = X.reshape([N, 3])
            self.X = copy.deepcopy(aux)

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  #within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options


        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        #K es el nº de centroides que nos pasan por parámetro
        #D es el nº de canales, siempre es 3

        if self.options['km_init'].lower() == 'first':
            #la D es esto shape[1]
            self.centroids = np.empty([self.K, self.X.shape[1]])
            self.old_centroids = np.empty([self.K, self.X.shape[1]])

            #Bucle para revisar todos los valores de la primer fila hasta que sean diferentes
            i= 1
            kDiferentes = [self.X[0]]
            self.centroids[0]=self.X[0]

            for elemento in self.X:
                if i == self.K:
                    break
                exits = False

                for kDiferent in kDiferentes:
                    if np.array_equal(elemento,kDiferent):
                        exits = True
                if not exits:
                    kDiferentes.append(elemento)
                    i += 1

            self.centroids = np.array(copy.deepcopy(kDiferentes))

        elif self.options['km_init'].lower() == 'random':

            # Bucle para obtener diferentes pixeles aleatorios de la matriz X los cuales usar como centroides

            #Cogemos el indice de un pixel cualquiera de X , como cada row de X es un pixel, le decimos que entre 0 y el num de filas
            indices = random.sample(range(0,self.X.shape[0]), 1)

            #obtendremos tantos pixeles o centroids como K
            k=1
            while k < self.K:

                #generamos un nuevo indice aleatorio de la matriz X, osea que escogemos un nuevo pixel aleatorio
                new=random.sample(range(0, self.X.shape[0]), 1)
                i=0
                repetit=False

                #bucle para comparar el nuevo pixel/centroid con los ya obtenidos
                while i<len(indices) and repetit != True:

                    #nuevo centroide aleatorio
                    nuevo=self.X[new[0]]

                    #centroide aleatorio ya guardado
                    old=self.X[indices[i]]


                    if np.array_equal(nuevo, old):
                        repetit=True
                    else:
                        i+=1

                #si el nuevo centroide no lo teniamos, lo añadimos y buscamos el siguiente
                if repetit != True:
                    indices.append(new[0])
                    k+=1

            self.centroids = self.X[indices]
            self.old_centroids = np.zeros([self.K, self.X.shape[1]], dtype=float)
            """
            while  len(self.centroids) != len(np.unique(self.centroids, axis=0)):
                self.centroids = np.matrix(np.unique(self.centroids, axis=0))
                b = self.K-len(self.centroids)
                a = np.random.rand(b, self.X.shape[1])
                self.centroids = np.append(self.centroids, a, axis=0)
            """

        elif self.options['km_init'].lower() == 'custom':
            # Inicializar las dimensiones
            self.centroids = np.empty([self.K, self.X.shape[1]])
            #self.old_centroids = np.empty([self.K, self.X.shape[1]])

            # Coger diagonal del hipercubo de dimensiones [m,n,z]
            diag_1 = self.Original_Matrix.diagonal(0,0,1)

            # Proceso para escoger únicamente valores no repetidos
            unique = []
            for filas in diag_1:
                for columnas in filas:
                    if columnas not in unique:
                        unique.append(columnas)

            # Convertir en array y posteriormente redimensionarla
            not_repeated = np.array(unique)
            res_final = np.resize(not_repeated, (self.K, self.X.shape[1]) )

            #setar los resultados con los atributos de la clase
            self.centroids = res_final
            self.old_centroids = np.zeros([self.K, self.X.shape[1]], dtype=float)

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        MT_cercanos=distance(self.X, self.centroids)
        self.labels = np.argmin(MT_cercanos, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #guardem els valors dels centroids en old centroids
        self.old_centroids = copy.deepcopy(self.centroids)

        # recorremos el vector X en función de los centroides disponibles en labels
        for nCentroid in range(self.labels.max()+1):
            #creamos variables en las que guardamos el sumatorio

            index = np.where(self.labels == nCentroid)
            result = np.mean(self.X[index], axis=0)
            self.centroids[nCentroid] = result

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return(np.array_equal(self.centroids, self.old_centroids))

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()
        it = 0
        while not self.converges():
            self.get_labels()
            self.get_centroids()
            it += 1

        return it

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
        WCD = np.array([], dtype=float).reshape([0, 1])
        centroid=0
        distancia=[]

        dist = distance(self.X, self.centroids)

        while centroid < self.centroids.shape[0]:
            for indice, x in np.ndenumerate(self.X):
                if centroid == self.labels[indice[0]]:
                    distancia.append((dist[indice[0]][centroid])**2)
            centroid += 1

        return np.mean(distancia)

    def inter_class(self):



        pass

    def find_bestK(self, max_K, llindar=20):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #Declaración de valores
        k = 2
        WCD_A = 0
        WCD_S = 0

        while k != max_K:
            WCD_A = WCD_S
            self.K = k
            self.fit()
            WCD_S = self.whitinClassDistance()

            if k != 2:
                if (100 - (WCD_S/WCD_A) * 100) < llindar:
                    self.K= k-1

                    break
            k+=1
            self.num_iter += 1

def distance(X, C):

    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    dist = np.zeros((C.shape[0], X.shape[0]), dtype=float)

    for i, centroide in enumerate(C):
        func_lambda = lambda x: x-centroide
        dist[i] = np.linalg.norm(func_lambda(X), axis=1)

    return dist.T


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    p_colorines = utils.get_color_prob(centroids)


    lista_colores = []

    for centroid in p_colorines:
        max_color = 0.0
        res_indice = 0
        for indice, valor in enumerate(centroid):
            if valor > max_color:
                max_color = valor
                res_indice = indice
        #if utils.colors[res_indice] not in lista_colores:
        lista_colores.append(utils.colors[res_indice])


    return lista_colores
