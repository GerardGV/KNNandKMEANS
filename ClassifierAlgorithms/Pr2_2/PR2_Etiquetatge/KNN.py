__authors__ = ['1605947', '1537514','1603392']
__group__ = 'DX.12/DJ.10'

import numpy as np
import math
import operator

import scipy.spatial.distance
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels, grey=False):
        self.grey = grey
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################



    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        ##self.train_data = np.random.randint(8,size=[10,14400])

        train_data=train_data.astype('float64')
        #P=train_data.shape[0]
        #D=train_data.shape[1]*train_data.shape[2]*train_data.shape[3]
        #self.train_data=np.reshape(train_data, (P, D))
        #self.train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]*train_data.shape[3]))

        if self.grey:
            train_data[:, :, :, 0] = 0.2989 * train_data[:, :, :, 0] + 0.5870 * train_data[:, :, :,1] + 0.1140 * train_data[:, :, :, 2]
            train_data = np.delete(train_data, [1, 2], 1)

        self.train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2] * train_data.shape[3]))

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        # reshape igual que con train_data
        test_data = test_data.astype('float64')
        #D = test_data.shape[1] * test_data.shape[2] * test_data.shape[3]
        if self.grey:
            test_data[:, :, :, 0] = 0.2989 * test_data[:, :, :, 0] + 0.5870 * test_data[:, :, :,1] + 0.1140 * test_data[:, :, :, 2]
            test_data = np.delete(test_data, [1, 2], 1)

        test_data=np.reshape(test_data,(test_data.shape[0], test_data.shape[1] * test_data.shape[2] * test_data.shape[3]))

        # cambiamos shape de neigbhours N*k
        self.neighbors=np.empty([test_data.shape[0], k], dtype=float)

        # calcular distancias entre
        distancias = cdist(test_data, self.train_data)

        #guardamos las k etiquetas mas cercanas a cada punto
        #obtenemos las columnas k con distancias mas pequeñas de cada fila y lo guardamos ordenado de menor a mayor(distancias es segun lo que ordenamos y indicies lo que guardamos)
        indices=np.argsort(distancias, 1,)[:,:k]
        self.neighbors=self.labels[indices]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #elemento mas repetido en cada fila
        #tmp=np.bincount(self.neighbors)
        #tmp2=np.argmax(tmp)

        ##func_lambda1 = lambda x: np.bincount(x).argmax()

        maxRep=[]
        porcentajes=[]
        for fila in self.neighbors:
            uniques, counts = np.unique(fila, return_counts=True)
            maxRep.append(uniques[np.argmax(counts)])
            porcentajes.append(float(np.amax(counts))/float(fila.shape[0]))

        #uniques, counts=np.unique(self.neighbors, return_counts=True, axis=1)
        #array1=np.array(maxRep)
        self.percent=np.array(porcentajes)
        #return array1, array2
        return np.array(maxRep)
        #return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """


        #return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)
        self.get_k_neighbours(test_data,k)
        return self.get_class()
