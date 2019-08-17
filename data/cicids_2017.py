import logging
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
RANDOM_SEED = 13
def get_train(*args):
    rng = np.random.RandomState(RANDOM_SEED)
    """Get training dataset for KDD 10 percent"""
    train_list=[x for x in range(0,8)]
    train_path='MachineLearningCVE/'
    train_list_path=[train_path + str(x) + '.csv' for x in train_list]
    InD = np.zeros((0,79),dtype=object)
    for x in train_list_path:
        InD=np.vstack((InD,pd.read_csv(x)))
    Dt=InD[:,:-1].astype(float)
    L_N=InD[~np.isnan(Dt).any(axis=1),-1]
    #DtNMV -- data without nan values
    D_N=Dt[~np.isnan(Dt).any(axis=1)]


    #Remove Inf values
    #labels without nan and inf values
    L_NI=L_N[~np.isinf(D_N).any(axis=1)]
    #data without nan and inf values
    D_NI=D_N[~np.isinf(D_N).any(axis=1)]
    del(D_N)

    D_NI=MinMaxScaler().fit_transform(D_NI)

    x_train_net=D_NI[L_NI=='BENIGN',:]
    x_train_net=x_train_net[rng.permutation(x_train_net.shape[0])]
    trainx=x_train_net[:int(x_train_net.shape[0]*0.6),:]

    y_train=L_NI[L_NI=='BENIGN']


    x_test_anomaly=D_NI[L_NI!='BENIGN',:]
    x_test_benign=x_train_net[int(x_train_net.shape[0]*0.6):,:]
    del(D_NI)
    del(L_NI)

    rho=0.2


    # normal data - x_test_benign

    # anomalous data - x_test_anomaly

    inds = rng.permutation(x_test_anomaly.shape[0])
    x_test_anomaly=x_test_anomaly[inds]

    inds = rng.permutation(x_test_benign.shape[0])
    x_test_benign = x_test_benign[inds] 

    size_test = x_test_benign.shape[0]
    out_size_test = int(size_test*rho/(1-rho))

    x_test_anomaly = x_test_anomaly[:out_size_test]

    y_test_benign=np.ones(x_test_benign.shape[0])
    y_test_anomaly=np.zeros(x_test_anomaly.shape[0])
    print("benign_shape",y_test_benign.shape)
    print("anomaly_shape",y_test_anomaly.shape)

    testx = np.concatenate((x_test_benign,x_test_anomaly), axis=0)
    testy = np.concatenate((y_test_benign,y_test_anomaly), axis=0)

    return trainx, y_train, testx,testy

def get_shape_input():
    """Get shape of the dataset for cicids_2017"""
    return (None, 78)

def get_shape_label():
    """Get shape of the labels in cicids_2017"""
    return (None,)


