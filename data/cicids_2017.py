import logging
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler



logger = logging.getLogger(__name__)

#*
def get_shape_input():
    """Get shape of the dataset for CICIDS_2017"""
    return (None, 78)
#*
def get_shape_label():
    """Get shape of the labels in CICIDS_2017"""
    return (None,)


