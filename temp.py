import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from typing import List, Dict, Union, Optional
from pathlib import Path

from utils import df_to_sparse

'''
train-test split 
you should consider whether it is possible to simply divide.
'''

class Dataset(object):
    def __init__(self, data_path:str, separator:str='::', seed:int=42):

        self.data_path = Path(data_path)
        self.separator = separator
        self.seed = seed

        self._load_data()

    def _load_data(self) -> None:
        names=['user', 'item', 'rating', 'timestamp']
        dtype={'user': int, 'item': int, 'rating': float, 'timestamp': float}

        full_data = pd.read_csv(self.data_path, sep=self.separator, names=names, dtype=dtype)
        train_df, valid_df = train_test_split(full_data, test_size=0.3, shuffle=True, random_state=self.seed)
        valid_df, test_df = train_test_split(valid_df, test_size=0.4, shuffle=True, random_state=self.seed)
        
        self.train_data = df_to_sparse(train_df)
        self.valid_data = df_to_sparse(valid_df)
        self.test_data = df_to_sparse(test_df)