'''
original data shape

userid/movieid/ratings
...  / ... / ...  
... / ... / ...

-> pivot table shape

'''


import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from typing import List, Dict, Union, Optional
from pathlib import Path

from utils import df_to_sparse

class Dataset(object):
    def __init__(self, data_path:str, generalization:str='weak', valid_ratio:Union[int, float]=0.1, test_ratio:Union[int, float]=0.2, seed:int=42):

        self.data_path = Path(data_path)

        self.generalization = generalization

        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

        self.seed= seed

    def _load_preprocessed_data(self) -> None:
        print('Load data...')

        self.user2id = self._load_id_map(self._user2id_file)
        self.item2id = self._load_id_map(self._item2id_file)
        self.num_users, self.num_items = len(self.user2id), len(self.item2id)

        names=['user', 'item', 'rating', 'timestamp']
        dtype={'user': int, 'item': int, 'rating': float, 'timestamp': float}

        if self.generalization == 'weak':
            train_df = pd.read_csv(self._prepro_file_dict['train'], sep=',', names=names, dtype=dtype)
            valid_df = pd.read_csv(self._prepro_file_dict['valid'], sep=',', names=names, dtype=dtype)
            test_df = pd.read_csv(self._prepro_file_dict['test'], sep=',', names=names, dtype=dtype)

            self.train_users = self.valid_users = self.test_users = list(pd.unique(train_df.user))

            self.train_data = df_to_sparse(train_df, shape=(self.num_users, self.num_items))
            self.valid_target = df_to_sparse(valid_df, shape=(self.num_users, self.num_items))
            self.test_target = df_to_sparse(test_df, shape=(self.num_users, self.num_items))





    def _load_id_map(self, id_map_file: Path) -> Dict:
        old2new = {}
        with open(id_map_file, 'rt') as f:
            for line in f.readlines():
                u, uid = line.strip().split(', ')
                old2new[int(u)] = int(uid)
        return old2new

    #Write user/item id map into files
    def _save_id_map(self, id_map: Dict, id_map_file: Path) -> None:
        with open(id_map_file, 'wt') as f:
            for u, uid in id_map.items():
                f.write('%d, %d\n' % (u, uid))

    @property
    def valid_input(self) -> sp.csr_matrix:
        if self.generalization == 'weak':
            return self.train_data
        else:
            return self.valid_input

    @property
    def test_input(self) -> sp.csr_matrix:
        if self.generalization == 'weak':
            return self.train_data + self.valid_target
        else:
            return self.test_input

    @property
    def num_train_users(self) -> int:
        return len(self.train_users)
    
    @property
    def num_valid_users(self) -> int:
        return len(self.valid_users)

    @property
    def num_test_users(self) -> int:
        return len(self.test_users)