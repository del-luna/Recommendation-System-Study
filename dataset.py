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

class Dataset(object):
    def __init__(self, data_path:str, valid_ratio:Union[int, float]=0.1, test_ratio:Union[int, float]=0.2, seed:int=42):

        self.data_path = Path(data_path)

        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

        self.seed= seed