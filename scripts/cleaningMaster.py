import numpy as np

import pandas as pd
from scripts.importMaster import importData

def cleanData(importedData, type):
    return 0


if __name__ == "__main__":
    dataPath = "../data/features_labels/"
    data = importData(dataPath)

    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(data)
