import numpy as np
import pandas as pd
pd.options.display.max_rows = 20


from classifier import Classifier


data = np.load("train_64x64.npz")
X, y = data['X'], data['y']

labels = pd.read_table("taxon_id_to_french_names.txt", sep="\s{2,}", names=["name", "taxon_id"], index_col="taxon_id")

c = Classifier()

c.fit(X,y)
