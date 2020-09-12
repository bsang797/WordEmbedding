import umap
import pandas as pd
import numpy as np
import json
import os
import hdbscan
import time
import pickle

import matplotlib.pyplot as plt

#This is the raw text in which the sentences are embedded in - edit this
EMBEDDING_COLUMN = ''
EMBEDDING_DIMENSION = 5

#Get encodings here
encodings = os.listdir('./encodings')

dfs = []

for batch in encodings:

    df = pd.read_csv(f'./encodings/{batch}')
    dfs.append(df)

test_data = pd.concat(dfs)[[EMBEDDING_COLUMN, 'SerializedEmbedding']]

X = test_data.SerializedEmbedding
X = np.array([json.loads(embedding) for embedding in X])
y = np.array(test_data.Title)

time_start = time.time()

encoder = umap.UMAP(n_components=EMBEDDING_DIMENSION)

embeddings = encoder.fit_transform(X)

with open('./encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('./embeddings_test.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

time_end = time.time()

print(f'UMAP complete in {time_end - time_start} seconds')

clusterer = hdbscan.HDBSCAN()
cluster_labels = clusterer.fit_predict(embeddings)

print(cluster_labels.max())

classes = pd.DataFrame([y, cluster_labels]).T
classes.columns = ['Title', 'ClusterID']
classes.to_csv('./classes.csv', index=None)