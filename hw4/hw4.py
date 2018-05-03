import sys
import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

x = np.load(sys.argv[1])

pca_x = PCA(n_components = 300, whiten = True).fit_transform(x)

kmeans = KMeans(n_clusters = 2).fit(pca_x)

label = kmeans.labels_

test = pd.read_csv(sys.argv[2])

ans = []
for i in range(test.shape[0]):
    if label[test.iloc[i][1]] == label[test.iloc[i][2]]:
        ans.append(1)
    else:
        ans.append(0)
        
outfile = open(sys.argv[3], 'w')

lines = ['ID,Ans\n']
for i in range(len(ans)):
    lines.append(str(i) + ',' + str(int(ans[i])) + '\n')
    
outfile.writelines(lines)
outfile.close()
