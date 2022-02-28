import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import datetime
from datetime import date
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
 
class TopicClustering():
  freq = []
  def __init__(self):
    self.model = SentenceTransformer('sentence-transformers/paraphrase-albert-base-v2')
 

  def transformation(self, data):
    vectors = []
    for index, row in data.iterrows():
      embeddings = self.model.encode(row.articleHead)
      vectors.append(np.array(embeddings))
    sentences = data.articleHead.to_list()
    indexes = data._id.to_list()
    transformed = {'vectors' : vectors, 'sentences' : sentences, 'indexes' : indexes}
    return transformed
 
 
  def dbscan(self, transformed, eps = 0.42, min_samples = 5, pcaDims=16):
    vectors = transformed['vectors']
    sentences = transformed['sentences']
    indexes = transformed['indexes']
    pca = PCA(pcaDims)
    vectors_128 = pca.fit_transform(vectors)
    x=np.array(vectors_128)
    n_classes = {}
    dbscan = DBSCAN(eps = eps, min_samples = min_samples, metric = 'cosine').fit(x)
 
    results = pd.DataFrame({'_id': indexes, 'articleHead': sentences, 'label':dbscan.labels_})
 
    for i in results.label.unique():
      self.freq.append((i, len(results[results.label == i])))
 
    self.freq.sort(key=lambda x:x[1], reverse = True)
    return results
 
  def getResults(self, data):
    transformed = self.transformation(data)
    results = self.dbscan(transformed)
    return results
 
  def getFreq(self):
    return self.freq
 
  def getClusterList(self, results, proper_sets):
    topClusterList = []
    for i in proper_sets:
      cluster = pd.DataFrame(columns = ['Label','_id', 'Article_Head'])
      for index, row in results[results.label== i].iterrows():
        cluster = cluster.append({'Label':i,'_id':row[2], 'Article_Head':row[1]}, ignore_index=True)
      
      topClusterList.append(cluster)
    return topClusterList