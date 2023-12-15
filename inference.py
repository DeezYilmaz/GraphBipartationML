import math
import networkx as nx
import pandas as pd
import pickle
import numpy as np

import karateclub.graph_embedding.graph2vec as Graph2Vec
import karateclub.graph_embedding.feathergraph as Feather


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class bisectionModel:
    def __init__(self,EmbedModel,Model,scaler):
        self.EmbedModel=EmbedModel
        self.Model=Model
        self.scaler=scaler
    def inference(self,Graphs):
        GraphEmbedding= self.EmbedModel.infer(Graphs)
        X = self.scaler.transform(GraphEmbedding)
        y_preds=self.Model.predict(X)

        return y_preds
    def getEmbedding(self,Graphs):
        GraphEmbedding= self.EmbedModel.infer(Graphs)
        X = self.scaler.transform(GraphEmbedding)
        return X


model=pickle.load(open("model.pkl",'rb'))
graphs=pickle.load(open("Graphs.pickle",'rb'))
y_pred=model.inference(graphs)
print(y_pred)
