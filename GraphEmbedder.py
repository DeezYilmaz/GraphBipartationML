import os
import pandas as pd
import pickle
import karateclub.graph_embedding.feathergraph as Feather
from node2vec import Node2Vec



filename="Graphdata"
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "GraphDatas/"+filename+".pickle"
abs_file_path = os.path.join(script_dir, rel_path)

Graph_df=pickle.load(open(abs_file_path,'rb'))

# rel_path = "GraphDatas/GraphdataEmbed.pickle"
# abs_file_path = os.path.join(script_dir, rel_path)

# G2VEmbedding=pickle.load(open(abs_file_path,'rb'))


"""
GRAPH2VEC
"""
Gmodel = Feather.FeatherGraph()
Gmodel.fit(Graph_df['Graph'])
G2VEmbedding= Gmodel.get_embedding()


rel_path = "GraphDatas/"+filename+"Embed.pickle"
abs_file_path = os.path.join(script_dir, rel_path)

pickle.dump(G2VEmbedding,open(abs_file_path,'wb'))