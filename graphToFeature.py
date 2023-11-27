import math
import os
import networkx as nx
# from itertools import combinations, groupby
import random
from networkx.algorithms import community
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import karateclub.graph_embedding.feathergraph as Feather
from node2vec import Node2Vec


from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier




# rel_path = "GraphDatas/GraphdataEmbed.pickle"
# abs_file_path = os.path.join(script_dir, rel_path)

# G2VEmbedding=pickle.load(open(abs_file_path,'rb'))


"""
GRAPH2VEC
"""

filename="Graphdata"
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "GraphDatas/"+filename+"Embed.pickle"
abs_file_path = os.path.join(script_dir, rel_path)

G2VEmbedding=pickle.load(open(abs_file_path,'rb'))


rel_path = "GraphDatas/"+filename+".pickle"
abs_file_path = os.path.join(script_dir, rel_path)

Graph_df=pickle.load(open(abs_file_path,'rb'))

print("Completed Graph2Vec")
# G2V_data = {"x":[], "y":[], "z":[], "label":[]}
# for i in range(len(Graph_df['Graph'])):
#     G2V_data["x"].append(G2VEmbedding[i][0])
#     G2V_data["y"].append(G2VEmbedding[i][1])
    # G2V_data["z"].append(G2VEmbedding[i][2])

# plt.style.use("seaborn-v0_8")
# fig=plt.figure(figsize=(10,8))
# ax=plt.axes(projection ='3d')

# plt.title('Scatter Plot', fontsize=20)
# plt.xlabel('x', fontsize=15)
# plt.ylabel('y', fontsize=15)
# plt.scatter(G2V_data["x"], G2V_data["y"],marker='o',color=colorList)

# plt.show()

knn = KNeighborsClassifier(n_neighbors=50)


X_train, X_test, y_train, y_test =train_test_split(G2VEmbedding,Graph_df['Labels'],test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(f1_score(y_test, y_pred, average=None))
ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.show()



label_list=[]
Graph_list=[]
for k in range(1,8):
    for i in range(50):
        nodeCount=random.randint(150*k,150*(k+1))*2
        G=nx.barabasi_albert_graph(random.randint(200,300),random.randint(1,10))
        Partition=community.kernighan_lin.kernighan_lin_bisection(G,max_iter=int( (0.4*len(G.nodes())*math.log10(len(G.nodes())))) )
        Gs1=nx.subgraph(G,Partition[0])
        Gs2=nx.subgraph(G,Partition[1])
        if(nx.is_connected(Gs1) and nx.is_connected(Gs2) and len(Gs1.nodes()) <= (len(G.nodes())/2+len(G.nodes())*0.01) 
                and len(Gs1.nodes()) >= (len(G.nodes())/2-len(G.nodes())*0.01)   ):
                label_list=[True]+label_list
                Graph_list=[G]+Graph_list
        else:
                label_list=label_list+[False]
                Graph_list=Graph_list+[G]


y_pred=knn.predict(Graph_list)
accuracy = accuracy_score(label_list, y_pred)
print("Accuracy:", accuracy)

print(f1_score(label_list, y_pred, average=None))
ConfusionMatrixDisplay.from_estimator(knn, Graph_list, label_list)

exit()





###################

rel_path = "GraphDatas/GraphdataSmall.pickle"
abs_file_path = os.path.join(script_dir, rel_path)

Graph_df=pickle.load(open(abs_file_path,'rb'))

Gmodel = Graph2Vec.FeatherGraph()
Gmodel.fit(Graph_df['Graph'])
G2VEmbedding= Gmodel.get_embedding()



X_test = scaler.transform(G2VEmbedding)


y_pred = knn.predict(X_test)
accuracy = accuracy_score(Graph_df['Labels'], y_pred)
print("Accuracy:", accuracy)

print(f1_score(Graph_df['Labels'], y_pred, average=None))
ConfusionMatrixDisplay.from_estimator(knn, X_test, Graph_df['Labels'])
plt.show()

exit()
"""
NODE2VEC
"""

Graph=nx.erdos_renyi_graph(100,0.01)

node2vec = Node2Vec(Graph, dimensions=2, walk_length=100, num_walks=20, workers=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
node_embeddings = {str(node): model.wv[str(node)] for node in Graph.nodes()}

node2Vec_data = {"x":[], "y":[], "z":[], "label":[]}

for i in node_embeddings:
    node2Vec_data["x"].append(node_embeddings[i][0])
    node2Vec_data["y"].append(node_embeddings[i][1])




fig=plt.figure(figsize=(10,8))
plt.title('Scatter Plot', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.scatter(node2Vec_data["x"], node2Vec_data["y"],marker='o', color='red')

plt.show()
# Partition=community.kernighan_lin_bisection(Gs,max_iter=50)

# Gs1=nx.subgraph(Gs,Partition[0])
# Gs2=nx.subgraph(Gs,Partition[1])
# colors = ['#d7191c', '#ffffbf', '#2b83ba']
# node_colors=[]
# for node in Gs.nodes():
#     if node in list(Gs1.nodes()):
#         node_colors.append(colors[0])
#     else:
#         node_colors.append(colors[1])
        
# plt.subplot(231)
# nx.draw(Gs, node_color=node_colors, with_labels='True')
# plt.title("main")

# plt.subplot(232)
# nx.draw(Gs1,with_labels=True)
# plt.title("part1")

# plt.subplot(233)
# nx.draw(Gs2,with_labels=True)
# plt.title("part2")
# plt.show()