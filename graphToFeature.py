import networkx as nx
# from itertools import combinations, groupby
# import random
# from networkx.algorithms import community
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import karateclub.graph_embedding.graph2vec as Graph2Vec

from node2vec import Node2Vec

Graph_df=pickle.load(open('GraphData.csv','rb'))

"""
GRAPH2VEC
"""
gList=[]
colorList=[]
for i in range(40):
    gList.append(nx.connected_caveman_graph(5,5))
    colorList.append('red')
for i in range(40):
    gList.append(nx.erdos_renyi_graph(50,0.1))
    colorList.append('blue')


Gmodel = Graph2Vec.Graph2Vec(dimensions=2,wl_iterations=10,min_count=30)
Gmodel.fit(gList)
G2VEmbedding= Gmodel.get_embedding()


G2V_data = {"x":[], "y":[],  "label":[]}
for i in range(len(gList)):
    G2V_data["x"].append(G2VEmbedding[i][0])
    G2V_data["y"].append(G2VEmbedding[i][1])

fig=plt.figure(figsize=(10,8))
plt.title('Scatter Plot', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.scatter(G2V_data["x"], G2V_data["y"],marker='o',color=colorList)

plt.show()


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