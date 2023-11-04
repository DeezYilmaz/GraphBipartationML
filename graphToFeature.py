import networkx as nx
# from itertools import combinations, groupby
# import random
# from networkx.algorithms import community
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from node2vec import Node2Vec

Graph_df=pickle.load(open('GraphData.csv','rb'))
Gs= Graph_df.iloc[7]['Graph']

Graph=nx.erdos_renyi_graph(1000,0.04)
# nx.draw(Graph)
# plt.show()
node2vec = Node2Vec(Graph, dimensions=2, walk_length=4, num_walks=10, workers=1)

model = node2vec.fit(window=10, min_count=1, batch_words=4)
node_embeddings = {str(node): model.wv[str(node)] for node in Graph.nodes()}




data = {"x":[], "y":[], "z":[], "label":[]}

for i in node_embeddings:
    data["x"].append(node_embeddings[i][0])
    data["y"].append(node_embeddings[i][1])

fig=plt.figure(figsize=(10,8))
plt.title('Scatter Plot', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.scatter(data["x"], data["y"],marker='o')

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