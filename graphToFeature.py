import networkx as nx
from itertools import combinations, groupby
import random
from networkx.algorithms import community
import matplotlib.pyplot as plt
import pandas as pd
import pickle

Graph_df=pickle.load(open('GraphData.csv','rb'))
Gs= Graph_df.iloc[3]['Graph']

Partition=community.kernighan_lin_bisection(Gs,max_iter=10)
Gs1=nx.subgraph(Gs,Partition[0])
Gs2=nx.subgraph(Gs,Partition[1])
print(nx.is_connected(Gs1) and nx.is_connected(Gs2) and len(Gs1.nodes())==len(Gs2.nodes()))


colors = ['#d7191c', '#ffffbf', '#2b83ba']
node_colors=[]
for node in Gs.nodes():
    if node in list(Gs1.nodes()):
        node_colors.append(colors[0])
    else:
        node_colors.append(colors[1])
        
plt.subplot(231)
nx.draw(Gs, node_color=node_colors, with_labels='True')
plt.title("main")

plt.subplot(232)
nx.draw(Gs1,with_labels=True)
plt.title("part1")

plt.subplot(233)
nx.draw(Gs2,with_labels=True)
plt.title("part2")
plt.show()
