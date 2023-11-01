import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt


fig = plt.subplots(1, figsize=(12,10))


G=nx.erdos_renyi_graph(40,0.05,seed=None,directed=False)


iterable= max(nx.connected_components(G), key=len)
Gs=nx.subgraph(G,iterable)


Partition=community.kernighan_lin_bisection(Gs,max_iter=50)

Gs1=nx.subgraph(Gs,Partition[0])
Gs2=nx.subgraph(Gs,Partition[1])

print(Partition)
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
nx.draw(Gs)
plt.title("part1")

plt.subplot(233)
nx.draw(Gs2)
plt.title("part2")
plt.show()