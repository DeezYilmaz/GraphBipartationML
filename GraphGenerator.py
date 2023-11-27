import math
import networkx as nx
from itertools import combinations, groupby
import random
from networkx.algorithms import community
import matplotlib.pyplot as plt
import pandas as pd
import pickle
def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G



fig = plt.subplots(1, figsize=(12,10))

label_list=[]
Graph_list=[]

"""
def generate_barabasi_albert_graph(num_nodes_range, m_range):
    num_nodes=random.randint(*num_nodes_range)
    m=random.randint(*m_range)*2
    return nx.barabasi_albert_graph(num_nodes, m)

# Example usage:
num_instances = 100
num_nodes = (50,150)
m_parameter = (1,1)

barabasi_albert_graphs = [generate_barabasi_albert_graph(num_nodes, m_parameter) for _ in range(num_instances)]
trueCount=0


for G in barabasi_albert_graphs:
    Partition=community.kernighan_lin.kernighan_lin_bisection(G,max_iter=(int)(0.4*len(G.nodes())*math.log10(len(G.nodes()))))
    Gs1=nx.subgraph(G,Partition[0])
    Gs2=nx.subgraph(G,Partition[1])
    if(nx.is_connected(Gs1) and nx.is_connected(Gs2) and len(Gs1.nodes()) <= (len(G.nodes())/2+len(G.nodes())*0.01) 
        and len(Gs1.nodes()) >= (len(G.nodes())/2-len(G.nodes())*0.01)   ):
        trueCount+=1

print(trueCount)
"""
def viewGraph(G):
    nx.draw_networkx(G)
    plt.show()

def getOptimalProb():
    
    edgeProbMin= 0
    edgeProbMax=0.1
    for rng in range(1,8):
        percentage=0
        iteration=150
        while(percentage>55 or percentage<45):
            trueCount=0
            edgeProb= (edgeProbMin+edgeProbMax)/2
            for i in range(iteration):
                nodeCount=random.randint(150*rng,150*(rng+1))*2
                G=gnp_random_connected_graph(nodeCount,edgeProb)
                #print("on graph: ",i,end="\r")
                
                Partition=community.kernighan_lin.kernighan_lin_bisection(G,max_iter=int( (0.4*len(G.nodes())*math.log10(len(G.nodes())))) )
                Gs1=nx.subgraph(G,Partition[0])
                Gs2=nx.subgraph(G,Partition[1])
                if(nx.is_connected(Gs1) and nx.is_connected(Gs2) and len(Gs1.nodes()) <= (len(G.nodes())/2+len(G.nodes())*0.01) 
                        and len(Gs1.nodes()) >= (len(G.nodes())/2-len(G.nodes())*0.01)   ):
                    trueCount+=1

            percentage=(trueCount/iteration)*100
            if(percentage>55):
                edgeProbMax=edgeProb
            if(percentage<45):
                edgeProbMin=edgeProb
        print(rng,"%i-%i" %(150*rng*2,150*(rng+1)*2) ,"%.5f" % edgeProbMin,"%.5f" % edgeProbMax, "%.2f" % percentage)
        edgeProbMin=0
        

getOptimalProb()
    

exit()
graph_data= {
    "Graph": Graph_list,
    "Labels":label_list
}
graphDataFrame= pd.DataFrame(graph_data)
graphDataFrame.to_csv("Graphdata.csv")
pickle.dump(graphDataFrame,open('GraphData.pickle','wb'))

print(graphDataFrame)

"""
    Eski plot kodu
"""
# iterable= max(nx.connected_components(G), key=len)
# Gs=nx.subgraph(G,iterable)


# Partition=community.kernighan_lin_bisection(Gs,max_iter=50)

# Gs1=nx.subgraph(Gs,Partition[0])
# Gs2=nx.subgraph(Gs,Partition[1])


# if(nx.is_connected(Gs1) and nx.is_connected(Gs2) and len(Gs1.nodes())==len(Gs2.nodes())):
#     print("useable")
# else:
#     print("not usable")



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