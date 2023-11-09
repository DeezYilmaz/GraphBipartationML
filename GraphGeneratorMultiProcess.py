import os
import networkx as nx
from itertools import combinations, groupby
import random
from networkx.algorithms import community
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import multiprocessing 
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

def GenerateGraphData(n):
    label_list=[]

    Graph_list=[]
    
    for i in range(n):
        G=gnp_random_connected_graph(random.randint(5,2500)*2,random.uniform(0,0.0015))
        trueCount=0
        print("on graph: ",i,end="\r")
        for i in range(1):
            Partition=community.kernighan_lin.kernighan_lin_bisection(G,max_iter=2000)
            Gs1=nx.subgraph(G,Partition[0])
            Gs2=nx.subgraph(G,Partition[1])
            if(nx.is_connected(Gs1) and nx.is_connected(Gs2) and len(Gs1.nodes())==len(Gs2.nodes())):
                trueCount+=1
                break
        if(trueCount>0):
            label_list=[True]+label_list
            Graph_list=[G]+Graph_list
        else:
            label_list=label_list+[False]
            Graph_list=Graph_list+[G]
        
    graph_data= {
        "Graph": Graph_list,
        "Labels":label_list
    }

    return graph_data

if __name__== '__main__':
    fileName="Graphdata"
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = "GraphDatas/"+fileName+".pickle"
    abs_file_path = os.path.join(script_dir, rel_path)


    p=multiprocessing.Pool(16)
    GraphDataList=p.map(GenerateGraphData,[150]*16)

    midPointDf= pd.DataFrame(GraphDataList)
    Glist=[]
    LabelList=[]


    for GraphLists in midPointDf['Graph']:
        Glist=Glist+GraphLists
    
    for LabLists in midPointDf['Labels']:
        LabelList=LabelList+LabLists

    finalData= {
        "Graph": Glist,
        "Labels":LabelList
    }
    graphDataFrame= pd.DataFrame(finalData)
    graphDataFrame.to_csv(fileName+".csv")
    pickle.dump(graphDataFrame,open(abs_file_path,"wb"))
    

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