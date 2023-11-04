import networkx as nx
from itertools import combinations, groupby
import random
from networkx.algorithms import community
import matplotlib.pyplot as plt
import pandas as pd
import pickle

Graph_df=pickle.load(open('GraphData.csv','rb'))
Gs= Graph_df.iloc[3]['Graph']

print(Graph_df)
