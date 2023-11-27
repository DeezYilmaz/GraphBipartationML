import math
import networkx as nx
from itertools import combinations, groupby
import random
from networkx.algorithms import community
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

filename="Graphdata"
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "GraphDatas/"+filename+".pickle"
abs_file_path = os.path.join(script_dir, rel_path)

Graph_df=pickle.load(open(abs_file_path,'rb'))

barData=[]