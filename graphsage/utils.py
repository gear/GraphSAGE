from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
from collections import namedtuple

import networkx as nx
from networkx.readwrite import json_graph

version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

def random_flip(class_map, G, ratio=0.1):
    for n in G.nodes():
        if G.node[n]['val'] == False and G.node[n]['test'] == False:
            feat = class_map[n]
            for i, v in enumerate(feat):
                if np.random.random() < ratio:
                    feat[i] = 1 - v
            class_map[n] = feat
    return class_map
            

def load_data(prefix, feats_suf="", 
              normalize=True, 
              load_walks=False, 
              corrupt_label=None):
    # corrupt_label - Function to corrupt the labels of training data
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if len(feats_suf) == 0 and os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    elif os.path.exists(prefix+"-feats-"+feats_suf+".npy"):
        print("Load an alternate feature set {}".format(feats_suf))
        feats = np.load(prefix+"-feats-"+feats_suf+".npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}
    if corrupt_label is not None:
        class_map = corrupt_label(class_map, G)

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map


default_config = namedtuple('config', ['g_func_name', 'g_func_args', 'eps'])
default_config.g_func_name = 'barabasi_albert_graph'
default_config.g_func_args = {'n': 200, 'm':10}
default_config.eps = 0.5
default_config.num_train_per_class = 5
default_config.num_val_per_class = 5
default_config.feat_dim = 50
def load_data_highfreq(config=default_config):
    """Config is a namedtuple to config the random generation process"""
    g_gen_func = getattr(nx.random_graphs, config.g_func_name)
    G = g_gen_func(**config.g_func_args)
    labels_1 = frozenset(nx.algorithms.maximal_independent_set(G))
    labels_0 = frozenset([i for i in G.nodes() if i not in labels_1])  
    id_map = G.nodes() 
    walks = None
    class_map = dict()
    for i in G.nodes():
        class_map[i] = [0, 1] if i in labels_1 else [1, 0]
    feats = np.array([np.random.normal(config.eps, 1.0, config.feat_dim) \
                      if i in labels_1 else \
                      np.random.normal(-config.eps, 1.0, config.feat_dim) \
                      for i in G.nodes()])
    labels_1 = list(labels_1)
    labels_0 = list(labels_0)
    np.random.shuffle(labels_1)
    np.random.shuffle(labels_0)
    train_idx = []
    train_idx.extend(labels_1[:config.num_train_per_class])
    train_idx.extend(labels_0[:config.num_train_per_class])
    val_idx = []
    val_idx.extend(labels_1[config.num_train_per_class:config.num_train_per_class+config.num_val_per_class])
    val_idx.extend(labels_0[config.num_train_per_class:config.num_train_per_class+config.num_val_per_class])
    val_set = set(val_idx)
    train_set = set(train_idx) 
    for i in G.nodes():
        if i in train_set:
            G.node[i]['val'] = False
            G.node[i]['test'] = False
        elif i in val_set:
            G.node[i]['val'] = True
            G.node[i]['test'] = False
        else:
            G.node[i]['val'] = False
            G.node[i]['test'] = True
    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
    return G, feats, id_map, walks, class_map


def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for _ in range(num_walks):
            curr_node = node
            for _ in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
