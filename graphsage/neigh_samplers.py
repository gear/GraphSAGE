from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        print(adj_lists)
        return adj_lists


class ERNeighborSampler(Layer):
    """
    Sample neighbors uniformly from the whole graph.
    This method is equivalent with UniformNeighborSampler on
    an Erdos-Renyi graph with the same number of nodes and edges.
    """
    def __init__(self, adj_info, **kwargs):
        super(ERNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_shape = self.adj_info.get_shape().as_list()
        num_nodes = adj_shape[0]
        adj_lists = tf.random_uniform((ids.get_shape()[0], num_samples), 
                                      minval=0, 
                                      maxval=num_nodes, 
                                      dtype=tf.int32)
        return adj_lists

