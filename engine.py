# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:25:00 2019

@author: achuiko
"""
import os
import numpy as np
import tensorflow as tf

from datasets.test import Dataset
from networks.lift import Network
import copy

from utils import (IDX_ANGLE, XYZS2kpList, draw_XYZS_to_img, get_patch_size,
                   get_ratio_scale, get_XYZS_from_res_list, restore_network,
                   saveh5, saveKpListToTxt, update_affine, loadh5)


class ImportGraph(object):
    def __init__(self, config, subtask, dataset, tfconfig):
        logdir = os.path.join(config.logdir, subtask)
        self.config = copy.deepcopy(config)
        self.config.subtask = subtask
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=tfconfig)
        with self.graph.as_default():
            if os.path.exists(os.path.join(logdir, "mean.h5")):
                training_mean = loadh5(os.path.join(logdir, "mean.h5"))
                training_std = loadh5(os.path.join(logdir, "std.h5"))
                print("[{}] Loaded input normalizers for testing".format(
                    subtask))
    
                # Create the model instance
                self.network = Network(self.sess, self.config, dataset, {
                                       'mean': training_mean, 'std': training_std})
            else:
                self.network = Network(self.sess, self.config, dataset)
    
            self.saver = {}
            self.best_val_loss = {}
            self.best_step = {}
            # Create the saver instance for both joint and the current subtask
            for _key in ["joint", subtask]:
                self.saver[_key] = tf.train.Saver(self.network.allparams[_key])

        # We have everything ready. We finalize and initialie the network here.
            self.sess.run(tf.global_variables_initializer())

class Engine(object):
    def __init__(self, config):
        self.config = config
        
        tf.reset_default_graph()
        self.rng = np.random.RandomState(config.random_seed)
        tf.set_random_seed(config.random_seed)
        
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        
        self.dataset = Dataset(self.config, self.rng)
        graph_kp = ImportGraph(config, 'kp', self.dataset, tfconfig)
        graph_ori = ImportGraph(config, 'ori', self.dataset, tfconfig)
        graph_desc = ImportGraph(config, 'desc', self.dataset, tfconfig)
