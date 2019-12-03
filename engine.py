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
                   get_ratio_scale, get_XYZS_from_res_list,
                   saveh5, saveKpListToTxt, update_affine, loadh5)

best_val_loss_filename = "best_val_loss.h5"
best_step_filename = "step.h5"


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
            restore_res = self.restore_network()
            if not restore_res:
                raise RuntimeError("Could not load network weights!")
            
    def restore_network(self):
        """Restore training status"""
    
        # Skip if there's no saver of this subtask
        if self.config.subtask not in self.saver:
            return False
    
        is_loaded = False
    
        # Check if pretrain weight file is specified
        predir = getattr(self.config, "pretrained_{}".format(self.config.subtask))
        # Try loading the old weights
        is_loaded += self.load_legacy_network(self.config.subtask, predir)
        # Try loading the tensorflow weights
        is_loaded += self.load_network(predir)
    
        # Load network using tensorflow saver
        logdir = os.path.join(self.config.logdir, self.config.subtask)
        is_loaded += self.load_network(logdir)
    
        return is_loaded

    def load_legacy_network(self, load_dir):
        """Load function for our old framework"""
    
        print("[{}] Checking if old pre-trained weights exists in {}"
              "".format(self.config.subtask, load_dir))
        model_file = os.path.join(load_dir, "model.h5")
        norm_file = os.path.join(load_dir, "norm.h5")
        base_file = os.path.join(load_dir, "base.h5")
    
        if os.path.exists(model_file) and os.path.exists(norm_file) and \
           os.path.exists(base_file):
            model = loadh5(model_file)
            norm = loadh5(norm_file)
            base = loadh5(base_file)
            # Load the input normalization parameters.
            with self.graph.as_default():
                self.network.mean["kp"] = float(norm["mean_x"])
                self.network.mean["ori"] = float(norm["mean_x"])
                self.network.mean["desc"] = float(base["patch-mean"])
                self.network.std["kp"] = float(norm["std_x"])
                self.network.std["ori"] = float(norm["std_x"])
                self.network.std["desc"] = float(base["patch-std"])
                # Load weights for the component
                self.network.legacy_load_func[self.config.subtask](self.sess, model)
                print("[{}] Loaded previously trained weights".format(self.config.subtask))
            return True
        else:
            print("[{}] No pretrained weights from the old framework"
                  "".format(self.config.subtask))
            return False
    
    def load_network(self, load_dir):
        """Load function for our new framework"""
    
        print("[{}] Checking if previous Tensorflow run exists in {}"
              "".format(self.config.subtask, load_dir))
        latest_checkpoint = tf.train.latest_checkpoint(load_dir)
        if latest_checkpoint is not None:
            # Load parameters
            with self.graph.as_default():
                self.saver[self.config.subtask].restore(
                    self.sess,
                    latest_checkpoint
                )
                print("[{}] Loaded previously trained weights".format(self.config.subtask))
                # Save mean std (falls back to default if non-existent)
                if os.path.exists(os.path.join(load_dir, "mean.h5")):
                    self.network.mean = loadh5(os.path.join(load_dir, "mean.h5"))
                    self.network.std = loadh5(os.path.join(load_dir, "std.h5"))
                    print("[{}] Loaded input normalizers".format(self.config.subtask))
                # Load best validation result
                self.best_val_loss[self.config.subtask] = loadh5(
                    os.path.join(load_dir, best_val_loss_filename)
                )[self.config.subtask]
                print("[{}] Loaded best validation result = {}".format(
                    self.config.subtask,self.best_val_loss[self.config.subtask]))
                # Load best validation result
                self.best_step[self.config.subtask] = loadh5(
                    os.path.join(load_dir, best_step_filename)
                )[self.config.subtask]
                print("[{}] Loaded best step = {}".format(
                    self.config.subtask, self.best_step[self.config.subtask]))
    
            return True
    
        else:
            print("[{}] No previous Tensorflow result".format(self.config.subtask))
    
            return False

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
