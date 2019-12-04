# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:25:00 2019

@author: achuiko
"""
import os
import time
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
        is_loaded += self.load_legacy_network(predir)
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
        self.graph_kp = ImportGraph(config, 'kp', self.dataset, tfconfig)
        self.graph_ori = ImportGraph(config, 'ori', self.dataset, tfconfig)
        self.graph_desc = ImportGraph(config, 'desc', self.dataset, tfconfig)


    def compute_kp(self, image):
        """Compute Keypoints.

        LATER: Clean up code

        """
        total_time = 0.0
        start_time = time.clock()


        # check size
        image_height = image.shape[0]
        image_width = image.shape[1]

        scoremap = None
        scoremap = self.network.test(self.config.subtask, image.reshape(1, image_height, image_width, 1)).squeeze()

        end_time = time.clock()
        compute_time = (end_time - start_time) * 1000.0
        print("Time taken for image size {} is {} milliseconds".format(
                      image.shape, compute_time))

        total_time += compute_time

        # pad invalid regions and add to list
        start_time = time.clock()
        test_res_list.append(np.pad(scoremap, int((self.config.kp_filter_size - 1) / 2),
                       mode='constant',
                       constant_values=-np.inf)
        )
        end_time = time.clock()
        pad_time = (end_time - start_time) * 1000.0
        print("Time taken for padding and stacking is {} ms".format(pad_time))
        total_time += pad_time

        # ------------------------------------------------------------------------
        # Non-max suppresion and draw.

        # The nonmax suppression implemented here is very very slow. Consider
        # this as just a proof of concept implementation as of now.

        # Standard nearby : nonmax will check approximately the same area as
        # descriptor support region.
        nearby = int(np.round(
            (0.5 * (self.config.kp_input_size - 1.0) *
             float(self.config.desc_input_size) /
             float(get_patch_size(self.config)))
        ))
        fNearbyRatio = self.config.test_nearby_ratio
        # Multiply by quarter to compensate
        fNearbyRatio *= 0.25
        nearby = int(np.round(nearby * fNearbyRatio))
        nearby = max(nearby, 1)

        nms_intv = self.config.test_nms_intv
        edge_th = self.config.test_edge_th

        print("Performing NMS")
        start_time = time.clock()
        res_list = test_res_list
        # check whether the return result for socre is right
        XYZS = get_XYZS_from_res_list(
            res_list, resize_to_test, scales_to_test, nearby, edge_th,
            scl_intv, nms_intv, do_interpolation=True,
        )
        end_time = time.clock()
        XYZS = XYZS[:self.config.test_num_keypoint]

        # For debugging
        # TODO: Remove below
        draw_XYZS_to_img(XYZS, image_color, self.config.test_out_file + '.jpg')

        nms_time = (end_time - start_time) * 1000.0
        print("NMS time is {} ms".format(nms_time))
        total_time += nms_time
        print("Total time for detection is {} ms".format(total_time))

        # ------------------------------------------------------------------------
        # Save as keypoint file to be used by the oxford thing
        print("Turning into kp_list")
        kp_list = XYZS2kpList(XYZS)  # note that this is already sorted

        print("Saving to txt")
        saveKpListToTxt(kp_list, None, self.config.test_out_file)