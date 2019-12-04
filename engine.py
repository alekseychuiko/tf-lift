# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:25:00 2019

@author: achuiko
"""
import os
import time
import cv2
import numpy as np
import tensorflow as tf

from six.moves import xrange
from datasets.test import Dataset
from networks.lift import Network
import copy
from datasets.eccv2016.helper import load_patches

from utils import (IDX_ANGLE, XYZS2kpList, get_patch_size,
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
        
    def test_squeeze(self, data):
        with self.graph.as_default():
            return self.network.test(
                    self.config.subtask,
                    data
                ).squeeze()

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
        #self.graph_desc = ImportGraph(config, 'desc', self.dataset, tfconfig)


    def compute_kp(self, image):
        """Compute Keypoints.

        LATER: Clean up code

        """

        total_time = 0.0

        # check size
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Multiscale Testing
        scl_intv = self.config.test_scl_intv
        # min_scale_log2 = 1  # min scale = 2
        # max_scale_log2 = 4  # max scale = 16
        min_scale_log2 = self.config.test_min_scale_log2
        max_scale_log2 = self.config.test_max_scale_log2
        # Test starting with double scale if small image
        min_hw = np.min(image.shape[:2])
        # for the case of testing on same scale, do not double scale
        if min_hw <= 1600 and min_scale_log2!=max_scale_log2:
            print("INFO: Testing double scale")
            min_scale_log2 -= 1
        # range of scales to check
        num_division = (max_scale_log2 - min_scale_log2) * (scl_intv + 1) + 1
        scales_to_test = 2**np.linspace(min_scale_log2, max_scale_log2,
                                        num_division)

        # convert scale to image resizes
        resize_to_test = ((float(self.config.kp_input_size - 1) / 2.0) /
                          (get_ratio_scale(self.config) * scales_to_test))

        # check if resize is valid
        min_hw_after_resize = resize_to_test * np.min(image.shape[:2])
        is_resize_valid = min_hw_after_resize > self.config.kp_filter_size + 1

        # if there are invalid scales and resizes
        if not np.prod(is_resize_valid):
            # find first invalid
            first_invalid = np.where(~is_resize_valid)[0][0]

            # remove scales from testing
            scales_to_test = scales_to_test[:first_invalid]
            resize_to_test = resize_to_test[:first_invalid]

        print('resize to test is {}'.format(resize_to_test))
        print('scales to test is {}'.format(scales_to_test))

        # Run for each scale
        test_res_list = []
        for resize in resize_to_test:

            # resize according to how we extracted patches when training
            new_height = np.cast['int'](np.round(image_height * resize))
            new_width = np.cast['int'](np.round(image_width * resize))
            start_time = time.clock()
            image = cv2.resize(image, (new_width, new_height))
            end_time = time.clock()
            resize_time = (end_time - start_time) * 1000.0
            print("Time taken to resize image is {}ms".format(
                resize_time
            ))
            total_time += resize_time

            # run test
            # LATER: Compatibility with the previous implementations
            start_time = time.clock()

            # Run the network to get the scoremap (the valid region only)
            scoremap = None
            if self.config.test_kp_use_tensorflow:
                scoremap = self.graph_kp.test_squeeze(image.reshape(1, new_height, new_width, 1))
            else:
                # OpenCV Version
                raise NotImplementedError(
                    "TODO: Implement OpenCV Version")

            end_time = time.clock()
            compute_time = (end_time - start_time) * 1000.0
            print("Time taken for image size {}"
                  " is {} milliseconds".format(
                      image.shape, compute_time))

            total_time += compute_time

            # pad invalid regions and add to list
            start_time = time.clock()
            test_res_list.append(
                np.pad(scoremap, int((self.config.kp_filter_size - 1) / 2),
                       mode='constant',
                       constant_values=-np.inf)
            )
            end_time = time.clock()
            pad_time = (end_time - start_time) * 1000.0
            print("Time taken for padding and stacking is {} ms".format(
                pad_time
            ))
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
        #print(res_list[0][400:500,300:400])
        # check whether the return result for socre is right
        XYZS = get_XYZS_from_res_list(
            res_list, resize_to_test, scales_to_test, nearby, edge_th,
            scl_intv, nms_intv, do_interpolation=True,
        )
        end_time = time.clock()
        XYZS = XYZS[:self.config.test_num_keypoint]

        nms_time = (end_time - start_time) * 1000.0
        print("NMS time is {} ms".format(nms_time))
        total_time += nms_time
        print("Total time for detection is {} ms".format(total_time))
        # ------------------------------------------------------------------------
        # Save as keypoint file to be used by the oxford thing
        print("Turning into kp_list")
        kp_list = XYZS2kpList(XYZS)  # note that this is already sorted
        return self.compute_ori(image, kp_list)
    
    def compute_ori(self, image, keypoints):
        """Compute Orientations """

        total_time = 0.0

        # Read image
        start_time = time.clock()
        cur_data = self.load_data(image, keypoints)
        end_time = time.clock()
        load_time = (end_time - start_time) * 1000.0
        print("Time taken to load patches is {} ms".format(
            load_time
        ))
        total_time += load_time

        # -------------------------------------------------------------------------
        # Test using the test function
        start_time = time.clock()
        oris = self._test_multibatch(self.graph_ori, cur_data)
        end_time = time.clock()
        compute_time = (end_time - start_time) * 1000.0
        print("Time taken to compute is {} ms".format(
            compute_time
        ))
        total_time += compute_time

        # update keypoints and save as new
        start_time = time.clock()
        kps = cur_data["kps"]
        for idxkp in xrange(len(kps)):
            kps[idxkp][IDX_ANGLE] = oris[idxkp] * 180.0 / np.pi % 360.0
            kps[idxkp] = update_affine(kps[idxkp])
        end_time = time.clock()
        update_time = (end_time - start_time) * 1000.0
        print("Time taken to update is {} ms".format(
            update_time
        ))
        total_time += update_time
        print("Total time for orientation is {} ms".format(total_time))

        # save as new keypoints
        return kps
        
        
    def load_data(self, img, keypoint):
        """Returns the patch, given the keypoint structure

        LATER: Cleanup. We currently re-use the utils we had from data
               extraction.

        """
        kp = np.asarray(keypoint)
        
        in_dim = 1

        # Use load patches function
        # Assign dummy values to y, ID, angle
        y = np.zeros((len(kp),))
        ID = np.zeros((len(kp),), dtype='int64')
        # angle = np.zeros((len(kp),))
        angle = np.pi / 180.0 * kp[:, IDX_ANGLE]  # store angle in radians

        # load patches with id (drop out of boundary)
        bPerturb = False
        fPerturbInfo = np.zeros((3,))
        dataset = load_patches(img, kp, y, ID, angle,
                               get_ratio_scale(self.config), 1.0,
                               int(get_patch_size(self.config)),
                               int(self.config.desc_input_size), in_dim,
                               bPerturb, fPerturbInfo, bReturnCoords=True,
                               is_test=True)

        # Change old dataset return structure to necessary data
        x = dataset[0]
        # y = dataset[1]
        # ID = dataset[2]
        pos = dataset[3]
        angle = dataset[4]
        coords = dataset[5]

        # Return the dictionary structure
        cur_data = {}
        cur_data["patch"] = np.transpose(x, (0, 2, 3, 1))  # In NHWC
        cur_data["kps"] = coords
        cur_data["xyz"] = pos
        # Make sure that angle is a Nx1 vector
        cur_data["angle"] = np.reshape(angle, (-1, 1))

        return cur_data
    
    def _test_multibatch(self, graph, cur_data):
        """A sub test routine.

        We do this since the spatial transformer implementation in tensorflow
        does not like undetermined batch sizes.

        LATER: Bypass the spatial transformer...somehow
        LATER: Fix the multibatch testing

        """
        batch_size = self.config.batch_size
        num_patch = len(cur_data["patch"])
        num_batch = int(np.ceil(float(num_patch) / float(batch_size)))
        # Initialize the batch items
        cur_batch = {}
        for _key in cur_data:
            cur_batch[_key] = np.zeros_like(cur_data[_key][:batch_size])

        # Do muiltiple times
        res = []
        for _idx_batch in xrange(num_batch):
            # start of the batch
            bs = _idx_batch * batch_size
            # end of the batch
            be = min(num_patch, (_idx_batch + 1) * batch_size)
            # number of elements in batch
            bn = be - bs
            for _key in cur_data:
                cur_batch[_key][:bn] = cur_data[_key][bs:be]
            cur_res = graph.test_squeeze(cur_batch)[:bn]
            # Append
            res.append(cur_res)

        return np.concatenate(res, axis=0)
    