from os import makedirs
from os.path import exists, join, isfile, dirname, abspath
import tensorflow as tf
import numpy as np
import yaml
import pickle

BASE_DIR = dirname(abspath(__file__))

data_config = join(BASE_DIR, 'utils', 'semantic-kitti.yaml')
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map_inv"]

# make lookup table for mapping
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        my_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.compat.v1.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open('log_test_2' + str(dataset.val_split) + '.txt', 'a')

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.compat.v1.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=c_proto)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)
        self.test_probs = 0
        self.idx = 0

    def test(self, model, dataset):

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)
        self.test_probs = [np.zeros(shape=[len(l), model.config.num_classes], dtype=np.float16)
                           for l in dataset.possibility]

        test_path = join(dirname(dataset.dataset_path), 'test', 'sequences')
        makedirs(test_path) if not exists(test_path) else None
        for seq_id in range(11, 22, 1):
            makedirs(join(test_path, str(seq_id))) if not exists(join(test_path, str(seq_id))) else None
            makedirs(join(test_path, str(seq_id), 'predictions')) if not exists(
                join(test_path, str(seq_id), 'predictions')) else None
        test_smooth = 0.98
        epoch_ind = 0

        while True:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, labels, point_inds, cloud_inds = self.sess.run(ops, {model.is_training: False})
                if self.idx % 10 == 0:
                    print('step ' + str(self.idx))
                self.idx += 1
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size,
                                                           model.config.num_points,
                                                           model.config.num_classes])
                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_inds[j, :]
                    c_i = cloud_inds[j][0]
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs

            except tf.errors.OutOfRangeError:
                new_min = np.min(dataset.min_possibility)
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_ind, new_min), self.Log_file)
                if np.min(dataset.min_possibility) > 0.5:
                    log_out(' Min possibility = {:.1f}'.format(np.min(dataset.min_possibility)), self.Log_file)
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    for j in range(len(self.test_probs)):
                        test_file_name = dataset.test_list[j]
                        frame = test_file_name.split('/')[-1][:-4]
                        proj_path = join(dataset.dataset_path, dataset.test_scan_number, 'proj')
                        proj_file = join(proj_path, str(frame) + '_proj.pkl')
                        if isfile(proj_file):
                            with open(proj_file, 'rb') as f:
                                proj_inds = pickle.load(f)
                        probs = self.test_probs[j][proj_inds[0], :]
                        pred = np.argmax(probs, 1)
                        store_path = join(test_path, dataset.test_scan_number, 'predictions',
                                          str(frame) + '.label')
                        pred = pred + 1
                        pred = pred.astype(np.uint32)
                        upper_half = pred >> 16  # get upper half for instances
                        lower_half = pred & 0xFFFF  # get lower half for semantics
                        lower_half = remap_lut[lower_half]  # do the remapping of semantics
                        pred = (upper_half << 16) + lower_half  # reconstruct full label
                        pred = pred.astype(np.uint32)
                        pred.tofile(store_path)
                    log_out(str(dataset.test_scan_number) + ' finished', self.Log_file)
                    self.sess.close()
                    return
                self.sess.run(dataset.test_init_op)
                epoch_ind += 1
                continue
