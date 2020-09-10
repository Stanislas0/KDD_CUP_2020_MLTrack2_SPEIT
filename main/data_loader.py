import dgl
import pickle
import numpy as np


class KddDataset(object):
    def __init__(self, adj_path, feat_path, label_path, num_class=20, test_size=50000, indices=None):
        fg = open(adj_path, 'rb')
        self.adj = pickle.load(fg)
        self.features = np.load(feat_path)
        self.labels = np.load(label_path)

        self.num_labels = num_class
        size_raw = self.features.shape[0]
        size_reduced = size_raw - test_size

        if indices is None:
            indices_train = np.array([i for i in range(size_reduced - test_size)])
            indices_val = np.array([i for i in range(size_reduced - test_size, size_reduced)])
            indices_test = np.array([i for i in range(size_raw - test_size, size_raw)])
        else:
            indices_train, indices_val, indices_test = indices

        self.train_mask = np.zeros(size_reduced).astype(bool)
        self.val_mask = np.zeros(size_reduced).astype(bool)
        self.test_mask = np.zeros(size_raw).astype(bool)
        self.train_mask[indices_train] = True
        self.val_mask[indices_val] = True
        self.test_mask[indices_test] = True

        self.graph = dgl.DGLGraph()
        self.graph.from_scipy_sparse_matrix(self.adj)

        print('Finished data loading.')
        print('NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('NumEdges: {}'.format(self.graph.number_of_edges()))
        print('NumFeats: {}'.format(self.features.shape[1]))
        print('NumClasses: {}'.format(self.num_labels))
        print('NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
