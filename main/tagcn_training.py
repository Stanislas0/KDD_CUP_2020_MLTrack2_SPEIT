import os
import dgl
import time
import argparse
import numpy as np
import torch as th
import distutils.util
import torch.nn.functional as F

import utils
import models
import data_loader

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
dev = th.device('cuda' if th.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("training")
    argparser.add_argument('--adj-path', type=str, default='../data/adj_matrix_formal_stage.pkl')
    argparser.add_argument('--feat-path', type=str, default='../data/feature_formal_stage.npy')
    argparser.add_argument('--label-path', type=str, default='../data/train_labels_formal_stage.npy')
    argparser.add_argument('--output-dir', type=str, default='./saved_models/')
    argparser.add_argument('--output-name', type=str, default='tagcn_128_3.pkl')
    argparser.add_argument('--if-load-model', type=lambda x: bool(distutils.util.strtobool(x)), default=False)
    argparser.add_argument('--model-dir', type=str, default='./saved_models/')
    argparser.add_argument('--model-name', type=str, default='tagcn_128_3.pkl')
    argparser.add_argument('--num-epochs', type=int, default=5000)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--adj-norm', type=lambda x: bool(distutils.util.strtobool(x)), default=True)
    argparser.add_argument('--feat-norm', type=str, default=None)

    args = argparser.parse_args()
    print(vars(args))

    dataset = data_loader.KddDataset(args.adj_path, args.feat_path, args.label_path, indices)
    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    size_raw = features.shape[0]
    size_reduced = size_raw - 50000

    graph = dgl.DGLGraph()
    if args.adj_norm:
        adj = utils.adj_preprocess(adj)
    feat_norm_func = utils.feat_norm(args.feat_norm)
    graph.from_scipy_sparse_matrix(adj)
    features = th.FloatTensor(features).to(dev)
    features[th.where(features < -1.0)[0]] = 0
    features[th.where(features > 1.0)[0]] = 0
    features = feat_norm_func(features)
    labels = th.LongTensor(labels).to(dev)

    graph.ndata['features'] = features
    model = models.TAGCN(100, args.num_hidden, 20, args.num_layers, activation=F.leaky_relu, dropout=args.dropout)

    if args.if_load_model:
        model_states = th.load(os.path.join(args.model_dir, args.model_name), map_location=dev)
        model.load_state_dict(model_states)
    model = model.to(dev)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    dur = []
    for epoch in range(args.num_epochs):
        t0 = time.time()
        logits = model(graph, features).to(dev)
        logp = F.log_softmax(logits, 1)[:size_reduced]
        loss = F.nll_loss(logp[train_mask], labels[train_mask]).to(dev)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dur.append(time.time() - t0)

        if epoch % 10 == 0:
            train_acc = utils.compute_acc(logp, labels, train_mask)
            val_acc = utils.compute_acc(logp, labels, val_mask)
            print('Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} '
                  '| Time(s) {:.4f} | GPU {:.1f} MiB'.format(
                epoch, loss, train_acc, val_acc, np.mean(dur), th.cuda.max_memory_allocated() / 1000000))

    th.save(model.state_dict(), os.path.join(args.output_dir, args.output_name))
