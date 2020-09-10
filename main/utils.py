import numpy as np
import scipy.sparse as sp
import torch as th


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[:len(labels)]
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def adj_preprocess(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(axis=1).A1
    deg = sp.diags(rowsum ** (-0.5))
    adj_ = deg @ adj_ @ deg.tocsr()

    return adj_


def atan_norm(x):
    return 2 * th.atan(x) / th.Tensor([np.pi]).to(x.device)


def tanh_norm(x):
    return th.tanh(x)


def sigmoid_norm(x):
    return 2 / (1 + th.exp(-x)) - 1


def feat_norm(norm_type):
    if norm_type == 'atan':
        return atan_norm
    elif norm_type == 'tanh':
        return tanh_norm
    elif norm_type == 'sigmoid':
        return sigmoid_norm
    else:
        return lambda x: x


def compute_acc(pred, labels, mask=None):
    if mask is None:
        return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    else:
        return (th.argmax(pred[mask], dim=1) == labels[mask[:len(labels)]]).float().sum() / np.sum(mask)


def clip_by_tensor(t, t_min, t_max):
    t = t.float()
    t_min = t_min.float()
    t_max = t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max

    return result
