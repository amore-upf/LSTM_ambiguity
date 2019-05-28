
import torch
from torch.autograd import Variable


def batchify(data, bsz, cuda=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


def get_batch(source, i, bptt, mode, evaluation):
    if mode == "forward":
        return _get_batch(source, i, bptt, evaluation)
    elif mode == "backward":
        return _get_batch(flip(source, dim=0), i, bptt, evaluation)
    elif "bidir" in mode:
        return _get_batch_bidirectional(source, i, bptt, evaluation)


def _get_batch(source, i, bptt, evaluation):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], requires_grad=False)  # Here it starts from one
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1),
                                                    ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def _get_batch_bidirectional(data, i, seq_len, evaluation):
    seq_len = min(seq_len, len(data) - 2 - i)
    data_f, targets = _get_batch(data, i, seq_len, evaluation)
    data_b = Variable(data[i + 2:i+seq_len + 2], requires_grad=False)
    data_b = flip(data_b, dim=0)
    return (data_f, data_b), targets


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def update_hidden(model, mode, hidden, batch_size):
    if mode == "bidir":
        # fixed window length, suboptimal for tokens close to the boundaries
        return model.init_hidden(batch_size)
    elif mode == "bidir_cont": # forward continuous
        hidden1 = repackage_hidden(hidden)
        hidden2 = model.init_hidden(batch_size)
        # keep forward, restart backward
        return hidden1, hidden2
    elif mode == "forward" or mode == "backward":
        return repackage_hidden(hidden)    # continuous hidden state


