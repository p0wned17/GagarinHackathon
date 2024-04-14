import os

import torch

from collections import OrderedDict


def save_checkpoint(model, optimizer, scheduler, epoch, outdir, epoch_avg_acc):
    """Saves checkpoint to disk"""
    filename = "model_{:04d}_{}.pt".format(epoch, epoch_avg_acc)
    directory = outdir
    filename = os.path.join(directory, filename)
    best_filename = os.path.join(directory, "best.pt")
    weights = model.state_dict()
    state = OrderedDict(
        [
            ("state_dict", weights),
            ("optimizer", optimizer.state_dict()),
            ("scheduler", scheduler.state_dict()),
            ("epoch", epoch),
        ]
    )
    if os.path.isfile(best_filename):
        os.remove(best_filename)
    torch.save(state, filename)
    torch.save(state, best_filename)
    os.chmod(filename, 0o777)
    os.chmod(best_filename, 0o777)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self):
        return self.val, self.avg
