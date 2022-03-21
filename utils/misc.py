import torch


class EarlyStop:

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.losses = []
        self.best_loss = 99999999

    def should_stop(self, loss):
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
        if len(self.losses) <= self.n_epochs:
            return False
        best_loss_pos = self.losses.index(self.best_loss)
        if len(self.losses) - best_loss_pos <= self.n_epochs:
            return False
        return True


def map_location(cuda):
    if torch.cuda.is_available() and cuda:
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]