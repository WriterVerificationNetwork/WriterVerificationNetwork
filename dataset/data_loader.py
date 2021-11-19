from torch.utils.data import DataLoader


class WriterDataLoader:

    def __init__(self, dataset, is_train, numb_threads, batch_size):
        self._is_train = is_train
        self.batch_size = batch_size
        self.numb_threads = numb_threads
        self.dataset = dataset

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(self.numb_threads),
            pin_memory=True,
            drop_last=self._is_train)



