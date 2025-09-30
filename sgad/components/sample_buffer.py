# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import DataLoader

from sgad.utils.data_utils import PreBatchedDataset


class BatchBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.batch_list = []

    def add(self, graph_states, grads):
        batch = [(graph_state, grad) for graph_state, grad in zip(graph_states, grads)]

        self.batch_list.extend(batch)

        if len(self.batch_list) > self.buffer_size:
            self.batch_list = self.batch_list[-self.buffer_size :]

    def get_data_loader(self, shuffle=True):
        dataset = PreBatchedDataset(self.batch_list)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=shuffle)
        return dataloader

    def save_state(self, filename):
        # TODO.
        pass

    def load_state(self, filename):
        # TODO.
        pass
