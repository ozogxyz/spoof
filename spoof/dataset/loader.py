import numpy as np
import torch


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def threaded_loader(loader, iscuda, threads=1, batch_size=1, shuffle=False):
    """Get a data loader, given the dataset and some parameters.
    Note: worker init function is required for proper randomization between batches if using randomizer from NumPy.
    Note 2: Using randomizer from NumPy is not recommended.

    Parameters
    ----------
    loader : object[i] returns the i-th training example.
    iscuda : bool
    batch_size : int
    threads : int
    shuffle : true
    Returns
    -------
        a multi-threaded pytorch loader.
    """

    return torch.utils.data.DataLoader(
        loader,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=threads,
        pin_memory=iscuda,
        worker_init_fn=worker_init_fn,
    )
