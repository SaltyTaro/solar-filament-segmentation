"""Draft detector iterator guard for within-epoch checkpoint resumption."""

import torch


def training_iterator(loader, continuing_epoch: bool = False):
    """Avoid an extra global RNG draw when recreating an interrupted iterator.

    Torch DataLoader creates a base seed even with no workers. A new epoch
    keeps that normal draw; resuming an existing epoch has already made it.
    Detector samples use local seed/epoch/sample generators, so replaying
    skipped samples consumes no global Torch state.
    """
    if not continuing_epoch:
        return iter(loader)
    rng = torch.get_rng_state()
    iterator = iter(loader)
    torch.set_rng_state(rng)
    return iterator
