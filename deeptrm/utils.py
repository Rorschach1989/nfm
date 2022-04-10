import torch


def get_default_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        return torch.device('cuda')


default_device = get_default_device()

