import torch


_ALLOW_MPS = False  # MPS support is currently not stable enough


def get_default_device():
    if not torch.cuda.is_available():
        if _ALLOW_MPS:
            try:
                device = torch.device('mps')
            except BaseException as inst:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
        return device
    else:
        return torch.device('cuda')


default_device = get_default_device()

