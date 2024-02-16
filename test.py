import torch
from torch import nn
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


x = torch.randn(1,8,8,320)
window_size = 4
windows = window_partition(x, window_size)

global_hidden_states = torch.randn(1, 64, 320)
b = global_hidden_states.shape[0]
print(b)
local_hidden_states = torch.randn(4, 16, 320)
local_hidden_states = local_hidden_states.view(b, -1, 320)
print(local_hidden_states.shape)  # torch.Size([1, 64, 320])