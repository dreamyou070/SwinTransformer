import torch
"""
query = torch.randn(64,3,49,32)
key = torch.randn(64,3,49,32)
value = torch.randn(64,3,49,32)
attn = query @ key.transpose(-2, -1)
"""
window_size = 2
coords = torch.stack(torch.meshgrid([torch.arange(window_size),torch.arange(window_size)]))  # 2, Wh, Ww -> x_cord, y_cord
coords_flatten = torch.flatten(coords, start_dim = 1)  # [0,1,2, ...] [012 / ... ]
# ------------------------------------------------------------------------------------------------------------ #
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
print(f'relative_coords : {relative_coords}')
relative_coords[:, :, 0] += window_size - 1  # shift to start from 0 (add 6)
relative_coords[:, :, 1] += window_size - 1
relative_coords[:, :, 0] *= 2 * window_size - 1
print(f'relative_coords : {relative_coords}')
relative_position_index = relative_coords.sum(-1)
#print(f'relative_position_index : {relative_position_index}')
"""

  # 
"""