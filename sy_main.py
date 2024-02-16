import torch
from models import build_model
import argparse
from config import get_config
import os, sys
kernel_path = os.path.abspath(os.path.join('..'))
sys.path.append(kernel_path)
from kernels.window_process.window_process import WindowProcess, WindowProcessReverse


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
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def main(args, config) :

    model = build_model(config)
    patch_embed = model.patch_embed
    #patch_embed: PatchEmbed((proj): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
    #(norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True))

    print(f'patch_embed : {patch_embed}')

    print(f' step 1. patch emb')
    images = torch.randn(1, 3, 224, 224)     # batch, 3, 224, 224
    x = patch_embed(images) # batch, 56*56, 96
    x = model.pos_drop(x)
    print(f'after patch embed, x (batch, 56*56+1, 96) : {x.shape}')

    print(f' step 2. transformer block')
    for basiclayer in model.layers:

        for swintransformerblock in basiclayer.blocks:

            H, W = swintransformerblock.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
            shortcut = x
            x = swintransformerblock.norm1(x)
            x = x.view(B, H, W, C)
            # --------------------------------------------------------------------------------------------------------
            # cyclic shift
            if swintransformerblock.shift_size > 0:
                if not swintransformerblock.fused_window_process:
                    shifted_x = torch.roll(x, shifts=(-swintransformerblock.shift_size, -swintransformerblock.shift_size), dims=(1, 2))
                    # partition windows
                    x_windows = window_partition(shifted_x, swintransformerblock.window_size)  # nW*B, window_size, window_size, C
                else:
                    x_windows = WindowProcess.apply(x, B, H, W, C, -swintransformerblock.shift_size, swintransformerblock.window_size)
            else:
                shifted_x = x
                # partition windows
                x_windows = window_partition(shifted_x,
                                             swintransformerblock.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, 7, 7, C)  # nW*B, window_size*window_size, C
            print(f'after window partitioning, x_windows (nW*B, window_size*window_size, C) : {x_windows.shape}')
            # W-MSA/SW-MSA
            attn_windows = swintransformerblock.attn(x_windows, mask=swintransformerblock.attn_mask)  # nW*B, window_size*window_size, C
            print(f'after attn, attn_windows (nW*B, window_size*window_size, C) : {attn_windows.shape}')

            # merge windows
            attn_windows = attn_windows.view(-1, swintransformerblock.window_size, swintransformerblock.window_size, C)

            # reverse cyclic shift
            if swintransformerblock.shift_size > 0:
                if not swintransformerblock.fused_window_process:
                    shifted_x = window_reverse(attn_windows, swintransformerblock.window_size, H, W)  # B H' W' C
                    x = torch.roll(shifted_x, shifts=(swintransformerblock.shift_size, swintransformerblock.shift_size), dims=(1, 2))
                else:
                    x = WindowProcessReverse.apply(attn_windows, B, H, W, C, swintransformerblock.shift_size, swintransformerblock.window_size)
            else:
                shifted_x = window_reverse(attn_windows, swintransformerblock.window_size, H, W)  # B H' W' C
                x = shifted_x
            x = x.view(B, H * W, C)
            x = shortcut + swintransformerblock.drop_path(x)
            # FFN
            x = x + swintransformerblock.drop_path(swintransformerblock.mlp(swintransformerblock.norm2(x)))
        if basiclayer.downsample is not None:
            x = basiclayer.downsample(x)



    x = model.norm(x)  # B L C
    x = model.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file',
                        default='configs/swin/swin_tiny_patch4_window7_224.yaml')
    parser.add_argument("--opts",help="Modify config options by adding 'KEY VALUE' pairs. ",
                        default=None,nargs='+',)
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
    # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                         help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    args = parser.parse_args()
    main(args, config)