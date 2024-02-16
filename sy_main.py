import torch
from models import build_model
import argparse
from config import get_config
from data.build import build_transform
import os
from torchvision import datasets, transforms
from data.cached_image_folder import CachedImageFolder
from data.imagenet22k_dataset import IN22KDATASET

def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'

        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            config.DATA.DATA_PATH = r"/home/dreamyou070/MyData/anomaly_detection/MVTec3D-AD/carrot"
            root = os.path.join(config.DATA.DATA_PATH,
                                prefix)
            dataset = datasets.ImageFolder(root,
                                           transform=transform)

        nb_classes = 1000

    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    else:
        raise NotImplementedError("We only support ImageNet Now.")
    return dataset, nb_classes

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


            print(f'swintransformerblock.shift_size : {swintransformerblock.shift_size}')
            print(f'swintransformerblock.window_size : {swintransformerblock.window_size}')
            print(f'input to the swinfertransformerblock : {x.shape}')
            x = swintransformerblock(x)
            print(f'output of the swintransformerblock : {x.shape}')

        if basiclayer.downsample is not None:
            x = basiclayer.downsample(x)
    x = model.norm(x)  # B L C
    x = model.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)
    """
        
        
        B, L, C = x.shape # 1, 56*56, 96
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = swintransformerblock.norm1(x)
        x = x.view(B, H, W, C) # 1, 56, 56, 96
        shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
    """





    #    target = torch.randn(1,4,64,64)
        # compute output
    #    with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
     #       output = model(images)


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