#
# demo.py
#
import argparse
import os
import numpy as np
import win32gui
from ctypes import windll

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from PIL import ImageGrab
import cv2
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid, save_image
import torch.cuda

print(torch.cuda.is_available())


user32 = windll.user32
user32.SetProcessDPIAware()

####################################################
hwnd = win32gui.FindWindow(None, 'Google Earth Pro')


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str,  help='image to test')
    parser.add_argument('--out-path', type=str,  help='mask image to save')
    parser.add_argument('--norm', type=str, default='gn', help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='resnet101.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    model = DeepLab(args, num_classes=3,

                    freeze_bn=args.freeze_bn)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    while True:
        dimensions = win32gui.GetWindowRect(hwnd)
        img = ImageGrab.grab(dimensions)
        img = img.resize((800, 600))

        device = "cuda"

        if "cuda" in device and not torch.cuda.is_available():
            device = "cpu"

        model.to(device)

        image = img.convert('RGB')
        target = img.convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        tensor_in = tensor_in.to(device)


        model.eval()
        #args.cuda = True
        if args.cuda:
            image = image.cuda()
        with torch.no_grad():
            output = model(tensor_in)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                               3, normalize=False, range=(0, 255))
        print("type(grid) is: ", type(grid_image))
        print("grid_image.shape is: ", grid_image.shape)
        #save_image(grid_image, args.out_path)
        im1 = np.copy(img)
        im2 = grid_image.permute(1, 2, 0).numpy()
        #image = cv2.addWeighted(im1, 0.5, im2, 0.5, 0)
        cv2.imshow("Result", im2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
