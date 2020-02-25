from __future__ import print_function
import os
import torch
import argparse


class ModelOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Classification of breast cancer histology')
        parser.add_argument('--dataset-path', type=str, default='./dataset_norm',  help='dataset path (default: ./dataset)')
        parser.add_argument('--testset-path', type=str, default='./dataset/test', help='file or directory address to the test set')
        parser.add_argument('--checkpoints-path', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
        parser.add_argument('--num-classes', type=int, default=15, metavar='N', help='Number of classes in dataset')
        parser.add_argument('--patch-stride', type=int, default=256, metavar='N', help='How far the centers of two consecutive patches are in the image (default: 256)')
        parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 30)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
        parser.add_argument('--beta1', type=float, default=0.9, metavar='M', help='Adam beta1 (default: 0.9)')
        parser.add_argument('--beta2', type=float, default=0.999, metavar='M', help='Adam beta2 (default: 0.999)')
        parser.add_argument('--pretrained', type=bool, default=True, help='Resnet pretrained on imagenet?')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--load-chkpt', action='store_true', default=False, help='Resume Training')
        parser.add_argument('--resnet', type=str, default='resnet18_v', help='type of resnet model')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--run', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--gpu-ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        opt.cuda = not opt.no_cuda and torch.cuda.is_available()

        args = vars(opt)
        print('\n------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------\n')

        return opt
