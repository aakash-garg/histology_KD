from __future__ import print_function
import os
import torch
import argparse
import logging

#1 - pretrained true
#2 - pretrained false, no aug
#3 - pretrained false, with aug
#4 - pretrained false, no test - 0.8 & 0.2
#5 - resnet50 pretrained true, batch=8, def 64 gpu - 0,1
#6 - trained using kd


# no pretraining batch 16
# with KD approach
class ModelOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Classification of breast cancer histology')
        parser.add_argument('--dataset-path', type=str, default='./dataset_norm',  help='dataset path (default: ./dataset)')
        parser.add_argument('--checkpoints-path', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 64)')
        parser.add_argument('--num-classes', type=int, default=4, metavar='N', help='Number of classes in dataset')
        parser.add_argument('--patch-stride', type=int, default=256, metavar='N', help='How far the centers of two consecutive patches are in the image (default: 256)')
        parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 30)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
        parser.add_argument('--beta1', type=float, default=0.9, metavar='M', help='Adam beta1 (default: 0.9)')
        parser.add_argument('--beta2', type=float, default=0.999, metavar='M', help='Adam beta2 (default: 0.999)')
        parser.add_argument('--pretrained', type=bool, default=False, help='Resnet pretrained on imagenet?')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--attention_transfer', action='store_true', default=False, help='Train via Attention Transfer?')
        parser.add_argument('--load-chkpt', action='store_true', default=False, help='Resume Training')
        parser.add_argument('--student', type=str, default='resnet8_v', help='type of student resnet model')
        parser.add_argument('--teacher', type=str, default='resnet50_v', help='type of teacher resnet model')
        parser.add_argument('--teacher_path', type=str, default='weights_resnet50_v_1_5', help='path for teacher model')
        parser.add_argument('--distill', type=bool, default=False, help='Apply Distillation?')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--run', type=int, default=7, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')
        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        opt.cuda = not opt.no_cuda and torch.cuda.is_available()
        args = vars(opt)

        logger_name = "Settings"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        checkpoint_folder = os.path.join(opt.checkpoints_path, 'weights_' + opt.student + "_" + str(opt.seed) + "_" + str(opt.run))
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        filename = os.path.join(checkpoint_folder, 'settings.log')
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)


        logger.info('\n------------ Options -------------')
        for k, v in sorted(args.items()):
            logger.info('%s: %s' % (str(k), str(v)))
        logger.info('-------------- End ----------------\n')

        return opt
