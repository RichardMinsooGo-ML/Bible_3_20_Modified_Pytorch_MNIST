import os
import argparse

import torch
from easydict import EasyDict as edict

def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation of Pyorch YOLOv3')
    
    # parser.add_argument("--data_config" , type=str,   default="config/custom.data", help="path to data config file")

    parser.add_argument("--input_size"   , type=int, default=784, help="input_size 28x28")
    parser.add_argument("--hidden_size"  , type=int, default=200, help="hidden_size")
    parser.add_argument("--output_dim"   , type=int, default=10, help="size of each image dimension")
    parser.add_argument("--num_epochs"   , type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size"   , type=int, default=100, help="size of each image batch")
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning_rate')
    configs = edict(vars(parser.parse_args()))
    
    ############## Dataset, logs, Checkpoints dir ######################
    
    configs.ckpt_dir    = os.path.join(configs.working_dir, 'checkpoints')
    configs.logs_dir    = os.path.join(configs.working_dir, 'logs')

    print(configs)

    if not os.path.isdir(configs.ckpt_dir):
        os.makedirs(configs.ckpt_dir)
    if not os.path.isdir(configs.logs_dir):
        os.makedirs(configs.logs_dir)

    ############## Hardware configurations #############################    
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return configs
