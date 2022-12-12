'''
A. Data Engineering
'''

'''
1. Import Libraries for Data Engineering
'''


import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

import tqdm
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
# import torchvision.transforms as transforms
from torchvision import transforms

from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim

import numpy as np

from models.models import *
from config.train_config import parse_train_configs

def main():
    
    '''
    4. Set Hyperparameters
    '''
    # input_size = 784 # 28x28
    # hidden_size = 200
    # output_dim = 10  # output layer dimensionality
    # Get data configuration
    configs = parse_train_configs()
    
    print(configs.device)
    
    '''
    2. Load MNIST data
    '''
    root = os.path.join('~', '.torch', 'mnist')
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda x: x.view(-1)])
    train_dataset = datasets.MNIST(root=root,
                                 download=True,
                                 train=True,
                                 transform=transform)
    test_dataset = datasets.MNIST(root=root,
                                download=True,
                                train=False,
                                transform=transform)

    '''
    B. Model Engineering
    '''

    '''
    3. Import Libraries for Model Engineering
    '''
    
    np.random.seed(123)
    torch.manual_seed(123)
    
    '''
    6. Build NN model
    '''

    model = Feed_Forward_Net(configs.input_size, configs.hidden_size, configs.output_dim)
    # model.apply(weights_init_normal)
    # model.print_network()
    model = model.to(configs.device)
    
    '''
    6. Load Pre-trained data
    # Skip for this implementation
    '''    
    '''
    7. Optimizer
    '''
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
    
    '''
    X. Metrics
    # Skip for this implementation
    '''
    
    '''
    7. learning rate scheduler
    # Skip for this implementation
    '''
    
    '''
    
    5. DataLoader
    '''
    train_dataloader = DataLoader(dataset=train_dataset,
        batch_size=configs.batch_size, 
        shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
        batch_size=configs.batch_size, 
        shuffle=False)
    
    '''
    8. Define Loss Fumction
    '''
    criterion = nn.CrossEntropyLoss()
    def compute_loss(t, y):
        return criterion(y, t)
    
    '''
    11. Define Episode / each step process
    '''

    start_time = time.time() 
    for epoch in range(0, configs.num_epochs, 1):
        
        num_iters_per_epoch = len(train_dataloader)

        print(num_iters_per_epoch)

        # switch to train mode
        model.train()
        
        epoch_loss = 0
        train_acc = 0.
        
        '''
        9. Define train loop
        '''
        for (x, t) in train_dataloader:
            '''
            Exlore Train_dataloader
            Not coded in this implementation.
            '''
            x, t = x.to(configs.device), t.to(configs.device)

            preds = model(x)
            total_loss = compute_loss(t, preds)

            total_loss.backward()
            if global_step % configs.gradient_accumulations:
                
                optimizer.step()

                optimizer.zero_grad()
                
            epoch_loss += total_loss.item()
            train_acc += \
                accuracy_score(t.tolist(),
                preds.argmax(dim=-1).tolist())

        epoch_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        print('epoch: {}, loss: {:.3}, acc: {:.3f}'.format(
            epoch+1,
            epoch_loss,
            train_acc
        ))
    
    '''
    12. Model evaluation
    '''

    test_loss = 0.
    test_acc = 0.

    for (x, t) in test_dataloader:
        x, t = x.to(configs.device), t.to(configs.device)
        # loss, preds = test_step(x, t)
        '''
        10. Define validation / test loop
        '''
        model.eval()
        preds = model(x)
        loss = compute_loss(t, preds)
        
        
        test_loss += loss.item()
        test_acc += \
            accuracy_score(t.tolist(),
                           preds.argmax(dim=-1).tolist())

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        test_loss,
        test_acc
    ))
            
if __name__ == '__main__':
    main()
    
