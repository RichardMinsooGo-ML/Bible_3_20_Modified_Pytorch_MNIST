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
from torchvision import transforms
import torch.optim as optim
import torchvision
from torchvision import datasets
from sklearn.metrics import accuracy_score

from models.models import *
from config.train_config import parse_train_configs

def main():
    
    '''
    4. Set Hyperparameters
    '''
    # Get data configuration
    configs = parse_train_configs()
    
    print(configs.device)
    
    '''
    2. Data Engineering
    = Load MNIST data
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
    6. Build NN model
    '''
    
    model = Feed_Forward_Net(configs.input_size, configs.hidden_size, configs.output_dim)
    # model.apply(weights_init_normal)
    # model.print_network()
    model = model.to(configs.device)
    
    '''
    6. Load Pre-trained data
    '''    
    print(configs.pretrained_path)
    
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
    '''
    # Not Applied in this IMPL
    
    '''
    
    5. DataLoader
    '''
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size, 
        shuffle=True,
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=configs.batch_size, 
        shuffle=False,
    )
    
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
        global_step = 0
        
        '''
        9. Define train loop
        '''
        for (x, t) in train_dataloader:
            """
            Exlore Train_dataloader
            Not coded in this implementation.
            
            End of exploration.
            """
            x, t = x.to(configs.device), t.to(configs.device)
            
            global_step += 1

            preds = model(x)
            total_loss = compute_loss(t, preds)
            
            epoch_loss += float(total_loss.item())
            '''
            '''            
            # compute gradient and perform backpropagation
            total_loss.backward()

            if global_step % configs.gradient_accumulations:
                '''
                '''
                # Accumulates gradient before each step
                optimizer.step()

                '''
                '''
                # zero the parameter gradients
                optimizer.zero_grad()
            
            train_acc += \
                accuracy_score(t.tolist(),
                preds.argmax(dim=-1).tolist())

            """
            Metric Table and Log progress 
            """
                
            """
            End of Metric Table and Log progress 
            """
            
        crnt_epoch_loss = epoch_loss/num_iters_per_epoch
        train_acc /= num_iters_per_epoch
        torch.save(model.state_dict(), configs.save_path)
        # global_epoch += 1
        
        # print("Global_epoch :",global_epoch, "Current epoch loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(configs.save_path))
        print("Current epoch loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(configs.save_path))
        
        print('epoch: {}, loss: {:.3}, acc: {:.3f}'.format( epoch+1, crnt_epoch_loss, train_acc ))
    
    '''
    Test Loop
    '''
    num_iters_per_epoch = len(test_dataloader)

    print(num_iters_per_epoch)
    # switch to evaluation mode
    model.eval()

    test_loss = 0.
    test_acc = 0.

    for (x, t) in test_dataloader:
        x, t = x.to(configs.device), t.to(configs.device)
        # loss, preds = test_step(x, t)
        '''
        10. Define validation / test loop
        '''
        preds = model(x)
        total_loss = compute_loss(t, preds)

        test_loss += float(total_loss.item())
        test_acc += \
            accuracy_score(t.tolist(),
                           preds.argmax(dim=-1).tolist())

    test_loss /= num_iters_per_epoch
    test_acc /= num_iters_per_epoch

    print("Current epoch loss : {:1.5f}".format(test_loss),'Saved at {}'.format(configs.save_path))
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(test_loss, test_acc ))
    
    #-------------------------------------------------------------------------------------
    """
    # Save checkpoint
    if (epoch+1) % configs.checkpoint_freq == 0:
        torch.save(model.state_dict(), configs.save_path)
        print('save a checkpoint at {}'.format(configs.save_path))
    """

if __name__ == '__main__':
    main()
    
