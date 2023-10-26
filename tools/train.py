import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from model.transformer import VIT
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_color_texture_dataset import MnistDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer):
    r"""
    Method to run the training for one epoch.
    :param epoch_idx: iteration number of current epoch
    :param model: Transformer model
    :param mnist_loader: Data loder for mnist
    :param optimizer: optimizer to be used taken from config
    :return:
    """
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    for data in tqdm(mnist_loader):
        im = data['image'].float().to(device)
        number_cls = data['number_cls'].long().to(device)
        optimizer.zero_grad()
        model_output = model(im)
        loss = criterion(model_output, number_cls)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print('Finished epoch: {} | Number Loss : {:.4f}'.
          format(epoch_idx + 1,
                 np.mean(losses)))
    return np.mean(losses)


def train(args):
    #  Read the config file
    ######################################
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #######################################
    
    # Set the desired seed value
    ######################################
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    #######################################
    
    # Create the model and dataset
    model = VIT(config['model_params']).to(device)
    mnist = MnistDataset('train', config['dataset_params'],
                         im_h=config['model_params']['image_height'],
                         im_w=config['model_params']['image_width'])
    mnist_loader = DataLoader(mnist, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    num_epochs = config['train_params']['epochs']
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    
    # Create output directories
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(config['train_params']['task_name'],
                                   config['train_params']['ckpt_name'])):
        print('Loading checkpoint')
        model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
                                                      config['train_params']['ckpt_name']), map_location=device))
    best_loss = np.inf
    
    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer)
        scheduler.step(mean_loss)
        # Simply update checkpoint if found better version
        if mean_loss < best_loss:
            print('Improved Loss to {:.4f} .... Saving Model'.format(mean_loss))
            torch.save(model.state_dict(), os.path.join(config['train_params']['task_name'],
                                                        config['train_params']['ckpt_name']))
            best_loss = mean_loss
        else:
            print('No Loss Improvement')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vit training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)
