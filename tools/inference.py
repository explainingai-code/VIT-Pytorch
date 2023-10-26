import yaml
import argparse
import torch
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from model.transformer import VIT
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_color_texture_dataset import MnistDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_accuracy(model, mnist_loader):
    r"""
    Method to get accuracy for number classification for trained model
    :param model:
    :param mnist_loader:
    :return:
    """
    num_total = 0.
    num_correct = 0.
    
    for data in tqdm(mnist_loader):
        im = data['image'].float().to(device)
        number_cls = data['number_cls'].long().to(device)
        model_output = model(im)
        pred_num_cls_idx = torch.argmax(model_output, dim=-1)
        num_total += pred_num_cls_idx.size(0)
        num_correct += torch.sum(pred_num_cls_idx == number_cls).item()
    num_accuracy = num_correct / num_total
    print('Number Accuracy : {:2f}'.format(num_accuracy))

   
def visualize_pos_embed(model):
    r"""
    Method to save the positional embeddings cosine similarity map
    Assumes number of patches to be 196
    :param model:
    :return:
    """
    # pos_embed = 1 x Num_patches+1 x D
    # Get indexes after CLS
    pos_emb = model.patch_embed_layer.pos_embed.detach().cpu()[0][1:]

    plt.tight_layout(pad=0.1, rect=(0.1, 0.1, 0.9, 0.9))
    fig, axs = plt.subplots(7, 7)
    count = 0
    for i in tqdm(range(196)):
        row = i // 14
        col = i % 14
        if row % 2 == 0 and col % 2 == 0:
            out = torch.cosine_similarity(pos_emb[i], pos_emb, dim=-1)
            fig.add_subplot(7, 7, count+1)
            plt.xticks([])
            plt.yticks([])
            count += 1
            plt.subplots_adjust(0.1, 0.1, 0.9, 0.9)
            plt.imshow(out.reshape(14, 14), vmin=-1, vmax=1)
    for idx, ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    if not os.path.exists('output'):
        os.mkdir('output')
    plt.savefig('output/position_plot.png', bbox_inches='tight')
    
    
def visualize_attn_weights(mnist, model):
    r"""
    Uses trivial implementation of rollout.
    :param mnist:
    :param model:
    :return:
    """
    num_images = 10
    idxs = torch.randint(0, len(mnist) - 1, (num_images,))
    ims = torch.cat([mnist[idx]['image'][None, :] for idx in idxs]).float()
    ims = ims.to(device)
    attentions = []
    
    def get_attention(model, input, output):
        attentions.append(output.detach().cpu())
        
    # Add forward hook
    for name, module in model.named_modules():
        if 'attn_dropout' in name:
            module.register_forward_hook(get_attention)
    
    model(ims)
    
    # Handle residuals
    attentions = [(torch.eye(att.size(-1)) + att)/(torch.eye(att.size(-1)) + att).sum(dim=-1).unsqueeze(-1) for att in attentions]
    
    result = torch.max(attentions[0], dim=1)[0]
    # Max or mean both are fine
    for i in range(1, 6):
        att = torch.max(attentions[i], dim=1)[0]
        result = torch.matmul(att, result)

    masks = result
    masks = masks[:, 0, 1:]
    for i in range(num_images):
        im_input = torch.permute(ims[i].detach().cpu(), (1, 2, 0)).numpy()
        im_input = im_input[:, :, [2, 1, 0]]
        im_input = (im_input+1)/2 * 255
        mask = masks[i].reshape((14, 14)).numpy()
        
        mask = mask/np.max(mask)
        
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)[..., None]
        if not os.path.exists('output'):
            os.mkdir('output')
        cv2.imwrite('output/input_{}.png'.format(i), im_input)
        cv2.imwrite('output/overlay_{}.png'.format(i), im_input*mask)


def inference(args):
    # Read the config file
    ######################################
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #######################################
    
    # Create the model and dataset
    model = VIT(config['model_params']).to(device)
    model.eval()
    mnist = MnistDataset('test', config['dataset_params'],
                         im_h=config['model_params']['image_height'],
                         im_w=config['model_params']['image_width'])
    mnist_loader = DataLoader(mnist, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(config['train_params']['task_name'],
                                   config['train_params']['ckpt_name'])):
        print('Loading checkpoint')
        model.load_state_dict(torch.load(os.path.join(config['train_params']['task_name'],
                                                      config['train_params']['ckpt_name']), map_location=device))
    else:
        print('No checkpoint found at {}'.format(os.path.join(config['train_params']['task_name'],
                                   config['train_params']['ckpt_name'])))
    with torch.no_grad():
        # Run inference and measure accuracy on number
        get_accuracy(model, mnist_loader)
        # Visualize positional embedding
        visualize_pos_embed(model)
        # Visualize attention weights
        visualize_attn_weights(mnist, model)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vit training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    inference(args)