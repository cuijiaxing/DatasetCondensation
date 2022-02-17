import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from PIL import Image
import random


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    model_eval =  'ConvNet'
    args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.


    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    ''' training '''
    criterion = nn.CrossEntropyLoss().to(args.device)

    # load trained synthetic image
    image_syn = torch.from_numpy(np.load('/home/justincui/Documents/original/DatasetCondensation/result/vis_DC_CIFAR10_ConvNet_50ipc_exp0_iter500.npy'))
    # image_syn = torch.from_numpy(np.load('/home/justincui/Documents/original/DatasetCondensation/result/vis_DC_CIFAR10_ConvNet_10ipc_exp0_iter1000.npy'))
    label_syn = torch.from_numpy(np.repeat(np.arange(0, num_classes), args.ipc))
    print('\n==================== adding coreset selection==========\n')
    syn_dataset_model = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
    syn_inital_model = copy.deepcopy(syn_dataset_model)
    syn_dataset_model, acc_train, acc_test = evaluate_synset(8888, syn_dataset_model, image_syn, label_syn, testloader, args)
    print('Evaluate sync dataset random %s, train_acc = %.4f test_acc = %.4f\n-------------------------'%(model_eval, acc_train, acc_test))

    args.epoch_eval_train = 20
    whole_dataset_model = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
    whole_dataset_model, acc_train, acc_test = evaluate_synset(9999, whole_dataset_model, images_all, labels_all, testloader, args)
    print('Evaluate whole dataset random %s, train_acc= %.4f test_acc= %.4f\n-------------------------'%(model_eval, acc_train, acc_test))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
    args.epoch_eval_train = 300

    selected_images = []
    selected_labels = []
    # load training data and see which one syn_model got wrong.
    for i_batch, datum in enumerate(trainloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        whole_model_output = whole_dataset_model(img)
        whole_model_correct = np.equal(np.argmax(whole_model_output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy())
        syn_output = syn_dataset_model(img)
        syn_model_wrong = np.not_equal(np.argmax(syn_output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy())

        img_to_select_index = whole_model_correct & syn_model_wrong
        selected_images.append(img[img_to_select_index == 1].cpu())
        selected_labels.append(lab[img_to_select_index == 1].cpu())

    images_to_select_per_class = { i: [] for i in range(num_classes)}
    for images, labels in zip(selected_images, selected_labels):
        for image, label in zip(images, labels):
            images_to_select_per_class[label.item()].append(image)

    # select num_selection times to get an average accuracy
    num_selection_exp = 5
    for i in range(num_selection_exp):
        syn_images = image_syn.detach().clone()
        syn_labels = label_syn.detach().clone()
        real_images = []
        real_labels = []
        core_set_num = 50
        save_name = os.path.join(args.save_path, 'syn_real_%s_%s_%s_%dipc_%dcoreset__iter%d.png'%(args.method, args.dataset, args.model, args.ipc, core_set_num, i))
        for j in range(num_classes):
            random.shuffle(images_to_select_per_class[j])
            real_images.append(torch.stack(images_to_select_per_class[j][:core_set_num]))
            real_labels.append(torch.ones(core_set_num) * j)
        aaa = syn_images.reshape([num_classes, args.ipc, 3, im_size[0], im_size[1]])
        syn_plus_real_image = torch.cat((aaa, torch.stack(real_images)), dim=1)
        syn_plus_real_image = syn_plus_real_image.reshape([(args.ipc + core_set_num) * num_classes, 3, im_size[0], im_size[1]])
        syn_plus_real_label = torch.cat((syn_labels.reshape([num_classes, args.ipc]), torch.stack(real_labels)), dim=1).reshape([(args.ipc + core_set_num) * num_classes])
        save_image(syn_plus_real_image, save_name, nrow=(args.ipc + core_set_num)) # Trying normalize = True/False may get better visual effects.
        _, acc_train, acc_test = evaluate_synset(7777, copy.deepcopy(syn_inital_model), syn_plus_real_image, syn_plus_real_label, testloader, args)
        print('Evaluate coreset random %s, train_acc = %.4f test_acc = %.4f\n-------------------------'%(model_eval, acc_train, acc_test))



if __name__ == '__main__':
    main()


