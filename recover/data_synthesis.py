'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import argparse
import collections
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from utils import *


def get_images(args, model_teacher, model_teacher_2, model_teacher_3, model_teacher_4, model_teacher_5, hook_for_display, ipc_id, train_ds, class_to_idxs):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size

    best_cost = 1e4

    loss_r_feature_layers = []
    for module in model_teacher_2.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    if model_teacher_3 is not None:
        loss_r_feature_layers_3 = []
        for module in model_teacher_3.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers_3.append(BNFeatureHook(module))   

    if model_teacher_4 is not None:
        loss_r_feature_layers_4 = []
        for module in model_teacher_4.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers_4.append(BNFeatureHook(module))

    if model_teacher_5 is not None:
        loss_r_feature_layers_5 = []
        for module in model_teacher_5.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers_5.append(BNFeatureHook(module))

    

    # setup all class IDs
    all_classes = list(range(len(train_ds.classes)))

    for start in range(0, len(all_classes), batch_size):
        chunk = all_classes[start:start+batch_size]
        # filter out classes without an ipc_id-th image
        valid = [c for c in chunk if len(class_to_idxs.get(c, [])) > ipc_id]
        if not valid:
            continue
        targets = torch.LongTensor(valid).to('cuda')

        # initialize inputs from the ipc_id-th ImageNet example per class
        imgs = []
        for c in valid:
            idx = class_to_idxs[c][ipc_id]
            img, _ = train_ds[idx]                  # [3×224×224], in [0,1]
            imgs.append(img)
        inputs = torch.stack(imgs, dim=0).to('cuda')
        inputs.requires_grad_(True)

        #data_type = torch.float
        #inputs = torch.randn((targets.shape[0], 3, 224, 224), requires_grad=True, device='cuda',
        #                     dtype=data_type)

        #iterations_per_layer = args.iteration
        iterations_per_layer = args.iteration - 500 * (ipc_id % 8)
        
        lim_0, lim_1 = args.jitter, args.jitter

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)
        criterion = nn.CrossEntropyLoss().cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            aug_function = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ])
            inputs_jit = aug_function(inputs)

            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)
            _ = model_teacher_2(inputs_jit)
            if model_teacher_3 is not None:
                _ = model_teacher_3(inputs_jit)
            if model_teacher_4 is not None:
                _ = model_teacher_4(inputs_jit)
            if model_teacher_5 is not None:
                _ = model_teacher_5(inputs_jit)
            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
            loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            if model_teacher_3 is not None:
                rescale_3 = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers_3)-1)]
                loss_r_bn_feature_3 = sum([mod.r_feature * rescale_3[idx] for (idx, mod) in enumerate(loss_r_feature_layers_3)])

            if model_teacher_4 is not None:
                rescale_4 = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers_4)-1)]
                loss_r_bn_feature_4 = sum([mod.r_feature * rescale_4[idx] for (idx, mod) in enumerate(loss_r_feature_layers_4)])

            if model_teacher_5 is not None:
                rescale_5 = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers_5)-1)]
                loss_r_bn_feature_5 = sum([mod.r_feature * rescale_5[idx] for (idx, mod) in enumerate(loss_r_feature_layers_5)])

            # R_prior losses
            _, loss_var_l2 = get_image_prior_losses(inputs_jit)

            # l2 loss on images (reshape to actual batch size)
            n = inputs_jit.size(0)
            loss_l2 = inputs_jit.view(n, -1).norm(2,1).mean()

            # combining losses
            loss_aux = args.r_bn * loss_r_bn_feature
            count_aux = 1
            if model_teacher_3 is not None:
                loss_aux += args.r_bn * loss_r_bn_feature_3
                count_aux += 1
            if model_teacher_4 is not None:
                loss_aux += args.r_bn * loss_r_bn_feature_4
                count_aux += 1
            if model_teacher_5 is not None:
                loss_aux += args.r_bn * loss_r_bn_feature_5
                count_aux += 1

            loss = loss_ce + loss_aux / count_aux

            if iteration % save_every==0:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("main criterion", criterion(outputs, targets).item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    hook_for_display(inputs, targets)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip(inputs.data)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone()
            best_inputs = denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
    torch.cuda.empty_cache()

def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path +'/class{:03d}_id{:03d}.jpg'.format(class_id,ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def parse_args():
    parser = argparse.ArgumentParser(
        "SRe2L: recover data from pre-trained model")
    """Data save flags"""
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    parser.add_argument('--train-data-path', type=str, required=True,
                        help="root of ImageNet train folder")
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=32, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.05,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    parser.add_argument('--tv-l2', type=float, default=0.0001,
                        help='coefficient for total variation L2 loss')
    parser.add_argument('--l2-scale', type=float,
                        default=0.00001, help='l2 loss on the image')
    """Model related flags"""
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")
    """IPC (Image Per Class) flags"""
    parser.add_argument("--ipc-start", default=0, type=int, help="start index of IPC")
    parser.add_argument("--ipc-end", default=50, type=int, help="end index of IPC")
    args = parser.parse_args()

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    return args

def replace_conv_with_identity(model, drop_prob=0.3):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and random.random() < drop_prob:
            setattr(model, name, nn.Identity())
        else:
            replace_conv_with_identity(module, drop_prob)

def main_syn(ipc_id):
    args = parse_args()

    # Define smaller torchvision models for rotation
    small_models = ["resnet18", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x0_5", "alexnet"]
    #[
    #    'resnet18',
    #    'resnet34',
    #    'shufflenet_v2_x1_0',
    #    'mnasnet1_0',
    #    'efficientnet_b0',
    #]

    # Select model based on IPC ID, rotating through the small_models list
    selected_model_name = small_models[0]

    print(f'Using model: {selected_model_name} for ipc_id: {ipc_id}')

    # Load the selected pretrained teacher model
    model_teacher = models.__dict__[selected_model_name](pretrained=True)


    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    # Select model based on IPC ID, rotating through the small_models list
    # Instead of:
    selected_model_name_2 = selected_model_name
    prob_include = 0.5

    # with probability 0.5, use the same model as model_teacher_2
    if random.random() < prob_include:
        selected_model_name_3 = random.choice(small_models)
        while selected_model_name_2 == selected_model_name_3:
            selected_model_name_3 = random.choice(small_models)
    else:
        selected_model_name_3 = None
    
    if random.random() < prob_include:
        selected_model_name_4 = random.choice(small_models)
        while selected_model_name_2 == selected_model_name_4 or selected_model_name_3 == selected_model_name_4:
            selected_model_name_4 = random.choice(small_models)
    else:
        selected_model_name_4 = None

    if random.random() < prob_include:
        selected_model_name_5 = random.choice(small_models)
        while selected_model_name_2 == selected_model_name_5 or selected_model_name_3 == selected_model_name_5 or selected_model_name_4 == selected_model_name_5:
            selected_model_name_5 = random.choice(small_models)
    else:
        selected_model_name_5 = None
    

    print(f'Using model: {selected_model_name_2} for ipc_id: {ipc_id}')
    print(f'Using model: {selected_model_name_3} for ipc_id: {ipc_id}')

    # Load the selected pretrained teacher model
    model_teacher_2 = models.__dict__[selected_model_name_2](pretrained=True)
    model_teacher_2 = nn.DataParallel(model_teacher_2).cuda()
    model_teacher_2.eval()
    for p in model_teacher_2.parameters():
        p.requires_grad = False

    if selected_model_name_3 is not None:
        model_teacher_3 = models.__dict__[selected_model_name_3](pretrained=True)
        model_teacher_3 = nn.DataParallel(model_teacher_3).cuda()
        model_teacher_3.eval()
        for p in model_teacher_3.parameters():
            p.requires_grad = False
    else:
        model_teacher_3 = None

    if selected_model_name_4 is not None:
        model_teacher_4 = models.__dict__[selected_model_name_4](pretrained=True)
        model_teacher_4 = nn.DataParallel(model_teacher_4).cuda()
        model_teacher_4.eval()
        for p in model_teacher_4.parameters():
            p.requires_grad = False
    else:
        model_teacher_4 = None

    if selected_model_name_5 is not None:

        model_teacher_5 = models.__dict__[selected_model_name_5](pretrained=True)
        model_teacher_5 = nn.DataParallel(model_teacher_5).cuda()
        model_teacher_5.eval()
        for p in model_teacher_5.parameters():
            p.requires_grad = False
    else:
        model_teacher_5 = None

    model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
    model_verifier = model_verifier.cuda()
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Create dataset and build class_to_idxs lookup
    train_dataset = datasets.ImageFolder(
        args.train_data_path,
        transform=T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize
        ])
    )
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    
    # build class_to_idxs lookup once
    class_to_idxs = {}
    for idx, (_, lbl) in enumerate(train_dataset.samples):
        class_to_idxs.setdefault(lbl, []).append(idx)

    hook_for_display = lambda x,y: validate(x, y, model_verifier)
    get_images(args, model_teacher, model_teacher_2, model_teacher_3, model_teacher_4, model_teacher_5, hook_for_display, ipc_id, train_dataset, class_to_idxs)


if __name__ == '__main__':
    args = parse_args()
    for ipc_id in range(args.ipc_start, args.ipc_end):
        print('ipc = ', ipc_id)
        main_syn(ipc_id)
