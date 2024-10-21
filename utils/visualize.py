import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

import warnings
import seaborn as sns

import umap
import umap.plot

from utils.adv_freq import FFT_Batch

warnings.filterwarnings("ignore")


def CollectFeature(model, module_name='avgpool', save_path=None, loader=None):
    # model_trunc = create_feature_extractor(model, return_nodes={module_name: 'semantic_feature'})
    feat_list = []
    label_list = []
    with torch.no_grad():
        for i, (img, label) in tqdm(enumerate(loader)):
            assert img.shape[0] == 1
            img = img.cuda()
            # pred_logits = model_trunc(img)
            # sem_feat = pred_logits['semantic_feature'].squeeze().detach().cpu().numpy()
            _, sem_feat = model.get_logits_feat(img)
            sem_feat = sem_feat.squeeze().detach().cpu().numpy()
            feat_list.append(sem_feat)
            label_list.append(label.numpy())

    feat_array = np.array(feat_list)
    label_array = np.array(label_list)
    if save_path:
        np.save(save_path + '_feat.npy', feat_array)
        np.save(save_path + '_label.npy', label_array)
    return feat_array, label_array


def LoadFeature(f_path):
    return np.load(f_path, allow_pickle=True)


def GetColor(color_num=7):
    palette = sns.hls_palette(color_num)
    return palette


def UMAP_Points(feat_array, class_num=7):
    mapper = umap.UMAP(n_neighbors=class_num, random_state=1).fit(feat_array)
    print("UMAP done!")
    umap_points = mapper.embedding_[:]
    return umap_points


def PlotPoints(points_array, label_array, class_num=7, s=70, legend=True):
    plt.figure(figsize=(10, 10))
    color = GetColor(class_num)
    for i in range(class_num):
        index = np.where(label_array == i)
        plt.scatter(points_array[index, 0], points_array[index, 1], color=color[i], s=s, label='Class ' + str(i + 1))

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)

    if legend:
        plt.legend()

    plt.show()


def PlotUMAP(model, loader, class_num=7):
    feat_array, label_array = CollectFeature(model, loader=loader)
    points_array = UMAP_Points(feat_array, class_num)
    PlotPoints(points_array, label_array, class_num)


def showImg(img_tensor, index=0):
    plt.imshow(np.transpose(img_tensor[index].cpu().detach().numpy(), (1, 2, 0)), cmap=plt.cm.rainbow_r)
    plt.show()


def AmpSensitivityMap_FFDNet(model, loader, save_fig=None, show_fig=True):
    grad_list = []
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    CE_LOSS = nn.CrossEntropyLoss()

    for i, (img, label) in tqdm(enumerate(loader)):
        img = img.cuda()
        label = label.cuda()

        amp, phase = FFT_Batch(img)
        amp.requires_grad = True
        recon_img = amp * torch.exp(1j * phase)
        recon_img = torch.fft.ifftshift(recon_img)
        recon_img = torch.abs(torch.fft.ifft2(recon_img))
        recon_img = recon_img / torch.max(recon_img)

        recon_img = normalize(recon_img)

        output, _, = model.get_logits_feat(recon_img)
        # output = model(recon_img)

        loss = CE_LOSS(output, label)
        # grad = torch.autograd.grad(loss, amp, retain_graph=False, create_graph=False)[0]
        loss.backward()
        grad = amp.grad.data
        grad = grad.mean(dim=1).abs().sum(dim=0)  # .cpu().numpy()
        # print(grad.shape)

        grad_list.append(grad)

    glt = torch.stack(grad_list)
    sum_grad = torch.sum(glt, dim=0)

    if show_fig:
        plt.imshow(sum_grad.cpu().detach().numpy(), cmap=plt.cm.rainbow)
        plt.colorbar()
        if save_fig is not None:
            plt.savefig(save_fig, dpi=80)

    return sum_grad


def GetImageGradAmp_FFDNet(model, loader):
    grad_list = []
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    CE_LOSS = nn.CrossEntropyLoss()

    for i, (img, label) in tqdm(enumerate(loader)):
        img = img.cuda()
        label = label.cuda()

        img.requires_grad = True
        img = normalize(img)

        output, _, = model.get_logits_feat(img)
        # output = model(recon_img)

        loss = CE_LOSS(output, label)
        # grad = torch.autograd.grad(loss, amp, retain_graph=False, create_graph=False)[0]
        grad = torch.autograd.grad(loss, img, )[0]

        grad_amp, _ = FFT_Batch(grad)
        grad_amp = torch.abs(grad_amp).mean(dim=1)
        # print(grad.shape)

        grad_list.append(grad_amp)

    glt = torch.cat(grad_list, dim=0)
    sum_grad = torch.mean(glt, dim=0)

    return sum_grad


def FetchBatch(loader):
    for i, (img, label) in enumerate(loader):
        img = img.cuda()
        label = label.cuda()
        break
    return img, label


def saveTensor2Img(timg, save_path):
    trans = transforms.ToPILImage()
    pilimg = trans(timg)
    pilimg.save(save_path)


def GetContributions(model, loader, save_path=None):
    sum_grad = AmpSensitivityMap_FFDNet(model, loader, show_fig=False, save_fig=save_path)
    norm_grad = sum_grad / torch.norm(sum_grad, p=1, dim=(0, 1), keepdim=True)

    max_g = torch.max(norm_grad)
    min_g = torch.min(norm_grad)
    s_g = (norm_grad - min_g) / (max_g - min_g)

    low_contributions = torch.sum(s_g[92:132, 92:132]) / torch.sum(s_g)
    high_contributions = 1 - low_contributions

    low_contributions = 10000*low_contributions / (40 * 40)
    high_contributions = 10000*high_contributions / (224 * 224 - 40*40)

    low_contributions = low_contributions.cpu().detach().numpy()
    high_contributions = high_contributions.cpu().detach().numpy()
    lfc = low_contributions / high_contributions
    return low_contributions, high_contributions, lfc

