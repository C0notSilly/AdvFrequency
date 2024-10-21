import torch

from utils.load_data import LoadDataset_SingleSource
import torchvision
from torchvision import transforms

import argparse
import random
import os
import numpy as np
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F

from models.AmpAug import AmpDrop, AmpUCNoise

from utils.adv_freq import FreqMask, FFT_Batch, IFFT_Batch
from utils.model_test import Tester
from utils.similarity import JS_Divergence
from models.resnet import resnet18


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed = 1024


def _init_fn(worker_id):
    np.random.seed(int(seed) + worker_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="PACS")  # PACS, Digits
    parser.add_argument("--data_root", default="./data/PACS")
    parser.add_argument("--source", default="sketch")
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--epochs", default=60, type=int)

    parser.add_argument("--optim_name", default="Adam")  # Adam, SGD, Adam is better
    parser.add_argument("--lr", default=1e-4)
    # parser.add_argument("--lr_decay", default=100)
    # parser.add_argument("--lr_decay_ratio", default=0.1)

    parser.add_argument("--class_num", default=7)

    parser.add_argument("--mode", default='train_augmix')  # train, test, train_corruption
    parser.add_argument("--save_path", default='./Result/B_P')
    parser.add_argument("--weight_path", default='./pretrained/resnet18_pretrained.pth')


    args = parser.parse_args()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    device = torch.device("cuda:2")

    train_loader, val_loader = LoadDataset_SingleSource(args.dataset_name, args.source, args.data_root, args.batch_size,
                                                        args.mode, train_transform)

    test_loader_dict = LoadDataset_SingleSource(args.dataset_name, args.source, args.data_root,
                                                args.batch_size, 'test', test_transform)

    model = resnet18()

    model.load_state_dict(torch.load(args.weight_path), strict=False)
    model.fc = nn.Linear(model.fc.in_features, args.class_num)
    model.to(device)

    ampDrop = AmpDrop(model)
    ampUC = AmpUCNoise(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    CE_LOSS = nn.CrossEntropyLoss()
    KL_DIV_LOSS = nn.KLDivLoss()
    MSE_LOSS = nn.MSELoss()

    tester = Tester(model, test_loader_dict, device)

    # prior = torch.load(args.prior_path)

    best_acc = 0
    best_ave_acc = 0

    for epoch in range(args.epochs):

        model.train()

        train_loader_tbar = tqdm(train_loader)
        postfix = {'Train Loss': 0, 'Epoch': epoch + 1, 'Iter': 0,
                   'Lr': optimizer.state_dict()['param_groups'][0]['lr'],
                   'cls loss': 0}
        train_loader_tbar.set_postfix(postfix)

        # class_centroids = calculate_class_centroids_FFDNet_AugMix(model, train_loader)
        gradlist = []
 
        for i, (img_tuple, label) in enumerate(train_loader_tbar):
            img = img_tuple[0]
            # am_img = img_tuple[1]

            img = img.cuda(device)
            # am_img = am_img.cuda(device)
            label = label.cuda(device)

            # low_Freq, high_Freq = FreqMask(img, 30, 30)
            # aug_img = phaseAug(img, label)
            # ad_img, _ = ampDrop(img, label)
            # ad_img = ampUC(img, label)
            au_img = ampUC(img, label)
            # au_img, _ = ampDrop(img, label)
            au_img = au_img.detach()
            amp, phase = FFT_Batch(au_img)
            amp.requires_grad = True
            au_img = IFFT_Batch(amp, phase)
            # img = normalize(img)
            # ad_img = normalize(ad_img)
            # am_img = normalize(am_img)
            au_img = normalize(au_img)

            # output, sem_feat = model.get_logits_feat(img)
            # ad_output, ad_sem_feat = model.get_logits_feat(ad_img)
            au_output, au_sem_feat = model.get_logits_feat(au_img)
            # am_output, am_sem_feat = model.get_logits_feat(am_img)

            # cls_loss = CE_LOSS(output, label) + CE_LOSS(au_output, label)# + CE_LOSS(am_output, label) + CE_LOSS(ad_output, label)
            cls_loss = CE_LOSS(au_output, label)
            #con_loss = JS_Divergence(output, ad_output, au_output, am_output)
            # centroid_loss = MSE_LOSS(sem_feat, class_centroids[label]) + MSE_LOSS(ad_sem_feat,class_centroids[label]) + MSE_LOSS(am_sem_feat, class_centroids[label]) + MSE_LOSS(au_sem_feat, class_centroids[label])
            loss = cls_loss #+ con_loss#  + centroid_loss
            grad = torch.autograd.grad(loss, amp, retain_graph=True)[0]
            gradlist.append(grad.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            postfix['Train Loss'] = loss.item()
            postfix['Iter'] = i + 1
            postfix['Lr'] = optimizer.state_dict()['param_groups'][0]['lr']
            postfix['cls loss'] = cls_loss.item()
            # postfix['centro loss'] = centroid_loss.item()
            # postfix['grad loss'] = gl_loss.item()
            train_loader_tbar.set_postfix(postfix)

            '''
            if i % args.print_interval == 0:
                print('Epoch [', epoch + 1, '/', args.epochs, '][', i + 1, '/', len(train_loader), '] : ',
                      loss.item())
            '''
        np.save('./aaua_grad.npy', gradlist)
        
        '''
        scheduler.step()
        model.eval()
        correct = torch.zeros(1).squeeze().cuda(device)
        total = torch.zeros(1).squeeze().cuda(device)

        with torch.no_grad():
            for i, (img, label) in enumerate(val_loader):
                img = img.cuda(device)
                label = label.cuda(device)

                img = normalize(img)

                output, _ = model.get_logits_feat(img)
                prediction = torch.argmax(output, 1)
                correct += (prediction == label).sum().float()
                total += len(label)

        acc = (correct / total).cpu().detach().data.numpy()
        print('Epoch: ', epoch + 1, ' test accuracy: ', acc, '/', best_acc)
        if acc.item() > best_acc:
            print('New Best Accuracy.')
            best_acc = acc.item()
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_val.pth'))


        if epoch % 1 == 0:
            ave_acc = tester.test()
            if ave_acc > best_ave_acc:
                best_ave_acc = ave_acc
                torch.save(model.state_dict(), os.path.join(args.save_path, 'best_gen.pth'))
        '''
