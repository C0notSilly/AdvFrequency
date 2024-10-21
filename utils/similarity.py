import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from torchvision.models.feature_extraction import create_feature_extractor


def cosine_similarity(feat1, feat2):
    b = feat1.shape[0]
    return F.cosine_similarity(feat1.view(b, -1), feat2.view(b, -1))


def geodesic_distance(feat1, feat2):
    cos_simi = cosine_similarity(feat1, feat2)
    g_d = torch.arccos(cos_simi) / torch.pi
    return g_d


def calculate_centroids(model, loader, module_name='avgpool', class_num=7):
    feat_list = []
    label_list = []

    model_trunc = create_feature_extractor(model, return_nodes={module_name: 'semantic_feature'})

    with torch.no_grad():
        for i, (img, label) in tqdm(enumerate(loader)):
            img = img.cuda()
            pred_logits = model_trunc(img)
            sem_feat = pred_logits['semantic_feature']
            feat_list.append(sem_feat)
            label_list.append(label)

    feat_tensor = torch.cat(feat_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)

    centroids = []

    for i in range(class_num):
        index = torch.where(label_tensor == i)
        class_tensor = feat_tensor[index]
        center = torch.mean(class_tensor, dim=0)
        centroids.append(center)

    centroids = torch.stack(centroids, dim=0)
    return centroids


def Geodesic_Distance_Batch(feat, label, centroids):
    selected_centroids = centroids[label]
    g_distance = geodesic_distance(feat, selected_centroids)
    return g_distance


def calculate_class_centroids(model, loader, class_num=7):
    feat_list = []
    label_list = []

    with torch.no_grad():
        for i, (img, label) in tqdm(enumerate(loader)):
            img = img.cuda()
            label = label.cuda()
            _, sem_feat = model.get_logits_feat(img)
            feat_list.append(sem_feat)
            label_list.append(label)

    feat_tensor = torch.cat(feat_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)

    centroids = []

    for i in range(class_num):
        index = torch.where(label_tensor == i)
        class_tensor = feat_tensor[index]
        center = torch.mean(class_tensor, dim=0)
        centroids.append(center)

    centroids = torch.stack(centroids, dim=0)
    return centroids


def calculate_class_centroids_FFDNet_AugMix(model, loader, class_num=7):
    feat_list = []
    label_list = []

    with torch.no_grad():
        for i, (img, label) in tqdm(enumerate(loader)):
            img = img[0].cuda()
            label = label.cuda()
            _, sem_feat = model.get_logits_feat(img)
            feat_list.append(sem_feat)
            label_list.append(label)

    feat_tensor = torch.cat(feat_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)

    centroids = []

    for i in range(class_num):
        index = torch.where(label_tensor == i)
        class_tensor = feat_tensor[index]
        center = torch.mean(class_tensor, dim=0)
        centroids.append(center)

    centroids = torch.stack(centroids, dim=0)
    return centroids


def calculate_domain_centroids_FFDNet(model, loader, module_name='se'):
    feat_list = []
    # label_list = []

    # model_trunc = create_feature_extractor(model, return_nodes={module_name: 'semantic_feature'})
    with torch.no_grad():
        for i, (img, label) in tqdm(enumerate(loader)):
            img = img.cuda()
            label = label.cuda()
            # zero_tensor = torch.zeros(label.shape, device=img.device, dtype=label.dtype)
            # pred_logits = model_trunc(img)
            # sem_feat = pred_logits['semantic_feature']
            _, sem_feat = model.get_logits_feat(img)
            feat_list.append(sem_feat)

    feat_tensor = torch.cat(feat_list, dim=0)
    domain_centroid = torch.mean(feat_tensor, dim=0)
    domain_centroid = domain_centroid.unsqueeze(dim=0)
    domain_centroid = domain_centroid.detach()
    return domain_centroid


def JS_Divergence(output_clean, output_aug1, output_aug2, output_aug3):
    p_clean, p_aug1 = F.softmax(output_clean, dim=1), F.softmax(output_aug1, dim=1)
    p_aug2, p_aug3 = F.softmax(output_aug2, dim=1), F.softmax(output_aug3, dim=1)
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2 + p_aug3) / 4., 1e-7, 1).log()
    loss_ctr = (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug1,
                                                                               reduction='batchmean') + F.kl_div(
        p_mixture, p_aug2, reduction='batchmean') + F.kl_div(p_mixture, p_aug3, reduction='batchmean')) / 4.
    return loss_ctr
