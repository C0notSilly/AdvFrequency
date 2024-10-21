import torch
import torch.nn as nn
from utils.adv_freq import FFT_Batch, IFFT_Batch
from tqdm import tqdm
from torchvision import transforms



class AmpDrop(nn.Module):
    """
        adversarial amp drop
    """

    def __init__(self, model, threshold=1.0):
        super(AmpDrop, self).__init__()
        self.model = model
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.threshold = threshold
        # self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def get_logits(self, img):
        amp, phase = FFT_Batch(img)
        recon_img = amp * torch.exp(1j * phase)
        recon_img = torch.fft.ifftshift(recon_img)
        recon_img = torch.abs(torch.fft.ifft2(recon_img))
        recon_img = recon_img / torch.max(recon_img)
        recon_img = self.normalize(recon_img)

        logits, sem_feat = self.model.get_logits_feat(recon_img)

        # return phase, amp, logits, d_logits, sem_feat
        return amp, phase, logits, sem_feat

    # def forward(self, img, label, domain_centroids):
    def forward(self, img, label, class_centroids=None):
        img_copy = img.detach()
        if class_centroids is not None:
            selected_centroids = class_centroids[label]
        else:
            selected_centroids = None
        img_copy.requires_grad = True
        loss = nn.CrossEntropyLoss()
        distance = nn.MSELoss()

        amp, phase, logits, sem_feat = self.get_logits(img_copy)

        # standard CE loss for classification
        # cost = loss(logits, label)
        # CE loss for classifier and domain discriminator
        if class_centroids is not None:
            cost = loss(logits, label) + distance(sem_feat, selected_centroids)
        else:
            cost = loss(logits, label)
        # cost = loss(logits, label) + distance(sem_feat, selected_centroids)

        grad = torch.autograd.grad(cost, amp, retain_graph=False, create_graph=False)[0]

        flatten_grad = grad.reshape(grad.shape[0], grad.shape[1], -1)
        grad_max = torch.max(flatten_grad, dim=2, keepdim=True).values
        grad_min = torch.min(flatten_grad, dim=2, keepdim=True).values

        # min max scaler
        ex_max = grad_max.unsqueeze(dim=3).expand(img.size())
        ex_min = grad_min.unsqueeze(dim=3).expand(img.size())

        scaler_grad = (grad - ex_min) / (ex_max - ex_min)

        # scaler_grad = torch.sigmoid(grad)
        prob = self.threshold * torch.rand(scaler_grad.shape, device=scaler_grad.device)
        dropout_mask_1 = torch.where(scaler_grad > prob, 0, 1)
        '''
        dropout_mask_1 = torch.where(scaler_grad > prob,
                                     torch.sigmoid((scaler_grad - prob)) - 0.5 * torch.rand(scaler_grad.shape,
                                                                                            device=scaler_grad.device),
                                     1)
        '''
        dropout_amp = dropout_mask_1 * amp

        prob = torch.rand(scaler_grad.shape, device=scaler_grad.device)
        dropout_mask_2 = torch.where(scaler_grad > prob, 0, 1)
        '''
        dropout_mask_2 = torch.where(scaler_grad > prob,
                                     torch.sigmoid((scaler_grad - prob)) - 0.5 * torch.rand(scaler_grad.shape,
                                                                                            device=scaler_grad.device),
                                     1)
        '''
        dropout_amp_2 = dropout_mask_2 * amp

        # reconstruct with dropout_amp and original phase
        adv_img = IFFT_Batch(dropout_amp, phase)
        adv_img_2 = IFFT_Batch(dropout_amp_2, phase)

        adv_img = adv_img.detach()
        adv_img_2 = adv_img_2.detach()

        return adv_img, adv_img_2


class AmpUCNoise(nn.Module):
    def __init__(self, model, steps=5, eps=1, alpha=1):
        super(AmpUCNoise, self).__init__()
        self.model = model
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.steps = steps
        self.eps = eps
        self.alpha = alpha

    def get_logits(self, amp, phase):
        img = IFFT_Batch(amp, phase)
        img = self.normalize(img)
        return self.model.get_logits_feat(img)

    def forward(self, img, label, class_centroids=None):
        img_copy = img.detach()
        if class_centroids is not None:
            selected_centroids = class_centroids[label]
        else:
            selected_centroids = None
        loss = nn.CrossEntropyLoss()
        distance = nn.MSELoss()

        amp, phase = FFT_Batch(img)
        amp.requires_grad = True

        B, C, H, W = img.size()

        amp_mean = torch.mean(
            amp[:, :, int(0.9 * (H // 2)):int(1.1 * (H // 2)), int(0.9 * (W // 2)):int(1.1 * (W // 2))], dim=(2, 3),
            keepdim=True)
        amp_std = torch.sqrt(torch.var(
            amp[:, :, int(0.9 * (H // 2)):int(1.1 * (H // 2)), int(0.9 * (W // 2)):int(1.1 * (W // 2))], dim=(2, 3),
            keepdim=True))

        mask = torch.zeros(amp.shape, device=amp.device)
        mask[:, :, int(0.9 * (H // 2)):int(1.1 * (H // 2)), int(0.9 * (W // 2)):int(1.1 * (W // 2))] = 1
        neg_mask = torch.where(mask == 0, 1, 0)
        mask = mask.detach()
        neg_mask = neg_mask.detach()

        uc_mean = torch.sqrt(torch.var(amp_mean, dim=0, keepdim=True)) + 1e-4
        uc_std = torch.sqrt(torch.var(amp_std, dim=0, keepdim=True)) + 1e-4

        dir_mean = torch.randn(amp_mean.shape, device=amp_mean.device)
        dir_std = torch.randn(amp_std.shape, device=amp_std.device)

        for i in range(self.steps):
            dir_mean.requires_grad = True
            dir_std.requires_grad = True

            recon_mean = amp_mean + dir_mean * uc_mean
            recon_std = amp_std + dir_std * uc_std

            adv_amp = (((amp - amp_mean) / (amp_std+1e-4)) * (recon_std+1e-4) + recon_mean) * mask + amp * neg_mask

            logits, feat = self.get_logits(adv_amp, phase)

            if class_centroids is not None:
                cost = loss(logits, label) + distance(feat, selected_centroids)
            else:
                cost = loss(logits, label)

            dm_grad = torch.autograd.grad(cost, dir_mean, retain_graph=True, create_graph=False)[0]
            ds_grad = torch.autograd.grad(cost, dir_std, )[0]

            dir_mean = dir_mean + self.alpha * dm_grad.sign()
            dir_std = dir_std + self.alpha * ds_grad.sign()

            dir_mean = dir_mean.detach()
            dir_std = dir_std.detach()

        recon_mean = amp_mean + dir_mean * uc_mean
        recon_std = amp_std + dir_std * uc_std

        adv_amp = ((amp - amp_mean) / (1e-4+amp_std) * (1e-4+recon_std) + recon_mean) * mask + amp * neg_mask
        adv_img = IFFT_Batch(adv_amp, phase)
        return adv_img


