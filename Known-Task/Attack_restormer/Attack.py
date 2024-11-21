import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import yaml
from basicsr.models.archs.restormer_arch import Restormer
import torch.nn.functional as F
import sys
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageNet10Dataset(Dataset):
    def __init__(self, noisy_img_dirs, gt_img_dirs, class_ids, transform=None):
        self.noisy_img_paths = []
        self.gt_img_paths = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(class_ids)}
        self.idx_to_class = {idx: class_id for class_id, idx in self.class_to_idx.items()}
        self.img_names = []

        for noisy_img_dir, gt_img_dir in zip(noisy_img_dirs, gt_img_dirs):
            for class_id in class_ids:
                noisy_class_dir = os.path.join(noisy_img_dir, class_id)
                gt_class_dir = os.path.join(gt_img_dir, class_id)
                if not os.path.isdir(noisy_class_dir) or not os.path.isdir(gt_class_dir):
                    print(f"Class directory {noisy_class_dir} or {gt_class_dir} does not exist, skipping.")
                    continue
                img_names = os.listdir(noisy_class_dir)
                for img_name in img_names:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        noisy_img_path = os.path.join(noisy_class_dir, img_name)
                        gt_img_path = os.path.join(gt_class_dir, img_name)
                        if not os.path.exists(gt_img_path):
                            print(f"GT image does not exist: {gt_img_path}, skipping.")
                            continue
                        self.noisy_img_paths.append(noisy_img_path)
                        self.gt_img_paths.append(gt_img_path)
                        self.labels.append(self.class_to_idx[class_id])
                        self.img_names.append(img_name)

    def __len__(self):
        return len(self.noisy_img_paths)

    def __getitem__(self, idx):
        noisy_img_path = self.noisy_img_paths[idx]
        gt_img_path = self.gt_img_paths[idx]
        label = self.labels[idx]
        img_name = self.img_names[idx]
        try:
            noisy_image = Image.open(noisy_img_path).convert('RGB')
            gt_image = Image.open(gt_img_path).convert('RGB')
        except Exception as e:
            print(f"Unable to read image: {noisy_img_path} or {gt_img_path}, error: {e}")
            noisy_image = Image.new('RGB', (224, 224), (0, 0, 0))
            gt_image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            noisy_image = self.transform(noisy_image)
            gt_image = self.transform(gt_image)
        return noisy_image, gt_image, label, img_name, self.idx_to_class[label]

def prepare_dataloader(noisy_img_dirs, gt_img_dirs, class_ids, batch_size=2, num_workers=4, transform=None):
    dataset = ImageNet10Dataset(noisy_img_dirs=noisy_img_dirs, gt_img_dirs=gt_img_dirs, class_ids=class_ids, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return dataloader

def load_restormer_model(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config_network_g = config['network_g'].copy()
    config_network_g.pop('type', None)
    model_restoration = Restormer(**config_network_g)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_restoration.load_state_dict(checkpoint['params'])
    model_restoration.eval().to(device)
    return model_restoration

def initialize_model(num_classes=10):
    resnet = torchvision.models.resnet50(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)
    return resnet

def load_vgg16(device, local_path='./pretrained_models/vgg16-397923af.pth'):
    if not os.path.exists(local_path):
        print(f"VGG16 pretrained model file does not exist: {local_path}")
        sys.exit(1)
    vgg = models.vgg16(pretrained=False).features.eval().to(device)
    state_dict = torch.load(local_path, map_location=device)
    features_state_dict = {k: v for k, v in state_dict.items() if 'features' in k}
    new_state_dict = {k.replace('features.', ''): v for k, v in features_state_dict.items()}
    vgg.load_state_dict(new_state_dict, strict=False)
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

def perceptual_loss(vgg, img1, img2):
    return F.mse_loss(vgg(img1), vgg(img2))

def optimized_attack(model_restoration, model_classifier, vgg, images_noisy, images_gt, labels, epsilon, num_iterations):
    images_noisy = images_noisy.to(device)
    images_gt = images_gt.to(device)
    labels = labels.to(device)
    delta = torch.zeros_like(images_noisy, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([delta], lr=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    lambda_reg = 1e-4
    lambda_original = 3.0
    enable_mse_denoised_gt = 1
    enable_vgg_denoised_gt = 1
    enable_mse_perturbed_noisy = 1
    enable_vgg_perturbed_noisy = 1
    enable_label_mse_loss = 1

    for step in range(num_iterations):
        optimizer.zero_grad()
        perturbed_images = torch.clamp(images_noisy + delta, 0, 1)
        denoised_images = model_restoration(perturbed_images)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1,3,1,1)
        denoised_images_norm = (denoised_images - mean) / std
        outputs = model_classifier(denoised_images_norm)
        loss_mse_denoised_gt = F.mse_loss(denoised_images, images_gt, reduction='none')
        loss_mse_denoised_gt = loss_mse_denoised_gt.view(loss_mse_denoised_gt.size(0), -1).mean(dim=1)

        if enable_vgg_denoised_gt:
            vgg_features_denoised = vgg(denoised_images)
            vgg_features_gt = vgg(images_gt)
            loss_vgg_denoised_gt = F.mse_loss(vgg_features_denoised, vgg_features_gt, reduction='none')
            loss_vgg_denoised_gt = loss_vgg_denoised_gt.view(loss_vgg_denoised_gt.size(0), -1).mean(dim=1)
        else:
            loss_vgg_denoised_gt = torch.zeros(images_noisy.size(0)).to(device)

        loss_mse_perturbed_noisy = F.mse_loss(perturbed_images, images_noisy, reduction='none')
        loss_mse_perturbed_noisy = loss_mse_perturbed_noisy.view(loss_mse_perturbed_noisy.size(0), -1).mean(dim=1)

        if enable_vgg_perturbed_noisy:
            vgg_features_perturbed = vgg(perturbed_images)
            vgg_features_noisy = vgg(images_noisy)
            loss_vgg_perturbed_noisy = F.mse_loss(vgg_features_perturbed, vgg_features_noisy, reduction='none')
            loss_vgg_perturbed_noisy = loss_vgg_perturbed_noisy.view(loss_vgg_perturbed_noisy.size(0), -1).mean(dim=1)
        else:
            loss_vgg_perturbed_noisy = torch.zeros(images_noisy.size(0)).to(device)

        loss_reg_per_image = lambda_reg * torch.norm(delta.view(delta.size(0), -1), dim=1)

        if enable_label_mse_loss:
            outputs_softmax = torch.softmax(outputs, dim=1)
            labels_one_hot = torch.zeros_like(outputs_softmax).scatter_(1, labels.view(-1,1), 1)
            loss_label_mse = -10 * F.mse_loss(outputs_softmax, labels_one_hot, reduction='none').mean(dim=1)
        else:
            loss_label_mse = torch.zeros(images_noisy.size(0)).to(device)

        loss_total_per_image = (
            enable_mse_denoised_gt * loss_mse_denoised_gt +
            enable_vgg_denoised_gt * 0.1 * loss_vgg_denoised_gt +
            lambda_original * (enable_mse_perturbed_noisy * loss_mse_perturbed_noisy +
                               enable_vgg_perturbed_noisy * 0.1 * loss_vgg_perturbed_noisy) +
            enable_label_mse_loss * loss_label_mse +
            loss_reg_per_image
        )
        loss = loss_total_per_image.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(images_noisy + delta.data, 0, 1) - images_noisy

    with torch.no_grad():
        perturbed_images = torch.clamp(images_noisy + delta, 0, 1).detach()

    return perturbed_images

if __name__ == '__main__':
    class_ids = [
        'n02056570', 'n02085936', 'n02128757', 'n02690373', 'n02692877',
        'n03095699', 'n04254680', 'n04285008', 'n04467665', 'n07747607'
    ]
    restorer_config_path = 'Options/GaussianColorDenoising_RestormerSigma50.yml'
    restorer_checkpoint_path = './pretrained_models/gaussian_color_denoising_sigma50.pth'
    print("Loading Restormer model...")
    model_restoration = load_restormer_model(restorer_config_path, restorer_checkpoint_path)

    print("Loading ResNet50_A model...")
    resnet = initialize_model(num_classes=10)
    resnet.load_state_dict(torch.load('resnet50_A.pth', map_location=device))
    resnet.eval()

    print("Loading VGG16 model for perceptual loss...")
    vgg = load_vgg16(device, local_path='./pretrained_models/vgg16-397923af.pth')

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    test_noisy_dir = '/root/autodl-tmp/imagenet-10/imagenet-10-noisy-50/test'
    test_gt_dir = '/root/autodl-tmp/imagenet-10/imagenet-10/test'
    dataloader = prepare_dataloader(
        noisy_img_dirs=[test_noisy_dir],
        gt_img_dirs=[test_gt_dir],
        class_ids=class_ids,
        batch_size=2,
        num_workers=4,
        transform=eval_transform
    )

    epsilon = 4/255
    num_iterations = 2

    adv_save_dir = '/root/autodl-tmp/imagenet-10/ablation/case1/test'
    for class_id in class_ids:
        os.makedirs(os.path.join(adv_save_dir, class_id), exist_ok=True)

    print("Starting adversarial attack and saving samples...")
    correct = 0
    total = 0

    for images_noisy, images_gt, labels, img_names, class_ids_batch in tqdm(dataloader):
        images_noisy = images_noisy.to(device)
        images_gt = images_gt.to(device)
        labels = labels.to(device)
        images_adv = optimized_attack(model_restoration, resnet, vgg, images_noisy, images_gt, labels, epsilon, num_iterations)

        images_adv_cpu = images_adv.cpu()
        for i in range(images_adv_cpu.size(0)):
            img_adv = images_adv_cpu[i]
            img_pil = transforms.ToPILImage()(img_adv)
            class_id = class_ids_batch[i]
            img_name = img_names[i]
            save_path = os.path.join(adv_save_dir, class_id, img_name)
            img_pil.save(save_path)

        with torch.no_grad():
            denoised_images = model_restoration(images_adv)
            mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1,3,1,1)
            denoised_images_norm = (denoised_images - mean) / std
            outputs = resnet(denoised_images_norm)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Model accuracy on adversarial samples: {accuracy:.2f}%")
