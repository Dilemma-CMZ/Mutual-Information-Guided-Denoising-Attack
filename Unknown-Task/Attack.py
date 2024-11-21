import os
import sys
import shutil
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from basicsr.models.archs.restormer_arch import Restormer
from PIL import Image
import yaml
from torchvision import models, transforms

GPU_ID = 0

if torch.cuda.is_available():
    device = torch.device(f'cuda:{GPU_ID}')
    torch.cuda.set_device(device)
    print(f"Using GPU: {GPU_ID}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")

def load_image_from_file(image_path):
    image_pil = Image.open(image_path).convert('RGB')
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return image_tensor

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, layers_to_extract):
        super(VGG16FeatureExtractor, self).__init__()
        self.layers_to_extract = layers_to_extract
        self.vgg = models.vgg16(weights=None).features.to(device).eval()

        local_path = './pretrained_models/vgg16-397923af.pth'
        if not os.path.exists(local_path):
            print(f"VGG model doesn't exist: {local_path}")
            sys.exit(1)
        state_dict = torch.load(local_path, map_location=device)
        new_state_dict = {k.replace('features.', ''): v for k, v in state_dict.items() if 'features' in k}
        self.vgg.load_state_dict(new_state_dict, strict=False)

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers_to_extract:
                outputs[i] = x
            if i >= max(self.layers_to_extract):
                break 
        return outputs

LAYERS_TO_USE = [4, 9, 16]

vgg = VGG16FeatureExtractor(LAYERS_TO_USE).to(device)

def perceptual_loss(img1, img2, vgg_model, layers):
    features_img1 = vgg_model(img1)
    features_img2 = vgg_model(img2)
    loss = torch.zeros(img1.size(0), device=img1.device)
    for layer in layers:
        feat1 = features_img1[layer]
        feat2 = features_img2[layer]
        layer_loss = F.mse_loss(feat1, feat2, reduction='none').view(feat1.size(0), -1).mean(dim=1)
        loss += layer_loss
    return loss

def transform_images_for_mine(images):
    images_resized = F.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    images_normalized = torch.stack([normalize(img) for img in images_resized])
    return images_normalized

class MINE(nn.Module):
    def __init__(self, input_channels, image_size):
        super(MINE, self).__init__()
        self.conv1 = nn.Conv2d(input_channels * 2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout4 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.act5 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout5 = nn.Dropout(0.25)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        reduced_size = image_size // (2 ** 5)  
        self.fc1 = nn.Linear(512 * reduced_size * reduced_size, 512)
        self.act_fc1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.act_fc2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, y):
        xy = torch.cat((x, y), dim=1)
        h = self.conv1(xy)
        h = self.bn1(h)
        h = self.act1(h)
        h = self.dropout1(h)
        h = self.pool(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act2(h)
        h = self.dropout2(h)
        h = self.pool(h)
        h = self.conv3(h)
        h = self.bn3(h)
        h = self.act3(h)
        h = self.dropout3(h)
        h = self.pool(h)
        h = self.conv4(h)
        h = self.bn4(h)
        h = self.act4(h)
        h = self.dropout4(h)
        h = self.pool(h)
        h = self.conv5(h)
        h = self.bn5(h)
        h = self.act5(h)
        h = self.dropout5(h)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = self.act_fc1(h)
        h = self.dropout_fc1(h)
        h = self.fc2(h)
        h = self.act_fc2(h)
        h = self.dropout_fc2(h)

        output = self.fc3(h)

        return output

def load_mine_model(model_path, device, input_channels=3, image_size=64):
    model = MINE(input_channels, image_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  
    return model

mine_model_path = 'mine_model.pth'
mine_model = load_mine_model(mine_model_path, device, input_channels=3, image_size=64)
print("MINE model loaded successfully.")

def make_pixel_adversary(original_images, target_images, model_restoration, vgg_model, layers, n_opt_steps=100, log_files=None, mine_model=None, lambda_mi=0.01):
    delta = torch.zeros_like(original_images, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([delta], lr=4e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    lambda_reg = 1e-4      
    lambda_original = 3.0   

    for step in trange(n_opt_steps, desc="Optimization Steps"):
        optimizer.zero_grad()
        perturbed_images = torch.clamp(original_images + delta, 0, 1)
        restored_images = model_restoration(perturbed_images)

        loss_mse_per_image = F.mse_loss(restored_images, target_images, reduction='none')
        loss_mse_per_image = loss_mse_per_image.view(loss_mse_per_image.size(0), -1).mean(dim=1)
        loss_perc_per_image = perceptual_loss(restored_images, target_images, vgg_model, layers)
        loss_original_per_image = F.mse_loss(perturbed_images, original_images, reduction='none')
        loss_original_per_image = loss_original_per_image.view(loss_original_per_image.size(0), -1).mean(dim=1)
        loss_perc_original_per_image = perceptual_loss(perturbed_images, original_images, vgg_model, layers)
        loss_reg_per_image = lambda_reg * torch.norm(delta.view(delta.size(0), -1), dim=1)

        images_mine = transform_images_for_mine(restored_images)
        target_images_mine = transform_images_for_mine(target_images)
        mi_output = mine_model(images_mine, target_images_mine)
        mi_loss_per_image = -mi_output.view(-1) 

        mse_1 = 1
        vgg_1 = 1
        mse_2 = 1
        vgg_2 = 1
        Mine = 1
        loss_total_per_image = (
            mse_2 * loss_mse_per_image +
            vgg_2 * 0.1 * loss_perc_per_image +
            lambda_original * (mse_1 * loss_original_per_image + vgg_1 * 0.1 * loss_perc_original_per_image) +
            loss_reg_per_image +
            Mine * lambda_mi * mi_loss_per_image  
        )

        loss = loss_total_per_image.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if ((step + 1) % 10 == 0):
            for idx in range(original_images.size(0)):
                log_message = (
                    f"Image {idx+1}, Step [{step + 1}/{n_opt_steps}], Total Loss: {loss_total_per_image[idx].item():.6f}, "
                    f"MSE Loss: {loss_mse_per_image[idx].item():.6f}, Perceptual Loss: {loss_perc_per_image[idx].item():.6f}, "
                    f"Original Similarity Loss: {loss_original_per_image[idx].item():.6f}, Original Perceptual Loss: {loss_perc_original_per_image[idx].item():.6f}, "
                    f"MI Loss: {mi_loss_per_image[idx].item():.6f}"
                )
                if log_files is not None:
                    log_files[idx].write(log_message + '\n')

    with torch.no_grad():
        perturbed_images = torch.clamp(original_images + delta, 0, 1).detach()
        restored_images = model_restoration(perturbed_images).detach()

    return perturbed_images, restored_images

with open('Options/RealDenoising_Restormer.yml', 'r') as f:
    config = yaml.safe_load(f)
config_network_g = config['network_g'].copy()
config_network_g.pop('type', None)
model_restoration = Restormer(**config_network_g)
checkpoint = torch.load('./pretrained_models/real_denoising.pth', map_location=device)
model_restoration.load_state_dict(checkpoint['params'])
model_restoration.eval().to(device)

def adversarial_attack(original_image_paths, target_image_paths, output_folders, vgg_model, layers, n_opt_steps=100, mine_model=None, lambda_mi=0.01):
    original_images = []
    target_images = []
    log_files = []
    original_sizes = [] 
    for idx in range(len(original_image_paths)):
        original_image_path = original_image_paths[idx]
        target_image_path = target_image_paths[idx]
        output_folder = output_folders[idx]
        log_file_path = os.path.join(output_folder, 'log.txt')

        original_image = load_image_from_file(original_image_path)
        target_image = load_image_from_file(target_image_path)

        _, _, h, w = original_image.size()
        original_sizes.append((h, w))

        if original_image.shape != target_image.shape:
            print(f"Original and target images have different sizes for {original_image_path}. Resizing target image.")
            target_image_pil = Image.open(target_image_path).convert('RGB')
            original_size = (original_image.shape[3], original_image.shape[2]) 
            target_image_pil = target_image_pil.resize(original_size, Image.BILINEAR)
            target_image_np = np.array(target_image_pil).astype(np.float32) / 255.0
            target_image = torch.from_numpy(target_image_np).permute(2, 0, 1).unsqueeze(0).to(device)

        original_images.append(original_image)
        target_images.append(target_image)

        log_file = open(log_file_path, 'w')
        log_files.append(log_file)

    heights = []
    widths = []
    for img in original_images:
        heights.append(img.size(2)) 
        widths.append(img.size(3))

    max_height = max(heights)
    max_width = max(widths)

    padded_original_images = []
    padded_target_images = []

    for idx in range(len(original_images)):
        img = original_images[idx]
        target_img = target_images[idx]
        _, _, h, w = img.size()
        pad_left = (max_width - w) // 2
        pad_right = max_width - w - pad_left
        pad_top = (max_height - h) // 2
        pad_bottom = max_height - h - pad_top
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        img_padded = F.pad(img, padding, mode='constant', value=0)
        target_img_padded = F.pad(target_img, padding, mode='constant', value=0)
        padded_original_images.append(img_padded)
        padded_target_images.append(target_img_padded)

    original_images = torch.cat(padded_original_images, dim=0)
    target_images = torch.cat(padded_target_images, dim=0)

    for idx in range(len(original_image_paths)):
        original_image_np = original_images[idx].permute(1, 2, 0).cpu().numpy()
        original_image_pil = Image.fromarray((original_image_np * 255).astype('uint8'))
        output_folder = output_folders[idx]
        h, w = original_sizes[idx]
        original_image_pil = original_image_pil.crop((0, 0, w, h))
        original_image_pil.save(os.path.join(output_folder, 'original.png'))

    with torch.no_grad():
        restored_image_original = model_restoration(original_images)

    for idx in range(len(original_image_paths)):
        restored_original_np = restored_image_original[idx].permute(1, 2, 0).cpu().numpy()
        restored_original_pil = Image.fromarray((restored_original_np * 255).astype('uint8'))
        output_folder = output_folders[idx]
        h, w = original_sizes[idx]
        restored_original_pil = restored_original_pil.crop((0, 0, w, h))
        restored_original_pil.save(os.path.join(output_folder, 'restored_original.png'))

    perturbed_images, restored_images = make_pixel_adversary(
        original_images, target_images, model_restoration, vgg_model, layers, n_opt_steps=n_opt_steps, log_files=log_files, mine_model=mine_model, lambda_mi=lambda_mi)

    for log_file in log_files:
        log_file.close()

    for idx in range(len(original_image_paths)):
        output_folder = output_folders[idx]
        h, w = original_sizes[idx]

        perturbed_image_np = perturbed_images[idx].permute(1, 2, 0).cpu().numpy()
        perturbed_image_pil = Image.fromarray((perturbed_image_np * 255).astype('uint8'))
        perturbed_image_pil = perturbed_image_pil.crop((0, 0, w, h))
        perturbed_image_pil.save(os.path.join(output_folder, 'perturbed.png'))

        restored_image_np = restored_images[idx].permute(1, 2, 0).cpu().numpy()
        restored_image_pil = Image.fromarray((restored_image_np * 255).astype('uint8'))
        restored_image_pil = restored_image_pil.crop((0, 0, w, h))
        restored_image_pil.save(os.path.join(output_folder, 'restored_perturbed.png'))

    print("Processing completed for batch")

original_folder = '/root/autodl-tmp/dataset1_MINE/origin_sigma=25'
target_folder = '/root/autodl-tmp/dataset1_MINE/target'
result_folder = '/root/autodl-tmp/dataset1_MINE/result_restormer'
os.makedirs(result_folder, exist_ok=True)
batch_size = 2  

file_numbers = sorted(
    [f.split('.')[0] for f in os.listdir(original_folder) if f.endswith('.png') or f.endswith('.jpg')],
    key=lambda x: int(x)
)
expected_numbers = [str(i) for i in range(1, len(file_numbers) + 1)]
if file_numbers != expected_numbers:
    print("File numbers are not continuous or do not start from 1. Please check the filenames.")
else:
    print("All file numbers start from 1 and increase consecutively. Total files:", len(file_numbers))

batches = [file_numbers[i:i + batch_size] for i in range(0, len(file_numbers), batch_size)]

for batch in tqdm(batches, desc="Processing Batches"):
    original_image_paths = []
    target_image_paths = []
    output_folders = []

    for file_number in batch:
        original_image_path = os.path.join(original_folder, f"{file_number}.png")
        target_image_path = os.path.join(target_folder, f"{file_number}.png")

        if not os.path.exists(original_image_path) or not os.path.exists(target_image_path):
            print(f"Original or target image does not exist: {original_image_path} or {target_image_path}")
            continue

        current_result_folder = os.path.join(result_folder, file_number)
        os.makedirs(current_result_folder, exist_ok=True)

        shutil.copy(original_image_path, os.path.join(current_result_folder, 'original.png'))
        shutil.copy(target_image_path, os.path.join(current_result_folder, 'target.png'))

        original_image_paths.append(os.path.join(current_result_folder, 'original.png'))
        target_image_paths.append(os.path.join(current_result_folder, 'target.png'))
        output_folders.append(current_result_folder)

    if original_image_paths:
        adversarial_attack(
            original_image_paths=original_image_paths,
            target_image_paths=target_image_paths,
            output_folders=output_folders,
            vgg_model=vgg,
            layers=LAYERS_TO_USE,
            n_opt_steps=100,
            mine_model=mine_model,
            lambda_mi=0.01  
        )
