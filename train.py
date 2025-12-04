import os
import time
import wandb
import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
from tqdm import tqdm
import warnings
import random
import pickle
import sys
from scipy.stats import entropy
from sklearn.metrics.pairwise import polynomial_kernel
import torch.nn as nn
import torchvision
from torch.hub import load_state_dict_from_url

warnings.filterwarnings('ignore')
#os.environ["WANDB_API_KEY"] = "652d515cee7db8d8467a876756b9b097b9a342b1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from data.dataset import TextDataset, TextDatasetval
from models.model import WriteViT
from params import *

# ==================== SEED SETTING FOR REPRODUCIBILITY ====================
def set_seed(seed=42):
    """Set seed for complete reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed} for reproducibility")

# ==================== FID/KID CALCULATION CODE ====================
# Adapted from https://github.com/bioinf-jku/TTUR and https://github.com/mbinkowski/MMD-GAN

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps and logits"""
    
    DEFAULT_BLOCK_INDEX = 3
    
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling features
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, 'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        self.last_fc = inception.fc

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp, inp_len=None):
        """Get Inception feature maps
        
        Parameters
        ----------
        inp : torch.Tensor
            Input tensor of shape BxCxHxW. Values are expected to be in range (0, 1)
        inp_len : torch.Tensor, optional
            Lengths for variable-width images
            
        Returns
        -------
        features : torch.Tensor
            Features from the specified block
        logits : torch.Tensor
            Class logits
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.normalize_input:
            x = x * 2 - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        # Handle variable-length inputs if needed
        if inp_len is not None and outp:
            # Mask and average over the width dimension
            mask = self._len2mask(inp_len, outp[-1].size(-1))
            feat = outp[-1].squeeze(2) * mask.unsqueeze(1)
            feat = feat.sum(dim=-1) / (inp_len.unsqueeze(dim=1) + 1e-8)
            outp[-1] = feat.view(*feat.size(), 1, 1)

        # Get logits from the final features
        logits = self.last_fc(outp[-1].squeeze())
        
        return outp[-1], logits

    def _len2mask(self, length, max_len, dtype=torch.float32):
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask


def _inception_v3(*args, **kwargs):
    """Wraps torchvision.models.inception_v3"""
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        version = (0,)

    if version >= (0, 6):
        kwargs['init_weights'] = False

    return torchvision.models.inception_v3(*args, **kwargs)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'

def fid_inception_v3():
    """Build pretrained Inception model for FID computation"""
    inception = _inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    
    # Patch FID-specific layers
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    inception.fc = nn.Linear(2048, 1008)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use padded zeros
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use padded zeros
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use padded zeros
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling here
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True, output=sys.stdout, **kernel_args):
    """Compute MMD averages over subsets"""
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    if subset_size > len(codes_g):
        print(f'Warning: subset_size is bigger than len(codes_g). [sub:{subset_size} code_g:{len(codes_g)}]')
        subset_size = len(codes_g)
    if subset_size > len(codes_r):
        print(f'Warning: subset_size is bigger than len(codes_r). [sub:{subset_size} code_r:{len(codes_r)}]')
        subset_size = len(codes_r)

    with tqdm(range(n_subsets), desc='MMD', file=output, disable=True) as bar:
        for i in bar:
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    """Compute polynomial MMD"""
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)


def _sqn(arr):
    """Square norm"""
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    """Compute MMD variance"""
    m = K_XX.shape[0]
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est


def calculate_inception_score(logits, splits=1):
    """Calculate Inception Score"""
    split_scores = []
    N = logits.shape[0]

    for k in range(splits):
        part = logits[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)


def get_activations(data_source, n_batches, model, dims, device, crop=False):
    """Calculate activations of the pool_3 layer for all images."""
    model.eval()
    pred_arr, pred_logits = [], []
    
    for batch in data_source:
        imgs = batch['img'].to(device)
        
        # Normalize from [-1, 1] to [0, 1] if needed
        if imgs.min() < 0:
            imgs = (imgs + 1) / 2
        
        # Convert grayscale to RGB
        if imgs.size(1) == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            if not crop:
                # Handle variable width if needed
                if 'wcl' in batch:
                    # Use writer class lengths if available
                    pred, logits = model(imgs, batch['wcl'].to(device))
                else:
                    pred, logits = model(imgs)
            else:
                # Crop mode (if needed)
                pred, logits = model(imgs[:, :, :, :imgs.size(-2) * 2], 
                                    torch.ones((imgs.size(0),)).to(device) * 4)

        # Apply global spatial average pooling if needed
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr.append(pred.cpu().data.numpy().reshape(pred.size(0), -1))
        pred_logits.append(logits.cpu().data.numpy())

    pred_arr = np.concatenate(pred_arr, axis=0)
    pred_logits = np.concatenate(pred_logits, axis=0)
    assert pred_arr.shape[-1] == dims
    return pred_arr, pred_logits
def calculate_fid_and_kid_for_writevit(model, dataloader, num_samples=500, cfg=None):
    """Calculate FID and KID for WriteViT model with proper preprocessing"""
    
    # Initialize Inception model with FID weights
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx], use_fid_inception=True).to(DEVICE)
    inception_model.eval()
    
    real_features_list = []
    fake_features_list = []
    real_logits_list = []
    fake_logits_list = []
    
    print(f"\nCalculating FID & KID with {num_samples} samples...")
    
    # Process real images
    real_count = 0
    for batch in dataloader:
        if real_count >= num_samples:
            break
            
        real_imgs = batch['img'].to(DEVICE)
        batch_size = real_imgs.size(0)
        
        # Normalize and convert to RGB
        real_imgs_norm = (real_imgs + 1) / 2 if real_imgs.min() < 0 else real_imgs
        if real_imgs_norm.size(1) == 1:
            real_imgs_norm = real_imgs_norm.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            features, logits = inception_model(real_imgs_norm)
            
        real_features_list.append(features.cpu().numpy().reshape(batch_size, -1))
        real_logits_list.append(logits.cpu().numpy())
        real_count += batch_size
        
        if real_count >= num_samples:
            break
    
    # Process fake images
    fake_count = 0
    for batch in dataloader:
        if fake_count >= num_samples:
            break
            
        batch_size = batch['img'].size(0)
        
        # Generate fake images using WriteViT
        with torch.no_grad():
            sdata = batch['img'].to(DEVICE)
            
            # Sample random text from lexicon
            sample_lex_idx = model.fake_y_dist.sample([batch_size])
            fake_y = [model.lex[i].encode("utf-8") for i in sample_lex_idx]
            
            # Encode text
            text_encode_fake, len_text_fake = model.netconverter.encode(fake_y)
            text_encode_fake = text_encode_fake.to(DEVICE)
            
            # Generate features and fake images
            feat_w, _ = model.netW(sdata.detach(), batch['wcl'].to(DEVICE))
            fakes = model.netG(feat_w, text_encode_fake)
            
            # Handle different output formats
            if isinstance(fakes, list):
                fake_batch = fakes[0].detach()
            else:
                fake_batch = fakes.detach()
            
            # Normalize and convert to RGB
            fake_batch_norm = (fake_batch + 1) / 2 if fake_batch.min() < 0 else fake_batch
            if fake_batch_norm.dim() == 3:  # [B, H, W]
                fake_batch_norm = fake_batch_norm.unsqueeze(1)  # [B, 1, H, W]
            if fake_batch_norm.size(1) == 1:
                fake_batch_norm = fake_batch_norm.repeat(1, 3, 1, 1)
            
            features, logits = inception_model(fake_batch_norm)
            
        fake_features_list.append(features.cpu().numpy().reshape(batch_size, -1))
        fake_logits_list.append(logits.cpu().numpy())
        fake_count += batch_size
        
        if fake_count >= num_samples:
            break
    
    # Concatenate all features
    real_features = np.concatenate(real_features_list[:num_samples], axis=0)[:num_samples]
    fake_features = np.concatenate(fake_features_list[:num_samples], axis=0)[:num_samples]
    real_logits = np.concatenate(real_logits_list[:num_samples], axis=0)[:num_samples]
    fake_logits = np.concatenate(fake_logits_list[:num_samples], axis=0)[:num_samples]
    
    # Calculate statistics
    real_mu = np.mean(real_features, axis=0)
    real_sigma = np.cov(real_features, rowvar=False)
    fake_mu = np.mean(fake_features, axis=0)
    fake_sigma = np.cov(fake_features, rowvar=False)
    
    # Calculate scores
    fid_value = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
    
    kid_value = polynomial_mmd_averages(
        real_features, fake_features,
        n_subsets=100, subset_size=min(1000, num_samples//2),
        degree=3, gamma=None, coef0=1, ret_var=False
    ).mean() * 100
    
    print(f"FID Score ({num_samples} samples): {fid_value:.4f}")
    print(f"KID Score ({num_samples} samples): {kid_value:.4f}")
    
    return fid_value, kid_value


# ==================== UTILITY FUNCTIONS ====================
def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    try:
        seconds = float(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "00:00:00"


def safe_division(numerator, denominator, default=0.0):
    """Safe division"""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return default


# ==================== CHECKPOINT MANAGEMENT ====================
def safe_torch_load(file_path):
    """Safe torch.load for different PyTorch versions"""
    try:
        return torch.load(file_path, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError) as e:
        print(f"weights_only=True failed: {e}")
        print("Trying with weights_only=False...")
        try:
            return torch.load(file_path, weights_only=False)
        except Exception as e2:
            print(f"weights_only=False also failed: {e2}")
            return torch.load(file_path, map_location='cpu', weights_only=False)


def save_checkpoint(model, optimizer, epoch, model_path, best_fid=float('inf'), 
                   best_kid=float('inf'), is_best_fid=False, is_best_kid=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_fid': best_fid,
        'best_kid': best_kid,
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'torch_cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }
    
    os.makedirs(model_path, exist_ok=True)
    torch.save(checkpoint, os.path.join(model_path, 'model.pth'))
    
    if is_best_fid:
        torch.save(checkpoint, os.path.join(model_path, 'best_fid_model.pth'))
    
    if is_best_kid:
        torch.save(checkpoint, os.path.join(model_path, 'best_kid_model.pth'))
    
    if epoch % SAVE_MODEL_HISTORY == 0:
        torch.save(checkpoint, os.path.join(model_path, f'model{epoch}.pth'))


def load_checkpoint(model, optimizer, model_path):
    """Load model checkpoint"""
    checkpoint_path = os.path.join(model_path, 'model.pth')
    if os.path.isfile(checkpoint_path):
        try:
            checkpoint = safe_torch_load(checkpoint_path)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0, float('inf'), float('inf')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                print("Warning: Could not load optimizer state")
        
        # Restore random states
        if 'random_state' in checkpoint:
            random.setstate(checkpoint['random_state'])
        if 'numpy_random_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_random_state'])
        if 'torch_random_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_random_state'])
        if 'torch_cuda_random_state' in checkpoint and checkpoint['torch_cuda_random_state'] is not None:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_random_state'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_fid = checkpoint.get('best_fid', float('inf'))
        best_kid = checkpoint.get('best_kid', float('inf'))
        
        print(f"Checkpoint loaded from epoch {start_epoch}")
        print(f"Previous best FID: {best_fid:.4f}, best KID: {best_kid:.4f}")
        
        return start_epoch, best_fid, best_kid
    else:
        print("No checkpoint found, starting from scratch")
        return 0, float('inf'), float('inf')


# ==================== MAIN TRAINING FUNCTION ====================
def main():
    set_seed(42)
    
    #wandb.init(project="WriteVit", name='WriteVit_DWConv_TEST_RUN', resume="allow")
    init_project()

    # Create datasets and dataloaders
    TextDatasetObj = TextDataset(num_examples=NUM_EXAMPLES)
    dataset = torch.utils.data.DataLoader(
        TextDatasetObj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True, 
        drop_last=True,
        collate_fn=TextDatasetObj.collate_fn
    )

    TextDatasetObjval = TextDatasetval(num_examples=NUM_EXAMPLES)
    datasetval = torch.utils.data.DataLoader(
        TextDatasetObjval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True, 
        drop_last=True,
        collate_fn=TextDatasetObjval.collate_fn
    )

    # Initialize model
    model = WriteViT().to(DEVICE)

    # Create save directory
    MODEL_PATH = os.path.join('saved_models', EXP_NAME)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Training parameters
    best_fid = float('inf')
    best_kid = float('inf')
    start_epoch = 0
    
    # FID/KID evaluation settings
    FID_EVAL_INTERVAL = 5  # Evaluate every epoch
    FID_NUM_SAMPLES = 500     # Number of samples for evaluation
    
    # Resume from checkpoint if available
    if RESUME and os.path.isdir(MODEL_PATH): 
        start_epoch, best_fid, best_kid = load_checkpoint(model, model.optimizer_G, MODEL_PATH)
    else:
        if not os.path.isdir(MODEL_PATH): 
            os.makedirs(MODEL_PATH, exist_ok=True)

    # Training progress
    epoch_pbar = tqdm(range(start_epoch, EPOCHS), desc="Training Progress", position=0)
    total_start_time = time.time()
    epoch_times = []
    
    for epoch in epoch_pbar:    
        epoch_start_time = time.time()
        
        # Training loop
        batch_pbar = tqdm(enumerate(dataset), total=len(dataset), 
                         desc=f"Epoch {epoch}/{EPOCHS}", position=1, leave=False)
        
        epoch_losses = {k: 0 for k in ['G', 'D', 'Dfake', 'Dreal', 'OCR_fake', 'OCR_real', 
                                       'w_fake', 'w_real', 'cycle1', 'cycle2', 'lda1', 'lda2',
                                       'KLD', 'patch_real', 'patch_fake', 'patch']}
        batch_count = 0
        
        for i, data in batch_pbar:
            # Training steps
            if (i % NUM_CRITIC_GOCR_TRAIN) == 0:
                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DOCR_TRAIN) == 0:
                model._set_input(data)
                model.optimize_D_OCR_W()
                model.optimize_D_OCR_W_step()
            
            # Update losses
            current_losses = model.get_current_losses()
            for key in epoch_losses.keys():
                if key in current_losses:
                    epoch_losses[key] += current_losses[key]
            batch_count += 1
            
            # Update progress bar
            current_loss_str = " | ".join([
                f"{k}: {safe_division(v, batch_count):.4f}" 
                for k, v in epoch_losses.items() 
                if v > 0 and k in ['G', 'D', 'OCR_fake', 'OCR_real']
            ])
            batch_pbar.set_postfix_str(current_loss_str)
        
        batch_pbar.close()
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Time estimation
        avg_epoch_time = np.mean(epoch_times) if epoch_times else epoch_time
        remaining_epochs = EPOCHS - epoch - 1
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        elapsed_time = time.time() - total_start_time
        
        # Average losses
        avg_losses = {k: safe_division(v, max(batch_count, 1)) for k, v in epoch_losses.items()}
        
        # Validation generation
        try:
            data_val = next(iter(datasetval))
            page_val = model._generate_page(
                data_val['img'].to(DEVICE), 
                data_val['simg'].to(DEVICE), 
                data_val['wcl'].to(DEVICE),
                data_val['swids'].to(DEVICE)
            )
        except Exception as e:
            print(f"Error in validation generation: {e}")
            page_val = torch.zeros(1, 1, 64, 64)

        # FID/KID evaluation
        fid_score = float('inf')
        kid_score = float('inf')
        is_best_fid = False
        is_best_kid = False
        
        if (epoch % FID_EVAL_INTERVAL == 0) or (epoch == EPOCHS - 1):
            print("\n" + "="*50)
            print(f"Calculating FID & KID at epoch {epoch}...")
            print("="*50)
            
            try:
                fid_score, kid_score = calculate_fid_and_kid_for_writevit(
                    model, datasetval, num_samples=FID_NUM_SAMPLES
                )
                
                if fid_score < best_fid:
                    best_fid = fid_score
                    is_best_fid = True
                    print(f"🏆 New best FID: {fid_score:.4f}")
                
                if kid_score < best_kid:
                    best_kid = kid_score
                    is_best_kid = True
                    print(f"🏆 New best KID: {kid_score:.4f}")
                    
            except Exception as e:
                print(f"❌ Error calculating FID/KID: {e}")
                import traceback
                traceback.print_exc()
                fid_score, kid_score = float('inf'), float('inf')
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=model.optimizer_G,
            epoch=epoch,
            model_path=MODEL_PATH,
            best_fid=best_fid,
            best_kid=best_kid,
            is_best_fid=is_best_fid,
            is_best_kid=is_best_kid
        )
        
        # WandB logging
        log_data = {
            **{f'loss-{k}': v for k, v in avg_losses.items()},
            'fid_score': fid_score,
            'kid_score': kid_score,
            'best_fid': best_fid,
            'best_kid': best_kid,
            "result": [wandb.Image(page_val * 255, caption="page_val")],
        }
        
        wandb.log({k: v for k, v in log_data.items() 
                  if v != 0 or 'score' in k or k == 'result'})
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'G': f"{avg_losses['G']:.4f}",
            'D': f"{avg_losses['D']:.4f}", 
            'FID': f"{fid_score:.2f}" if fid_score != float('inf') else 'N/A',
            'KID': f"{kid_score:.4f}" if kid_score != float('inf') else 'N/A',
            'Best_FID': f"{best_fid:.2f}" if best_fid != float('inf') else 'N/A',
            'Best_KID': f"{best_kid:.4f}" if best_kid != float('inf') else 'N/A',
            'Time': f"{epoch_time:.1f}s",
            'ETA': format_time(estimated_remaining_time)
        })
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"⏱️  Time: {epoch_time:.2f}s | Elapsed: {format_time(elapsed_time)} | ETA: {format_time(estimated_remaining_time)}")
        
        important_losses = ['G', 'D', 'OCR_fake', 'OCR_real', 'w_fake', 'w_real']
        loss_str = " | ".join([f"{k}: {avg_losses[k]:.4f}" for k in important_losses if k in avg_losses])
        print(f"📈 Losses: {loss_str}")
        print(f"🎯 FID: {fid_score:.4f}" if fid_score != float('inf') else "🎯 FID: N/A")
        print(f"🎯 KID: {kid_score:.4f}" if kid_score != float('inf') else "🎯 KID: N/A")
        print(f"🏆 Best FID: {best_fid:.4f}" if best_fid != float('inf') else "🏆 Best FID: N/A")
        print(f"🏆 Best KID: {best_kid:.4f}" if best_kid != float('inf') else "🏆 Best KID: N/A")
        print("-" * 80)

    epoch_pbar.close()
    
    # Training completed
    total_training_time = time.time() - total_start_time
    print("\n" + "🎉" * 30)
    print("TRAINING COMPLETED!")
    print(f"Total time: {format_time(total_training_time)}")
    print(f"Best FID: {best_fid:.4f}")
    print(f"Best KID: {best_kid:.4f}")
    print(f"Models saved at: {MODEL_PATH}")
    print("🎉" * 30)


if __name__ == "__main__":
    main()
