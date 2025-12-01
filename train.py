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
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["WANDB_API_KEY"] = "652d515cee7db8d8467a876756b9b097b9a342b1"

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
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed} for reproducibility")

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

def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    """
    Compute the polynomial kernel between X and Y
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    K = (gamma * np.dot(X, Y.T) + coef0) ** degree
    return K

def kernel_inception_distance(real_features, fake_features, max_subset_size=1000, degree=3, gamma=None, coef0=1, ret_subsets=False):
    """
    Compute the Kernel Inception Distance (KID) between real and fake features
    """
    real_features = np.array(real_features)
    fake_features = np.array(fake_features)
    
    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    
    if n_real == 0 or n_fake == 0:
        return float('inf')
    
    # Use minimum of the two for subset size
    subset_size = min(n_real, n_fake, max_subset_size)
    
    # Randomly select subsets
    np.random.seed(42)  # For reproducibility
    real_subset = real_features[np.random.choice(n_real, subset_size, replace=False)]
    fake_subset = fake_features[np.random.choice(n_fake, subset_size, replace=False)]
    
    # Compute kernel matrices
    K_rr = polynomial_kernel(real_subset, real_subset, degree, gamma, coef0)
    K_ff = polynomial_kernel(fake_subset, fake_subset, degree, gamma, coef0)
    K_rf = polynomial_kernel(real_subset, fake_subset, degree, gamma, coef0)
    
    # Compute KID
    kid = K_rr.mean() + K_ff.mean() - 2 * K_rf.mean()
    
    if ret_subsets:
        return kid, real_subset, fake_subset
    return kid

def extract_inception_features(inception_model, img_tensor):
    """Trích xuất features từ InceptionV3 cho 1 ảnh"""
    # Tiền xử lý ảnh
    processed_img = preprocess_for_inception(img_tensor)
    
    # Chuyển qua GPU và extract features
    features = inception_model(processed_img.cuda())[0]
    
    # Global average pooling và flatten
    features = F.adaptive_avg_pool2d(features, (1, 1))
    features = features.view(features.size(0), -1)
    
    return features.cpu()

def preprocess_for_inception(img_tensor):
    """Tiền xử lý ảnh cho InceptionV3"""
    # Chuẩn hóa về [0, 1] nếu cần
    if img_tensor.min() < 0:  # Nếu trong range [-1, 1]
        img_tensor = (img_tensor + 1) / 2
    elif img_tensor.max() > 1:  # Nếu trong range [0, 255]
        img_tensor = img_tensor / 255.0
    
    # Resize lên 299x299 (bắt buộc cho InceptionV3)
    if img_tensor.shape[-2:] != (299, 299):
        # Thêm batch dimension nếu cần
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = F.interpolate(img_tensor, size=(299, 299), 
                                 mode='bilinear', align_corners=False)
    
    # Nhân bản thành 3 kênh nếu là ảnh xám
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
    
    # Chuẩn hóa về range [-1, 1] cho InceptionV3
    img_tensor = img_tensor * 2 - 1
    
    return img_tensor

def calculate_fid_and_kid_for_writevit(model, dataloader, num_samples=500):
    """Tính FID score và KID score cho WriteViT model"""
    
    # Khởi tạo inception model
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], 
                           use_fid_inception=True).cuda()
    inception.eval()
    
    real_features = []
    fake_features = []
    
    print(f"Calculating FID & KID with target {num_samples} samples each...")
    
    # Sử dụng tqdm cho progress bar
    pbar = tqdm(total=num_samples * 2, desc="Collecting features")
    
    with torch.no_grad():
        real_count = 0
        fake_count = 0
        
        for data in dataloader:
            # Kiểm tra nếu đã đủ samples
            if real_count >= num_samples and fake_count >= num_samples:
                break
            
            batch_size = data['img'].shape[0]
            
            # Xử lý ảnh REAL từ batch hiện tại
            if real_count < num_samples:
                real_imgs = data['img']  # Shape: [B, 1, 32, width]
                
                for i in range(batch_size):
                    if real_count >= num_samples:
                        break
                    
                    try:
                        # Lấy ảnh individual và chuyển đổi kích thước
                        img_tensor = real_imgs[i].unsqueeze(0)  # [1, 1, 32, width]
                        features = extract_inception_features(inception, img_tensor)
                        real_features.append(features.numpy())
                        real_count += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing real image {real_count}: {e}")
                        continue
            
            # Generate và xử lý ảnh FAKE
            if fake_count < num_samples:
                try:
                    # Sử dụng phương thức từ code tham khảo để tạo fake images
                    sdata = data['img'].to(DEVICE)
                    
                    # Tạo text ngẫu nhiên từ lex như trong code tham khảo
                    sample_lex_idx = model.fake_y_dist.sample([batch_size])
                    fake_y = [model.lex[i].encode("utf-8") for i in sample_lex_idx]
                    
                    # Encode text
                    text_encode_fake, len_text_fake = model.netconverter.encode(fake_y)
                    text_encode_fake = text_encode_fake.to(DEVICE)
                    
                    # Tạo feature writer và fake images
                    feat_w, _ = model.netW(sdata.detach(), data['wcl'].to(DEVICE))
                    fakes = model.netG(feat_w, text_encode_fake)
                    
                    # Xử lý fakes dựa trên kiểu dữ liệu
                    if isinstance(fakes, list):
                        # Nếu là list, xử lý từng phần tử
                        for fake_batch in fakes:
                            if fake_count >= num_samples:
                                break
                            fake_batch_detached = fake_batch.detach().cpu()
                            
                            # Xử lý từng ảnh trong batch
                            for i in range(fake_batch_detached.shape[0]):
                                if fake_count >= num_samples:
                                    break
                                img_tensor = fake_batch_detached[i].unsqueeze(0)  # [1, 32, width]
                                # Thêm channel dimension nếu cần
                                if img_tensor.dim() == 3:
                                    img_tensor = img_tensor.unsqueeze(1)  # [1, 1, 32, width]
                                features = extract_inception_features(inception, img_tensor)
                                fake_features.append(features.numpy())
                                fake_count += 1
                                pbar.update(1)
                    else:
                        # Nếu là tensor trực tiếp
                        fake_combined = fakes.detach().cpu()
                        
                        # Xử lý dựa trên số chiều của tensor
                        if fake_combined.dim() == 3:  # [B, 32, width]
                            for i in range(fake_combined.shape[0]):
                                if fake_count >= num_samples:
                                    break
                                img_tensor = fake_combined[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 32, width]
                                features = extract_inception_features(inception, img_tensor)
                                fake_features.append(features.numpy())
                                fake_count += 1
                                pbar.update(1)
                        elif fake_combined.dim() == 4:  # [B, 1, 32, width] hoặc [B, C, 32, width]
                            for i in range(fake_combined.shape[0]):
                                if fake_count >= num_samples:
                                    break
                                # Lấy ảnh đầu tiên nếu có nhiều channel
                                if fake_combined.shape[1] > 1:
                                    img_tensor = fake_combined[i, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, 32, width]
                                else:
                                    img_tensor = fake_combined[i].unsqueeze(0)  # [1, 1, 32, width]
                                features = extract_inception_features(inception, img_tensor)
                                fake_features.append(features.numpy())
                                fake_count += 1
                                pbar.update(1)
                                
                except Exception as e:
                    print(f"Error in fake generation: {e}")
                    continue
            
            pbar.set_description(f"Real: {real_count}/{num_samples}, Fake: {fake_count}/{num_samples}")
    
    pbar.close()
    
    # Cân bằng số lượng samples
    final_count = min(len(real_features), len(fake_features))
    
    if final_count < 100:
        print(f"ERROR: Not enough samples for FID/KID (need 100+, got {final_count})")
        return float('inf'), float('inf')
    
    print(f"Final balanced calculation: {final_count} samples each")
    
    # Tính FID và KID từ features
    try:
        # Chuyển danh sách features thành numpy arrays
        real_features_np = np.vstack(real_features[:final_count])
        fake_features_np = np.vstack(fake_features[:final_count])
        
        # Tính FID score
        mu_real = np.mean(real_features_np, axis=0)
        sigma_real = np.cov(real_features_np, rowvar=False)
        
        mu_fake = np.mean(fake_features_np, axis=0)
        sigma_fake = np.cov(fake_features_np, rowvar=False)
        
        fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        print(f"FID Score ({final_count} samples): {fid_value:.4f}")
        
        # Tính KID score
        kid_value = kernel_inception_distance(real_features_np, fake_features_np)
        print(f"KID Score ({final_count} samples): {kid_value:.4f}")
        
        return fid_value, kid_value
        
    except Exception as e:
        print(f"FID/KID calculation error: {e}")
        return float('inf'), float('inf')

class InceptionV3(torch.nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling features
        768: 2,  # Pre-aux classifier features
        2048: 3, # Final average pooling features
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

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = torch.nn.ModuleList()

        inception = inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(torch.nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                torch.nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(torch.nn.Sequential(*block1))

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
            self.blocks.append(torch.nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(torch.nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps"""
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = x * 2 - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

def format_time(seconds):
    """Định dạng thời gian từ seconds sang HH:MM:SS"""
    try:
        # Đảm bảo seconds là số và chuyển thành integer
        seconds = float(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "00:00:00"

def calculate_initial_metrics_if_resumed(model, datasetval, model_path, resume):
    """Tính FID và KID ban đầu nếu resume từ checkpoint"""
    if resume and os.path.isfile(model_path + '/model.pth'):
        print("Resuming from checkpoint - calculating initial FID & KID...")
        try:
            # Tính FID và KID với số lượng samples nhỏ hơn để kiểm tra nhanh
            initial_fid, initial_kid = calculate_fid_and_kid_for_writevit(model, datasetval, num_samples=200)
            print(f"Initial FID after resuming: {initial_fid:.4f}")
            print(f"Initial KID after resuming: {initial_kid:.4f}")
            
            # Log initial metrics to wandb
            wandb.log({
                'initial_fid_after_resume': initial_fid,
                'initial_kid_after_resume': initial_kid
            })
            
            return initial_fid, initial_kid
        except Exception as e:
            print(f"Error calculating initial metrics: {e}")
            return float('inf'), float('inf')
    return float('inf'), float('inf')

def safe_division(numerator, denominator, default=0.0):
    """Thực hiện phép chia an toàn, tránh chia cho 0"""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return default

# ==================== CHECKPOINT MANAGEMENT ====================
def safe_torch_load(file_path):
    """Safe torch.load that handles both old and new PyTorch versions"""
    try:
        # First try with weights_only=True (PyTorch 2.6+ default)
        return torch.load(file_path, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError) as e:
        print(f"weights_only=True failed: {e}")
        print("Trying with weights_only=False (make sure you trust the source)...")
        try:
            # If that fails, try with weights_only=False
            return torch.load(file_path, weights_only=False)
        except Exception as e2:
            print(f"weights_only=False also failed: {e2}")
            # Last resort: try with map_location and other options
            return torch.load(file_path, map_location='cpu', weights_only=False)

def save_checkpoint(model, optimizer, epoch, model_path, best_fid=float('inf'), best_kid=float('inf'), is_best_fid=False, is_best_kid=False):
    """Save model checkpoint with additional training state"""
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
    
    # Save main checkpoint
    torch.save(checkpoint, model_path + '/model.pth')
    
    # Save best FID model
    if is_best_fid:
        torch.save(checkpoint, model_path + '/best_fid_model.pth')
    
    # Save best KID model  
    if is_best_kid:
        torch.save(checkpoint, model_path + '/best_kid_model.pth')
    
    # Save historical checkpoint
    if epoch % SAVE_MODEL_HISTORY == 0:
        torch.save(checkpoint, model_path + f'/model{epoch}.pth')

def load_checkpoint(model, optimizer, model_path):
    """Load model checkpoint and restore training state"""
    if os.path.isfile(model_path + '/model.pth'):
        try:
            # Try different loading strategies
            checkpoint = safe_torch_load(model_path + '/model.pth')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Trying to load only model weights...")
            try:
                # Last resort: load only model weights
                checkpoint = torch.load(model_path + '/model.pth', map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint)
                return 0, float('inf'), float('inf')
            except:
                print("Failed to load checkpoint, starting from scratch")
                return 0, float('inf'), float('inf')
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Only load optimizer state if it exists and matches current optimizer
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                print("Warning: Could not load optimizer state, continuing with fresh optimizer")
        
        # Restore random states for reproducibility
        if 'random_state' in checkpoint:
            try:
                random.setstate(checkpoint['random_state'])
            except:
                print("Warning: Could not restore random state")
        
        if 'numpy_random_state' in checkpoint:
            try:
                np.random.set_state(checkpoint['numpy_random_state'])
            except:
                print("Warning: Could not restore numpy random state")
                
        if 'torch_random_state' in checkpoint:
            try:
                torch.set_rng_state(checkpoint['torch_random_state'])
            except:
                print("Warning: Could not restore torch random state")
                
        if 'torch_cuda_random_state' in checkpoint and checkpoint['torch_cuda_random_state'] is not None:
            try:
                torch.cuda.set_rng_state(checkpoint['torch_cuda_random_state'])
            except:
                print("Warning: Could not restore torch cuda random state")
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_fid = checkpoint.get('best_fid', float('inf'))
        best_kid = checkpoint.get('best_kid', float('inf'))
        
        print(f"Checkpoint loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Previous best FID: {best_fid:.4f}, best KID: {best_kid:.4f}")
        
        return start_epoch, best_fid, best_kid
    else:
        print("No checkpoint found, starting from scratch")
        return 0, float('inf'), float('inf')

def main():
    # Set seed for reproducibility at the very beginning
    set_seed(42)
    
    #wandb.init(project="WriteVit", name='WriteVit_SWINatt_TEST_RUN', resume="allow")

    init_project()

    # Tạo datasets và dataloaders
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

    # Khởi tạo model
    model = WriteViT().to(DEVICE)

    # Tạo thư mục lưu model
    os.makedirs('saved_models', exist_ok=True)
    MODEL_PATH = os.path.join('saved_models', EXP_NAME)
    
    # Thêm FID và KID evaluation vào training loop
    best_fid = float('inf')
    best_kid = float('inf')
    start_epoch = 0
    
    # Thêm vào params.py hoặc định nghĩa ở đây
    FID_EVAL_INTERVAL = 10  # Tính FID/KID mỗi 10 epochs
    FID_NUM_SAMPLES = 500     # Số samples dùng để tính FID/KID
    
    # Kiểm tra resume và load checkpoint
    if os.path.isdir(MODEL_PATH) and RESUME: 
        start_epoch, best_fid, best_kid = load_checkpoint(model, model.optimizer_G, MODEL_PATH)
        
        # Tính metrics ban đầu khi resume
        initial_fid, initial_kid = calculate_initial_metrics_if_resumed(model, datasetval, MODEL_PATH, RESUME)
        best_fid = min(best_fid, initial_fid)
        best_kid = min(best_kid, initial_kid)
        
    else: 
        if not os.path.isdir(MODEL_PATH): 
            os.makedirs(MODEL_PATH, exist_ok=True)

    # Training progress bar chính
    epoch_pbar = tqdm(range(start_epoch, EPOCHS), desc="Training Progress", position=0)
    
    # Biến để theo dõi thời gian và losses
    total_start_time = time.time()
    epoch_times = []
    
    for epoch in epoch_pbar:    
        epoch_start_time = time.time()
        
        # Progress bar cho từng batch trong epoch
        batch_pbar = tqdm(enumerate(dataset), total=len(dataset), 
                         desc=f"Epoch {epoch+1}/{EPOCHS}", position=1, leave=False)
        
        # Theo dõi losses trung bình trong epoch
        epoch_losses = {
            'G': 0, 'D': 0, 'Dfake': 0, 'Dreal': 0,
            'OCR_fake': 0, 'OCR_real': 0, 'w_fake': 0, 'w_real': 0,
            'cycle1': 0, 'cycle2': 0, 'lda1': 0, 'lda2': 0,
            'KLD': 0, 'patch_real': 0, 'patch_fake': 0, 'patch': 0
        }
        batch_count = 0
        
        for i, data in batch_pbar:
            # Training steps - dựa trên code WriteViT
            if (i % NUM_CRITIC_GOCR_TRAIN) == 0:
                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DOCR_TRAIN) == 0:
                model._set_input(data)
                model.optimize_D_OCR_W()
                model.optimize_D_OCR_W_step()
            
            # Cập nhật losses trung bình
            current_losses = model.get_current_losses()
            for key in epoch_losses.keys():
                if key in current_losses:
                    epoch_losses[key] += current_losses[key]
            batch_count += 1
            
            # Cập nhật progress bar với thông tin chi tiết
            current_loss_str = " | ".join([
                f"{k}: {safe_division(v, batch_count):.4f}" 
                for k, v in epoch_losses.items() 
                if v > 0 and k in ['G', 'D', 'OCR_fake', 'OCR_real']
            ])
            batch_pbar.set_postfix_str(current_loss_str)
        
        batch_pbar.close()
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Tính thời gian trung bình và ước lượng thời gian còn lại
        avg_epoch_time = np.mean(epoch_times) if epoch_times else epoch_time
        remaining_epochs = EPOCHS - epoch - 1
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        elapsed_time = time.time() - total_start_time
        
        # Tính losses trung bình cho epoch
        avg_losses = {k: safe_division(v, max(batch_count, 1)) for k, v in epoch_losses.items()}
        
        # Validation và generation
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
            page_val = torch.zeros(1, 1, 64, 64)  # Fallback

        # Tính FID và KID mỗi FID_EVAL_INTERVAL epochs
        fid_score = float('inf')
        kid_score = float('inf')
        is_best_fid = False
        is_best_kid = False
        
        if (epoch % FID_EVAL_INTERVAL == 0) or (epoch == EPOCHS - 1):
            print("\nCalculating FID & KID scores...")
            try:
                fid_score, kid_score = calculate_fid_and_kid_for_writevit(model, datasetval, num_samples=FID_NUM_SAMPLES)
                print(f"FID Score at epoch {epoch}: {fid_score:.4f}")
                print(f"KID Score at epoch {epoch}: {kid_score:.4f}")
                
                # Kiểm tra xem có phải là model tốt nhất không
                if fid_score < best_fid:
                    best_fid = fid_score
                    is_best_fid = True
                    print(f"New best FID model: {fid_score:.4f}")
                
                if kid_score < best_kid:
                    best_kid = kid_score
                    is_best_kid = True
                    print(f"New best KID model: {kid_score:.4f}")
                    
            except Exception as e:
                print(f"Error calculating FID/KID: {e}")
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
        
        # wandb logging với FID, KID và tất cả losses
        log_data = {
            'loss-G': avg_losses['G'],
            'loss-D': avg_losses['D'], 
            'loss-Dfake': avg_losses['Dfake'],
            'loss-Dreal': avg_losses['Dreal'],
            'loss-OCR_fake': avg_losses['OCR_fake'],
            'loss-OCR_real': avg_losses['OCR_real'],
            'loss-w_fake': avg_losses['w_fake'],
            'loss-w_real': avg_losses['w_real'],
            'loss-cycle1': avg_losses['cycle1'],
            'loss-cycle2': avg_losses['cycle2'],
            'loss-lda1': avg_losses['lda1'],
            'loss-lda2': avg_losses['lda2'],
            'loss-KLD': avg_losses['KLD'],
            'loss-patch_real': avg_losses['patch_real'],
            'loss-patch_fake': avg_losses['patch_fake'],
            'loss-patch': avg_losses['patch'],
            'fid_score': fid_score,
            'kid_score': kid_score,
            'best_fid': best_fid,
            'best_kid': best_kid,
            'epoch': epoch,
            'timeperepoch': epoch_time,
            "result": [wandb.Image(page_val*255, caption="page_val")],
        }
        
        # Chỉ log những loss có giá trị > 0
        wandb.log({k: v for k, v in log_data.items() if v != 0 or k in ['fid_score', 'kid_score', 'best_fid', 'best_kid', 'epoch', 'timeperepoch', 'result']})

        # Cập nhật main progress bar
        epoch_pbar.set_postfix({
            'G_loss': f"{avg_losses['G']:.4f}",
            'D_loss': f"{avg_losses['D']:.4f}", 
            'OCR_fake': f"{avg_losses['OCR_fake']:.4f}",
            'FID': f"{fid_score:.2f}" if fid_score != float('inf') else 'N/A',
            'KID': f"{kid_score:.4f}" if kid_score != float('inf') else 'N/A',
            'Best_FID': f"{best_fid:.2f}" if best_fid != float('inf') else 'N/A',
            'Best_KID': f"{best_kid:.4f}" if best_kid != float('inf') else 'N/A',
            'Epoch_time': f"{epoch_time:.1f}s",
            'ETA': format_time(estimated_remaining_time)
        })
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Time: {epoch_time:.2f}s | Elapsed: {format_time(elapsed_time)} | ETA: {format_time(estimated_remaining_time)}")
        
        # Print các losses quan trọng
        important_losses = ['G', 'D', 'OCR_fake', 'OCR_real', 'w_fake', 'w_real']
        loss_str = " | ".join([f"{k}: {avg_losses[k]:.4f}" for k in important_losses if k in avg_losses])
        print(f"Losses: {loss_str}")
        print(f"FID Score: {fid_score:.4f}" if fid_score != float('inf') else "FID Score: N/A")
        print(f"KID Score: {kid_score:.4f}" if kid_score != float('inf') else "KID Score: N/A")
        print(f"Best FID: {best_fid:.4f}" if best_fid != float('inf') else "Best FID: N/A")
        print(f"Best KID: {best_kid:.4f}" if best_kid != float('inf') else "Best KID: N/A")
        print("-" * 80)

        # Additional model saving for history (redundant with checkpoint but kept for compatibility)
        if epoch % SAVE_MODEL_HISTORY == 0: 
            torch.save(model.state_dict(), MODEL_PATH + '/model' + str(epoch) + '.pth')
            print(f"Model history saved at epoch {epoch}")
    
    epoch_pbar.close()
    
    # Training completed
    total_training_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total training time: {format_time(total_training_time)}")
    print(f"Best FID score: {best_fid:.4f}")
    print(f"Best KID score: {best_kid:.4f}")
    print(f"Final model saved at: {MODEL_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()