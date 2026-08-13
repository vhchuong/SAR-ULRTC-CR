import os
# Removed unused import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from util import pytorch_ssim
from main_net import SAR_ULRTC_CR_SMILECR as net
# from dataLoader import SEN12MSCR_0  # Removed unused import SIM_MSCR
from smilecr_dataset import SMILECRNumpyDataset_1
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import logging
# Removed unused import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='CRTC Network Training/Inference')
    parser.add_argument(
        '--ckpt_dir',
        type=Path,
        default=Path('../ckpts_SMILECR/'),
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--N_iter',
        type=int,
        default=10,
        help='Number of unrolled iterations'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='Set GPU ID'
    )
    parser.add_argument(
        '--save_folder',
        type=Path,
        default=Path('SMILECR_CRTC'),
        help='Path for saving test output images'
    )
    parser.add_argument(
        '--fig_mode',
        type=str,
        default='all',
        help='Mode for showing or saving figures: all, none',
        choices=['all', 'none']
    )
    parser.add_argument(
        '--test_mode',
        type=str,
        default='single',
        help='Testing mode: all or single',
        choices=['all', 'single']
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='Epoch number for single checkpoint testing'
    )
    parser.add_argument('--data_path', default= '../dataset/SMILE-CR-npy', type=str, help='Path to training data.')
    parser.add_argument(
        '--skip_epochs',
        type=int,
        default=0,
        help='Number of epochs to skip for testing'
    )
    args = parser.parse_args()
    
    # Validate checkpoint directory
    if not args.ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.ckpt_dir}")
        
    return args

def load_pretrained(path: Path, N_iter: int) -> torch.nn.Module:
    """
    Load a pretrained model from a checkpoint file.
    
    Args:
        path (Path): Path to the checkpoint file
        N_iter (int): Number of unrolled iterations
    
    Returns:
        torch.nn.Module: Loaded model
    """
    model = net(N_iter=N_iter)
    model = model.cuda()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {path}")
    return model

def compute_metric(real_B, fake_B):
    rmse = torch.sqrt(torch.mean(torch.square(real_B - fake_B)))
    psnr = 20 * torch.log10(1 / rmse)
    mae = torch.mean(torch.abs(real_B - fake_B))
    
    # spectral angle mapper
    mat = real_B * fake_B
    mat = torch.sum(mat, 1)
    mat = torch.div(mat, torch.sqrt(torch.sum(real_B * real_B, 1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(fake_B * fake_B, 1)))
    sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1))*torch.tensor(180)/torch.pi)
    
    ssim = pytorch_ssim.ssim(real_B, fake_B)

    return {'RMSE': rmse.cpu().detach().numpy().item(),
            'MAE': mae.cpu().detach().numpy().item(),
            'PSNR': psnr.cpu().detach().numpy().item(),
            'SAM': sam.cpu().detach().numpy().item(),
            'SSIM': ssim.cpu().detach().numpy().item()}

def compute_metric_channelwise(real_B, fake_B):
    """
    Compute metrics channel-wise and take the average across channels.
    """
    n_channels = real_B.shape[1]
    rmse_list, mae_list, psnr_list, sam_list, ssim_list = [], [], [], [], []

    for ch in range(n_channels):
        real_ch = real_B[:, ch:ch+1, :, :]
        fake_ch = fake_B[:, ch:ch+1, :, :]

        rmse = torch.sqrt(torch.mean((real_ch - fake_ch) ** 2))
        psnr = 20 * torch.log10(1 / (rmse + 1e-8))
        mae = torch.mean(torch.abs(real_ch - fake_ch))

        # spectral angle mapper for single channel is always zero, so skip for SAM
        # Instead, compute SAM only if n_channels > 1
        if n_channels > 1:
            mat = real_B * fake_B
            mat = torch.sum(mat, 1)
            mat = torch.div(mat, torch.sqrt(torch.sum(real_B * real_B, 1)) + 1e-8)
            mat = torch.div(mat, torch.sqrt(torch.sum(fake_B * fake_B, 1)) + 1e-8)
            sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1)) * torch.tensor(180.0) / torch.pi)
        else:
            sam = torch.tensor(0.0)

        ssim = pytorch_ssim.ssim(real_ch, fake_ch)

        rmse_list.append(rmse.item())
        mae_list.append(mae.item())
        psnr_list.append(psnr.item())
        sam_list.append(sam.item())
        ssim_list.append(ssim.item())

    return {
        'RMSE': float(np.mean(rmse_list)),
        'MAE': float(np.mean(mae_list)),
        'PSNR': float(np.mean(psnr_list)),
        'SAM': float(np.mean(sam_list)),
        'SSIM': float(np.mean(ssim_list))
    }

def test(opt_args, ckpt_path, epoch_num):

    torch.backends.cudnn.benchmark = True

    data_path = opt_args.data_path
    sen12mscr_test       = SMILECRNumpyDataset_1(data_path, split='test', return_filename= True)
    data_test_loader      = torch.utils.data.DataLoader(sen12mscr_test, batch_size=1, shuffle=False, num_workers=4)
    
    img_dir = f'./results_SMILECR/{opt_args.save_folder}_{opt_args.fig_mode}'

    os.makedirs(img_dir, exist_ok=True)
    if opt_args.fig_mode != 'none':
        # Delete all files in img_dir
        file_list = glob.glob(img_dir + '/*')
        for file in file_list:
            os.remove(file)

    print('fig_mode:', opt_args.fig_mode)

    num_samples = len(data_test_loader)
    metrics_names = ['RMSE', 'MAE', 'PSNR', 'SAM', 'SSIM']
    metrics = np.zeros((num_samples, len(metrics_names)))

    with torch.cuda.device(opt_args.gpu):
        with torch.no_grad():
            model = load_pretrained(path=ckpt_path, N_iter=opt_args.N_iter)
            model.eval()

            for i, (samples) in enumerate(tqdm(data_test_loader)):

                data = samples['input'].to(torch.float32)
                sar = samples['sar'].to(torch.float32)
                target = samples['target'].to(torch.float32)
                trans = samples['trans'].to(torch.float32)

                data, target, sar, trans = data.cuda(), target.cuda(), sar.cuda(), trans.cuda()
                
                img_path = os.path.join(img_dir, samples['filename'][0].replace('.npy', '.png'))

                layers = model(data, sar, trans)
                
                if opt_args.fig_mode == 'all':
                    fig, ax = plt.subplots(1, 6, figsize=(21, 5))
                    rgb_input = (data[0, [2,1,0], :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    rgb_input = (rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())
                    rgb_input = (rgb_input * 255).astype(np.uint8)

                    sar_img = (samples['sar'][0][0].cpu().detach().numpy() * 255).astype(np.uint8)
                    sar_img = np.clip(sar_img, 0, 255)

                    rgb_target = (target[0, [2,1,0], :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    rgb_target = (rgb_target - rgb_target.min()) / (rgb_target.max() - rgb_target.min())
                    rgb_target = (rgb_target * 255).astype(np.uint8)
                    rgb_target = np.clip(rgb_target, 0, 255)

                    rgb_img = (layers[-1][2][0, [2,1,0], :, :].cpu().detach().numpy().transpose(1, 2, 0))
                    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                    rgb_img = (rgb_img * 255).astype(np.uint8)
                    rgb_img = np.clip(rgb_img, 0, 255)

                    plt.imsave(img_path+'_pred.png', rgb_img, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path+'_gt.png', rgb_target, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path+'_input.png', rgb_input, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path+'_sar.png', sar_img, dpi=300, cmap='viridis')

                elif opt_args.fig_mode == 'none':
                    pass
                else:
                    raise ValueError('Invalid figure mode')

                if opt_args.fig_mode != 'none':
                    plt.tight_layout()
                    plt.savefig(img_path)
                    plt.close(fig)
                
                # Compute metrics

                X = layers[-1][2]
                output = compute_metric_channelwise(target, X)
                metrics[i] = [output[name] for name in metrics_names]
            
            average_metrics = np.round(np.mean(metrics, axis=0), 4)
            average_metrics_dict = {name: average_metrics[idx] for idx, name in enumerate(metrics_names)}

            print(f'Average Metrics for {ckpt_path}:', average_metrics_dict)

            return average_metrics_dict

if __name__ == '__main__':
    opt_args = parse_args()
    if opt_args.test_mode == 'all':
        best_metrics = None
        best_ckpt = None
        best_SAM_metrics = None
        best_SAM_ckpt = None
        best_SSIM_metrics = None
        best_SSIM_ckpt = None
        for ckpt_path in opt_args.ckpt_dir.glob('*.pth'):
            # Extract epoch number from filename
            epoch_num = int(ckpt_path.stem.split('_')[1])
            if epoch_num < opt_args.skip_epochs:  # Skip first 100 epochs
                continue
            metrics = test(opt_args, ckpt_path, epoch_num)
            if best_metrics is None or metrics['PSNR'] > best_metrics['PSNR']:
                best_metrics = metrics 
                best_ckpt = ckpt_path
            if best_SAM_metrics is None or metrics['SAM'] < best_SAM_metrics['SAM']:
                best_SAM_metrics = metrics
                best_SAM_ckpt = ckpt_path
            if best_SSIM_metrics is None or metrics['SSIM'] > best_SSIM_metrics['SSIM']:
                best_SSIM_metrics = metrics
                best_SSIM_ckpt = ckpt_path
        print(f'Best checkpoint: {best_ckpt}')
        print(f'Best metrics: {best_metrics}')
        print(f'Best SAM checkpoint: {best_SAM_ckpt}')
        print(f'Best SAM metrics: {best_SAM_metrics}')
        print(f'Best SSIM checkpoint: {best_SSIM_ckpt}')
        print(f'Best SSIM metrics: {best_SSIM_metrics}')
    elif opt_args.test_mode == 'single':
        if opt_args.epoch is None:
            raise ValueError('Epoch number must be specified for single checkpoint testing mode')
        ckpt_path = opt_args.ckpt_dir / f'epoch_{opt_args.epoch}.pth'
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')
        metrics = test(opt_args, ckpt_path, opt_args.epoch)
    else:
        raise ValueError('Invalid test mode')
