import os
# Removed unused import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from util import pytorch_ssim
from main_net import RPCA_Net_4_stages_SAR_Trans_RGB_9 as net
from dataLoader import SEN12MSCR_0
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from thop import profile
import time
from skimage.transform import resize
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
        default=Path('../ckpts_SEN12MSCR/'),
        help='Directory containing checkpoint files'
    )
    parser.add_argument(
        '--N_iter',
        type=int,
        default=10,
        help='Number of unrolled iterations'
        # choices=range(1, 101)  # Reasonable range for iterations
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
        default=Path('SEN12MSCR_CRTC_results'),
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
        default=200,
        help='Epoch number for single checkpoint testing'
    )
    parser.add_argument(
        '--data_mode',
        type=str,
        default='partial',
        help='Path to training data',
        choices=['partial', 'full']
    )
    parser.add_argument('--data_path', default= '../dataset/SEN12MS-CR_1', type=str, help='Path to training data.')
    parser.add_argument(
        '--skip_epochs',
        type=int,
        default=150,
        help='Number of epochs to skip for testing'
    )
    parser.add_argument('--params', action='store_true', help='If true, get the model parameters.')
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
    checkpoint = torch.load(path, weights_only=True)
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

def test(opt_args, ckpt_path, epoch_num):

    torch.backends.cudnn.benchmark = True

    data_path = opt_args.data_path
    sen12mscr_test       = SEN12MSCR_1(data_path, split='test', data_mode=opt_args.data_mode)
    data_test_loader      = torch.utils.data.DataLoader(sen12mscr_test, batch_size=1, shuffle=False, num_workers=4)
    
    img_dir = f'results_SEN12MSCR/{opt_args.save_folder}_{opt_args.fig_mode}'
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
            
            #Calculate FLOPs and params
            if opt_args.params:
                try:
                    dummy_input = torch.randn(1, 13, 64, 64).cuda()
                    dummy_sar = torch.randn(1, 2, 64, 64).cuda()
                    dummy_trans = torch.randn(1, 3, 64, 64).cuda()
                    flops, params = profile(model, inputs=(dummy_input, dummy_sar, dummy_trans), verbose=False)
                    logger.info(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
                    logger.info(f"Params: {params / 1e3:.2f} K")
                except ImportError:
                    logger.warning("THOP not installed, skipping FLOPs and Params calculation.")
            model.eval()
            total_time = 0.0
            for i, (samples) in enumerate(tqdm(data_test_loader)):

                data = samples['input']
                sar = samples['sar']
                target = samples['target']
                data = data.to(torch.float32)
                target = target.to(torch.float32)
                data, target, sar, trans = data.cuda(), target.cuda(), sar.cuda(), samples['trans'].cuda()

                # print(trans.shape)
                # break
                img_path = os.path.join(img_dir, samples['filename'][0].replace('.npy', '.png'))

                tic = time.time()
                layers = model(data, sar, trans)
                toc = time.time()
                elapsed_time = (toc - tic) * 1000  # convert to milliseconds
                total_time += elapsed_time

                if opt_args.fig_mode == 'all':
                    
                    rgb_input = (data[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    rgb_input = np.clip(rgb_input, 0, 255)
                    sar_img = (samples['sar'][0][0].cpu().detach().numpy() * 255).astype(np.uint8)
                    sar_img = np.clip(sar_img, 0, 255)
                    rgb_target = (target[0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255 * 3).astype(np.uint8)
                    rgb_target = np.clip(rgb_target, 0, 255)
                    rgb_output = (layers[-1][2][0, [3,2,1], :, :].cpu().detach().numpy().transpose(1, 2, 0) * 255 * 3).astype(np.uint8)
                    rgb_output = np.clip(rgb_output, 0, 255)
                    
                    plt.imsave(img_path.replace('.png', '_pred.png'), rgb_output, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path.replace('.png', '_gt.png'), rgb_target, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path.replace('.png', '_input.png'), rgb_input, dpi=300, vmin=0, vmax=255)
                    plt.imsave(img_path.replace('.png', '_sar.png'), sar_img, dpi=300, cmap='gray', vmin=0, vmax=255)
                    
                    fig, ax = plt.subplots(1, 4, figsize=(21, 5))
                    ax[0].imshow(rgb_input)
                    ax[0].set_title('Input Image')
                    ax[0].axis('off')
                    ax[1].imshow(sar_img, cmap='gray')
                    ax[1].set_title('SAR Image')
                    ax[1].axis('off')
                    ax[2].imshow(rgb_output)
                    ax[2].set_title('Output Image')
                    ax[2].axis('off')
                    ax[3].imshow(rgb_target)
                    ax[3].set_title('Groundtruth')
                    ax[3].axis('off')
                    plt.savefig(img_path)
                    plt.close(fig)

                elif opt_args.fig_mode == 'none':
                    pass
                else:
                    raise ValueError('Invalid figure mode')

                # Compute metrics
                metric = compute_metric(target, layers[-1][2])
                metrics[i] = [metric[name] for name in metrics_names]

            average_metrics = np.round(np.mean(metrics, axis=0), 4)
            average_metrics_dict = {name: average_metrics[idx] for idx, name in enumerate(metrics_names)}
            print(f'Average Metrics for {ckpt_path}:', average_metrics_dict)

            return average_metrics_dict

if __name__ == '__main__':
    opt_args = parse_args()
    if opt_args.epoch is None:
        raise ValueError('Epoch number must be specified for single checkpoint testing mode')
    
    ckpt_path = opt_args.ckpt_dir / f'epoch_{opt_args.epoch}.pth'
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')
    metrics = test(opt_args, ckpt_path, opt_args.epoch)

