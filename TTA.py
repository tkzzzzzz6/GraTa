import torch
import os
import torch.nn as nn
from networks.ResUnet import ResUnet
from config import *
from utils.metrics import calculate_metrics
import numpy as np
import argparse
import sys, datetime, time
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.transform import collate_fn_wo_transform
from custom_optimizers.grata import GraTa
from tqdm import tqdm
from utils.visualization import create_visualization_results


# Logger class for training records
class Logger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


torch.set_num_threads(1)


def print_information(config):
    print('=' * 50)
    print('TRAINING CONFIGURATION')
    print('=' * 50)
    print('Model Root: ', config.path_save_model)
    print('Device: ', config.device)
    print('CUDA Available: ', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU Count: ', torch.cuda.device_count())
        print('Current GPU: ', torch.cuda.current_device())
        print('GPU Name: ', torch.cuda.get_device_name(0))
        print('CUDA Version: ', torch.version.cuda)
    else:
        print('Using CPU for training')
    print('PyTorch Version: ', torch.__version__)
    print('Time: ', config.time_now)
    print('Source Domain: ', config.Source_Dataset)
    print('Target Domain: ', config.Target_Dataset)
    print('Model: ' + str(config.model_type))
    print('Input Size: ', config.image_size)
    print('Batch Size: ', config.batch_size)
    print('Optimizer: ', config.optimizer)
    print('Learning Rate: ', config.lr)
    print('Auxiliary Loss: ', config.aux_loss)
    print('Pseudo Loss: ', config.pse_loss)
    print('=' * 50)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params


class TrainTTA:
    def __init__(self, config):
        config.time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        self.load_model = os.path.join(config.path_save_model, str(config.Source_Dataset))  # Pretrained Source Model
        self.log_path = os.path.join(config.path_save_log, 'TrainTTA')  # Save Log

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.log_path = os.path.join(self.log_path, config.time_now + '.log')
        sys.stdout = Logger(self.log_path, sys.stdout)

        # Data Loading
        target_test_csv = []
        if config.Target_Dataset != 'REFUGE_Valid':
            target_test_csv.append(config.Target_Dataset + '_train.csv')
            target_test_csv.append(config.Target_Dataset + '_test.csv')
        else:
            target_test_csv.append(config.Target_Dataset + '.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size, img_normalize=True)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=collate_fn_wo_transform,
                                             num_workers=config.num_workers)

        # Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.image_size = config.image_size
        self.model_type = config.model_type

        # Optimizer
        self.optimizer = None
        self.optim = config.optimizer
        self.lr = config.lr
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # Training
        self.device = config.device
        self.aux = config.aux_loss
        self.pse = config.pse_loss

        print_information(config)
        self.build_model()
        self.print_network()

    def build_model(self):
        self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False).to(self.device)
        # 修复：添加 map_location 参数，支持 CPU 加载
        checkpoint = torch.load(self.load_model + '/' + 'last-' + self.model_type + '.pth', 
                               map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)

        para = collect_params(self.model)

        if self.optim == 'SGD':
            base_optimizer = torch.optim.SGD(
                para,
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
            )
        elif self.optim == 'Adam':
            base_optimizer = torch.optim.Adam(
                para,
                lr=self.lr,
                betas=self.betas,
            )
        elif self.optim == 'AdamW':
            base_optimizer = torch.optim.AdamW(
                para,
                lr=self.lr,
                betas=self.betas,
            )
        else:
            raise NotImplementedError("ERROR: no such optimizer {}!".format(self.optim))
        # 传递num_classes参数给GraTa优化器
        self.optimizer = GraTa(para, base_optimizer, self.model, device=self.device, 
                              num_classes=self.out_ch)

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self):
        metric_dict = ['Disc_Dice', 'Disc_ASSD', 'Cup_Dice', 'Cup_ASSD']

        metrics_test = [[], [], [], []]
        total_batches = len(self.target_test_loader)
        print(f"Starting TTA training on {total_batches} batches...")
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(self.target_test_loader, desc=f"TTA Training", 
                   total=total_batches, ncols=100, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch, data in enumerate(pbar):
            x, y = data['data'], data['mask']
            x = torch.from_numpy(x).to(dtype=torch.float32).to(self.device)
            y = torch.from_numpy(y).to(dtype=torch.float32).to(self.device)

            self.model.train()
            self.model.requires_grad_(False)
            for nm, m in self.model.named_modules():
                if self.aux in nm or self.pse in nm: m.requires_grad_(True)
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

            self.optimizer.base_optimizer.zero_grad()
            self.optimizer.step(data, self.aux, self.pse)

            with torch.no_grad():
                pred_logit, fea = self.model(x)

            seg_output = torch.sigmoid(pred_logit)
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]
            
            # 更新进度条描述，显示当前batch的指标
            if batch % 5 == 0:  # 每5个batch更新一次描述
                current_dice = np.mean([metrics_test[0][-1] if metrics_test[0] else 0, 
                                      metrics_test[2][-1] if metrics_test[2] else 0])
                pbar.set_postfix({
                    'Dice': f'{current_dice:.3f}',
                    'Batch': f'{batch+1}/{total_batches}'
                })

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics Mean: ", print_test_metric_mean)

        test_metrics_y = np.std(metrics_test, axis=1)
        print_test_metric_std = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_std[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics Std: ", print_test_metric_std)
        
        print(f"TTA training completed for {config.Target_Dataset}")
        print("=" * 50)
        return print_test_metric_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aux_loss', type=str, default='ent', 
                        help='consis/ent/recon/rotate/supres/denoise/pixel_cl/enhanced_pseudo')   # auxiliary loss
    parser.add_argument('--pse_loss', type=str, default='consis', 
                        help='consis/ent/recon/rotate/supres/denoise/pixel_cl/enhanced_pseudo')    # pseudo loss

    parser.add_argument('--Source_Dataset', type=str,
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
                        
    parser.add_argument('--optimizer', type=str, required=False, default='Adam', help='SGD/Adam/AdamW')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--momentum', type=float, required=False, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, required=False, default=0.9)  # beta1 in Adam
    parser.add_argument('--beta2', type=float, required=False, default=0.999)  # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, required=False, default=0.00)
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    
    parser.add_argument('--model_type', type=str, required=False, default='Res_Unet')
    parser.add_argument('--backbone', type=str, required=False, default='resnet34')

    parser.add_argument('--in_ch', type=int, required=False, default=3)
    parser.add_argument('--out_ch', type=int, required=False, default=2)

    parser.add_argument('--image_size', type=int, required=False, default=512)
    parser.add_argument('--num_workers', type=int, required=False, default=8)

    parser.add_argument('--path_save_model', type=str)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--path_save_log', type=str, required=False, default='./logs/')
    
    # 智能选择设备
    if torch.cuda.is_available():
        # 检查可用的 GPU 数量
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            # 选择第一个可用的 GPU
            default_device = 'cuda:0'
            print(f"Found {gpu_count} GPU(s), using {default_device}")
        else:
            default_device = 'cpu'
            print("No GPU available, using CPU")
    else:
        default_device = 'cpu'
        print("CUDA not available, using CPU")
    
    parser.add_argument('--device', type=str, required=False, default=default_device)
    config = parser.parse_args()
    
    targets = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    targets.remove(config.Source_Dataset)
    dice_score = 0
    results_dict = {}  # 存储所有域的评估结果
    
    print(f"Starting TTA evaluation for {config.Source_Dataset} on {len(targets)} target domains")
    print("=" * 50)
    
    # 使用 tqdm 显示总体进度
    overall_pbar = tqdm(targets, desc=f"Overall Progress ({config.Source_Dataset})", 
                       ncols=100, position=0)
    
    for i, config.Target_Dataset in enumerate(overall_pbar):
        overall_pbar.set_description(f"Processing {config.Target_Dataset}")
        overall_pbar.set_postfix({'Progress': f'{i+1}/{len(targets)}'})
        
        TTA = TrainTTA(config)
        metric = TTA.run()
        mean_dice = (metric['Disc_Dice'] + metric['Cup_Dice']) / 2
        dice_score += mean_dice
        
        # 存储结果
        results_dict[config.Target_Dataset] = metric
        
        # 更新总体进度条
        overall_pbar.set_postfix({
            'Progress': f'{i+1}/{len(targets)}',
            'Current Dice': f'{mean_dice:.3f}',
            'Avg Dice': f'{dice_score/(i+1):.3f}'
        })
    
    overall_pbar.close()
    
    final_avg_dice = dice_score / len(targets)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS for {config.Source_Dataset}")
    print(f"{'='*60}")
    print(f"Average Dice Score: {final_avg_dice:.4f}")
    print(f"Total Target Domains: {len(targets)}")
    print(f"{'='*60}")
    
    # 生成可视化结果
    print("\nGenerating visualization results...")
    try:
        create_visualization_results(results_dict, config.Source_Dataset)
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Continuing without visualization...")

