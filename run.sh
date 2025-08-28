#!/bin/bash

# # 激活 conda 虚拟环境
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate grata

# 验证包版本
# python -c "
# import pandas as pd
# import numpy as np
# import torch
# print('pandas version:', pd.__version__)
# print('numpy version:', np.__version__)
# print('PyTorch version:', torch.__version__)
# print('CUDA available:', torch.cuda.is_available())
# "

Device=0
model_root=./models
data_root=./Dataset/Fundus
Aux=ent
Pse=consis

Source=RIM_ONE_r3
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux --pse_loss $Pse --path_save_model $model_root --dataset_root $data_root

Source=REFUGE
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux --pse_loss $Pse --path_save_model $model_root --dataset_root $data_root

Source=ORIGA
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux --pse_loss $Pse --path_save_model $model_root --dataset_root $data_root

Source=REFUGE_Valid
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux --pse_loss $Pse --path_save_model $model_root --dataset_root $data_root

Source=Drishti_GS
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux --pse_loss $Pse --path_save_model $model_root --dataset_root $data_root
