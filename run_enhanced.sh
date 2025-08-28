#!/bin/bash

# SPCL集成的GraTa增强版运行脚本
# 支持更多高级损失函数，提升医学图像分割性能

Device=0
model_root=./models
data_root=./Dataset/Fundus

# 基础配置：使用传统损失
echo "=== 运行基础配置 ==="
Aux=ent
Pse=consis

# SPCL增强配置：使用像素级对比学习
echo "=== 运行SPCL像素对比增强配置 ==="
Aux_Enhanced=pixel_cl
Pse_Enhanced=enhanced_pseudo

# 混合配置：结合传统损失和SPCL损失
Aux_Mixed=ent
Pse_Mixed=pixel_cl

echo "开始训练，支持多种损失组合..."

# 配置1: 传统配置 (基准对比)
echo "配置1: 传统损失组合"
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

echo "配置1完成！"

# 配置2: SPCL增强配置
echo "配置2: SPCL增强损失组合"
Source=RIM_ONE_r3
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Enhanced --pse_loss $Pse_Enhanced --path_save_model $model_root --dataset_root $data_root

Source=REFUGE
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Enhanced --pse_loss $Pse_Enhanced --path_save_model $model_root --dataset_root $data_root

Source=ORIGA
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Enhanced --pse_loss $Pse_Enhanced --path_save_model $model_root --dataset_root $data_root

Source=REFUGE_Valid
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Enhanced --pse_loss $Pse_Enhanced --path_save_model $model_root --dataset_root $data_root

Source=Drishti_GS
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Enhanced --pse_loss $Pse_Enhanced --path_save_model $model_root --dataset_root $data_root

echo "配置2完成！"

# 配置3: 混合配置
echo "配置3: 混合损失组合"
Source=RIM_ONE_r3
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Mixed --pse_loss $Pse_Mixed --path_save_model $model_root --dataset_root $data_root

Source=REFUGE
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Mixed --pse_loss $Pse_Mixed --path_save_model $model_root --dataset_root $data_root

Source=ORIGA
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Mixed --pse_loss $Pse_Mixed --path_save_model $model_root --dataset_root $data_root

Source=REFUGE_Valid
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Mixed --pse_loss $Pse_Mixed --path_save_model $model_root --dataset_root $data_root

Source=Drishti_GS
CUDA_VISIBLE_DEVICES=$Device python TTA.py --Source_Dataset $Source --aux_loss $Aux_Mixed --pse_loss $Pse_Mixed --path_save_model $model_root --dataset_root $data_root

echo "配置3完成！"

echo "=== 所有配置运行完成 ==="
echo "请查看日志文件比较不同损失组合的性能提升"