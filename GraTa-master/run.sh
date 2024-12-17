#!/bin/bash

Device=3
model_root=/media/userdisk0/zychen/GraTa-master/models
data_root=/media/userdisk0/zychen/Datasets/Fundus
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
