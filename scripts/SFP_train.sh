#ython SFP_export_distribution.py --gpu_ids 1  --network_arch resnet34 --lr 0.01 --prune_rate 0.50 --Epochs 70 --save_epoch 60 --name prune_SFP/msmt/0.50 --train_all --batchsize 32  --data_dir ./datasets/MSMT17_V1/pytorch/

python SFP_train.py --gpu_ids 0  --network_arch resnet50 --lr 0.01 --prune_rate 0.90 --Epochs 70 --save_epoch 60 --name prune_SFP/market/0.90 --train_all --batchsize 32  --data_dir ./datasets/Market/pytorch/
python SFP_train.py --gpu_ids 0  --network_arch resnet50 --lr 0.01 --prune_rate 0.90 --Epochs 70 --save_epoch 60 --name prune_SFP/duke/0.90 --train_all --batchsize 32  --data_dir ./datasets/DukeMTMC-reID/pytorch/
python SFP_train.py --gpu_ids 0  --network_arch resnet50 --lr 0.01 --prune_rate 0.90 --Epochs 70 --save_epoch 60 --name prune_SFP/msmt/0.90 --train_all --batchsize 32  --data_dir ./datasets/MSMT17_V1/pytorch/

