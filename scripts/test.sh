#APFP-Market-1501
python test.py --gpu_ids 1 --network_arch resnet18 --name ./results/ft_resnet18/prune_APFP/knn2/market/0.90/no_1x1_3e-1_70_0.01_m95_fast/ --test_dir ./datasets/Market/pytorch/  --which_epoch last
python evaluate_gpu.py

python test.py --gpu_ids 1 --network_arch resnet34 --name ./results/ft_resnet34/prune_APFP/knn2/market/0.90/no_1x1_3e-1_70_0.01_m95_fast/ --test_dir ./datasets/Market/pytorch/  --which_epoch last
python evaluate_gpu.py

python test.py --gpu_ids 1 --network_arch resnet50 --name ./results/ft_resnet50/prune_APFP/knn2/market/0.90/no_1x1_3e-1_70_0.01_m95_fast/ --test_dir ./datasets/Market/pytorch/  --which_epoch last
python evaluate_gpu.py

#APFP-DukeMTMC-reID
python test.py --gpu_ids 1 --network_arch resnet18 --name ./results/ft_resnet18/prune_APFP/knn2/duke/0.90/no_1x1_3e-1_70_0.01_m95_fast/ --test_dir ./datasets/DukeMTMC-reID/pytorch/  --which_epoch last
python evaluate_gpu.py

python test.py --gpu_ids 1 --network_arch resnet34 --name ./results/ft_resnet34/prune_APFP/knn2/duke/0.90/no_1x1_3e-1_70_0.01_m95_fast/ --test_dir ./datasets/DukeMTMC-reID/pytorch/  --which_epoch last
python evaluate_gpu.py

python test.py --gpu_ids 1 --network_arch resnet50 --name ./results/ft_resnet50/prune_APFP/knn2/duke/0.90/no_1x1_3e-1_70_0.01_m95_fast/ --test_dir ./datasets/DukeMTMC-reID/pytorch/  --which_epoch last
python evaluate_gpu.py

#APFP-MSMT17_V1
python test.py --gpu_ids 1 --network_arch resnet18 --name ./results/ft_resnet18/prune_APFP/knn2/msmt/0.90/no_1x1_3e-1_70_0.01_m95_fast/ --test_dir ./datasets/MSMT17_V1/pytorch/  --which_epoch last
python evaluate_gpu.py

python test.py --gpu_ids 1 --network_arch resnet34 --name ./results/ft_resnet34/prune_APFP/knn2/msmt/0.90/no_1x1_3e-1_70_0.01_m95_fast/ --test_dir ./datasets/MSMT17_V1/pytorch/  --which_epoch last
python evaluate_gpu.py

python test.py --gpu_ids 1 --network_arch resnet50 --name ./results/ft_resnet50/prune_APFP/knn2/msmt/0.90/no_1x1_3e-1_70_0.01_m95_fast/ --test_dir ./datasets/MSMT17_V1/pytorch/  --which_epoch last
python evaluate_gpu.py

#SFP
python test.py --gpu_ids 1 --network_arch resnet50 --name ./results/ft_resnet50/prune_SFP/market/0.90/ --test_dir ./datasets/Market/pytorch --which_epoch last
python evaluate_gpu.py

python test.py --gpu_ids 1 --network_arch resnet50 --name ./results/ft_resnet50/prune_SFP/duke/0.90/ --test_dir ./datasets/DukeMTMC-reID/pytorch --which_epoch last
python evaluate_gpu.py

python test.py --gpu_ids 1 --network_arch resnet50 --name ./results/ft_resnet50/prune_SFP/msmt/0.90/ --test_dir ./datasets/MSMT17_V1/pytorch --which_epoch last
python evaluate_gpu.py

#FPGM
python test .py --gpu_ids 1 --network_arch resnet18 --name ./results/ft_resnet18/prune_FPGM/market/0.90/ --test_dir ./datasets/Market/pytorch --which_epoch last
python evaluate_gpu.py

python test .py --gpu_ids 1 --network_arch resnet18 --name ./results/ft_resnet18/prune_FPGM/duke/0.90/ --test_dir ./datasets/DukeMTMC-reID/pytorch --which_epoch last
python evaluate_gpu.py

python test.py --gpu_ids 1 --network_arch resnet18 --name ./results/ft_resnet18/prune_FPGM/msmt/0.90/ --test_dir ./datasets/MSMT17_V1/pytorch --which_epoch last
python evaluate_gpu.py
