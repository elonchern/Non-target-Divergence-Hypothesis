## semantic kitti
python -m torch.distributed.launch --nproc_per_node=2 --master_port=63445 --use_env main.py config_server_kitti.yaml

