export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_DISABLE=1

python3 -u -m torch.distributed.launch \
  --nnodes=$HOST_NUM \
  --node_rank=$INDEX \
  --master_addr $CHIEF_IP \
  --nproc_per_node $HOST_GPU_NUM \
  --master_port 8081 \
  run.py