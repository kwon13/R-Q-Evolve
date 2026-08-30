import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    os.environ["MASTER_ADDR"] = "172.19.0.2"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    
    dist.init_process_group("nccl", rank=rank, world_size=size)
    
    torch.cuda.set_device(rank)
    tensor = torch.ones(1).cuda(rank)
    
    print(f"Rank {rank} before all_reduce: {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} after all_reduce: {tensor}")

if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = mp.Process(target=run, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
