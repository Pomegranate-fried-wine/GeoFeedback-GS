#!/usr/bin/env python3
"""Small CUDA runtime smoke check for A100 servers."""

import os
import time

import torch


def main():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch.")

    device = torch.device("cuda:0")
    print(f"current_device={torch.cuda.current_device()}")
    print(f"device_name={torch.cuda.get_device_name(device)}")
    x = torch.randn((2048, 2048), device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    y = x @ x.T
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(f"matmul_elapsed_sec={elapsed:.4f}")
    print(f"tensor_device={y.device}")
    print(f"memory_allocated_mb={torch.cuda.memory_allocated(device) / 1024 / 1024:.1f}")
    print(f"memory_reserved_mb={torch.cuda.memory_reserved(device) / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
