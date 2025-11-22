import torch, bitsandbytes as bnb
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Bitsandbytes loaded:", hasattr(bnb, 'Config'))
try:
    import bitsandbytes.cuda_setup as cuda_setup
    print("CUDA libs:", cuda_setup.get_compute_capability())
except:
    print("CUDA setup check failed")
