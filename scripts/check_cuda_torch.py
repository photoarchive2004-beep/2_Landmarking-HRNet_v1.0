import torch

print("torch version:", torch.__version__)
print("cuda.is_available:", torch.cuda.is_available())
print("cuda.device_count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(idx)
        print(f"[GPU {idx}] name={name}")
