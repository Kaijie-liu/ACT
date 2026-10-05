import torch, time, torch.nn.functional as F
torch.backends.cudnn.benchmark = True
for dt in (torch.float32, torch.float64):
    x = torch.randn(1024, 32, 32, 32, device="cuda", dtype=dt); W = torch.randn(32, 32, 3, 3, device="cuda", dtype=dt)
    for _ in range(2): F.conv2d(x, W, padding=1)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(3): F.conv2d(x, W, padding=1)
    torch.cuda.synchronize(); tc = (time.time() - t) / 3
    t = time.time()
    for _ in range(3):
        cols = F.unfold(x, 3, padding=1)                     # [N, C*9, HW]
        y = (W.reshape(32, -1) @ cols).reshape(1024, 32, 32, 32)
    torch.cuda.synchronize(); tu = (time.time() - t) / 3
    flops = 2 * 1024 * 32 * 32 * 32 * 32 * 9
    print(dt, f"cudnn {tc*1e3:.1f} ms ({flops/tc/1e12:.2f} TFLOPS)  unfold+gemm {tu*1e3:.1f} ms ({flops/tu/1e12:.2f} TFLOPS)")
