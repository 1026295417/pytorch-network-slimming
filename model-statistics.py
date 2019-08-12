import sys
import torch
from netslim import load_pruned_model
from thop import profile

from networks import resnet18, vgg11, vgg11s, densenet63

archs = {
    "resnet18": resnet18, 
    "vgg11": vgg11, "vgg11s": vgg11s, 
    "densenet63": densenet63
}

arch_name = sys.argv[1]
weight_path = sys.argv[2]
model = archs[arch_name](num_classes=100)
weight = torch.load(weight_path, map_location="cpu")

try:
    model.load_state_dict(weight)
except:
    model = load_pruned_model(model, weight)
    
input_t = torch.randn(1, 3, 32, 32)
flops, params = profile(model, inputs=(input_t,), verbose=False)
flops_str = format(int(flops), ',')
gflops = flops / 1024**3
gflops_str = "{:.2f} GFLOPS".format(gflops)
params_str = format(int(params), ',')
mparams = params / 1024**2
mparams_str = "{:.2f} M".format(mparams)
line = "{}/{}: FLOPS: {} / {}\t# of params: {} / {}".format(arch_name, weight_path, flops_str, gflops_str, params_str, mparams_str)
print(line)
