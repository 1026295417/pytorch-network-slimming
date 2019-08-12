import os
import sys
import time
import random
import numpy
from tqdm import tqdm
import torch
from openvino.inference_engine import IECore, IENetwork
from networks import vgg11s
from netslim import load_pruned_model

# Initialize inference engine core
ie = IECore()

# build network with FP32
model_name = "vgg11s-pr05-FP32"
net = IENetwork(model="{}.xml".format(model_name), weights="{}.bin".format(model_name))
input_name = next(iter(net.inputs))
output_name = next(iter(net.outputs))
exec_net = ie.load_network(network=net, device_name="CPU")
openvino_pr05_fp32_pack = ["vgg11s-pr05-fp32-openvino", exec_net.infer, input_name]

# build network with FP16
model_name = "vgg11s-pr05-FP16"
net = IENetwork(model="{}.xml".format(model_name), weights="{}.bin".format(model_name))
input_name = next(iter(net.inputs))
output_name = next(iter(net.outputs))
exec_net = ie.load_network(network=net, device_name="CPU")
openvino_pr05_fp16_pack = ["vgg11s-pr05-fp16-openvino", exec_net.infer, input_name]

def run_openvino(net, input_name, n):
    for i in range(n):
        inputs = {input_name: numpy.random.randn(1, 3, 32, 32)}
        _ = net(inputs)

openvino_pr05_fp32_pack.append(run_openvino)
openvino_pr05_fp16_pack.append(run_openvino)

# build network for pytorch
def run_pytorch(net, _, n):
    with torch.no_grad():
        for i in range(n):
            inputs = torch.randn(1, 3, 32, 32)
            _ = net(inputs)

pytorch_fp32_pack = ["vgg11s-fp32-pytorch", vgg11s(), 0, run_pytorch]
 
pruned_weights = torch.load("output-vgg11s-bn-pr05/ckpt_best.pth", map_location="cpu")
pruned_net = load_pruned_model(vgg11s(), pruned_weights)
pytorch_pr05_fp32_pack = ["vgg11s-pr05-fp32-pytorch", pruned_net, 0, run_pytorch]

# benchmark
packs = [pytorch_fp32_pack, pytorch_pr05_fp32_pack, openvino_pr05_fp32_pack, openvino_pr05_fp16_pack]
loops = 50
iters = 1000
total_t = {_[0]: 0. for _ in packs}
for _ in tqdm(range(loops)):
    random.shuffle(packs)
    for pack in packs:
        pack_name, net, input_name, run_fn = pack
        t = time.time()
        run_fn(net, input_name, n=iters)
        total_t[pack_name] += time.time() - t

for k, v in total_t.items():
    print("FPS of {}: {}".format(k, loops*iters/v))
