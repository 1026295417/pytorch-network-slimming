import os
import argparse
import torch
from torch import nn
from networks import resnet18, vgg11, vgg11s, densenet63
from train import archs
from netslim import load_pruned_model

cmd_template = 'python /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model {0} --input_shape "{1}" --data_type {3} --model_name {2}-{3}'
input_shape = (1, 3, 32, 32)

parser = argparse.ArgumentParser(description='Export to OpenVINO')
parser.add_argument('--arch', default='resnet18',
                    help='network architecture')
parser.add_argument('--weights', default='',
                    help='output name')
parser.add_argument('--outname', default='',
                    help='output name')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='use half precision')

args = parser.parse_args()
if not args.outname:
    args.outname = args.arch

net = archs[args.arch]()

if args.weights:
    weights = torch.load(args.weights, map_location="cpu")
    try:
        net.load_state_dict(weights)
    except Exception as e:
        print("Direct load failed, trying to load pruned weight ...")
        net = load_pruned_model(net, weights)

onnx_path = "{}.onnx".format(args.outname)

print("Exporting to ONNX ...")
torch.onnx.export(net, torch.randn(*input_shape), onnx_path)

cmd = cmd_template.format(
    onnx_path, 
    repr(input_shape), 
    args.outname, 
    "FP16" if args.fp16 else "FP32"
)
print("Converting to OpenVINO IR ...")
os.system(cmd)
os.system("rm {}".format(onnx_path))
