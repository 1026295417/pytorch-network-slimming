import os
import sys
import time
import numpy
import torch
from torchvision import transforms
from torchvision.datasets import cifar
from openvino.inference_engine import IECore, IENetwork

model_name = sys.argv[1]

normalize = transforms.Normalize(mean=[0.4914, 0.482, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])

test_loader = torch.utils.data.DataLoader(
    cifar.CIFAR100('./cifar-100', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       normalize
                   ])), batch_size=1, shuffle=True)

# Initialize inference engine core
ie = IECore()

# build network
net = IENetwork(model="{}.xml".format(model_name), weights="{}.bin".format(model_name))
input_name = next(iter(net.inputs))
output_name = next(iter(net.outputs))
exec_net = ie.load_network(network=net, device_name="CPU")

correct = 0
with torch.no_grad():
    t_start = time.time()
    for data, target in test_loader:
        outputs = exec_net.infer({input_name: data.numpy()})
        output = outputs[output_name][0]
        correct += numpy.argmax(output) == target.item()
    t_all = time.time() - t_start

accuracy = 100. * float(correct) / float(len(test_loader.dataset))
print("Accuracy: {}/{} ({:.2f}%)\n".format(correct, len(test_loader.dataset), accuracy))
print("Total time: {:.2f} s".format(t_all))
#print("Estimated FPS: {:.2f}".format(1/(t_all/len(test_loader))))
