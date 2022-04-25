import numpy as np
import torch
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')


USE_CUDA = torch.cuda.is_available()
weights_path = './checkpoints/weights.pth'
model = torch.load(weights_path)
print(model)
model = model.eval()
if USE_CUDA:
    model = model.cuda()

input_tensor = np.mgrid[0: 512, 0: 512].reshape(2, -1).T

input_coord = input_tensor.astype(np.float32) / 512.
input_coord = torch.from_numpy(input_coord[np.newaxis, :])
input_coord = input_coord.cuda()
input_coord = Variable(input_coord)

output_quantity = model(input_coord)
output_quantity = output_quantity.cpu().data.numpy()[0]

output_quantity = np.uint8(output_quantity.reshape((512, 512, 3)) * 255.)
plt.imshow(output_quantity)
plt.show()
