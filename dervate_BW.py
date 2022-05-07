import cv2
import matplotlib.pyplot as plt
import torch.nn
import numpy as np
from torch.autograd import Variable
import matplotlib
from BoniDL.utils import normalize01

matplotlib.use('TKAgg')

x_raw = np.mgrid[0: 512, 0: 512].reshape(2, -1).T
input_tensor = x_raw.copy()

input_coord = input_tensor.astype(np.float32) / 512.
input_coord = torch.from_numpy(input_coord[np.newaxis, :])
input_coord = input_coord.cuda()
input_coord = Variable(input_coord)

model_path = './checkpoints_gray/weights.pth'
model = torch.load(model_path)

model = model.eval()
model = model.cuda()
output_quantity = model(input_coord)
output_quantity_raw = output_quantity.cpu().data.numpy()[0]
output_quantity = np.uint8(output_quantity_raw.copy().reshape((512, 512)) * 255.)
output_quantity = cv2.rotate(output_quantity, cv2.ROTATE_90_CLOCKWISE)
output_quantity = cv2.flip(output_quantity, 1)

x = np.arange(output_quantity_raw.shape[0]).astype('float32')
y = output_quantity_raw.copy().flatten()
dx = x[1] - x[0]
dydx = np.gradient(y, dx)

dydx = normalize01(dydx)
derivate_image = np.uint8(dydx.copy().reshape((512, 512)) * 255.)
derivate_image = cv2.rotate(derivate_image, cv2.ROTATE_90_CLOCKWISE)
derivate_image = cv2.flip(derivate_image, 1)

plt.imshow(derivate_image, cmap='gray')
plt.show()

plt.imshow(output_quantity, cmap='gray')
plt.show()
