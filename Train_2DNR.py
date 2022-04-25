import cv2
import torch.nn
import numpy as np
from Model_2DNR import NeuralRendererModel2D
from DataLoader_2DNR import DL_2DNR
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch.autograd import Variable


def generate_image(ep, model):
    input_tensor = np.mgrid[0: 512, 0: 512].reshape(2, -1).T

    input_coord = input_tensor.astype(np.float32) / 512.
    input_coord = torch.from_numpy(input_coord[np.newaxis, :])
    input_coord = input_coord.cuda()
    input_coord = Variable(input_coord)

    model = model.eval()
    output_quantity = model(input_coord)
    output_quantity = output_quantity.cpu().data.numpy()[0]
    output_quantity = np.uint8(output_quantity.reshape((512, 512, 3)) * 255.)[:, :, ::-1]
    output_quantity = cv2.rotate(output_quantity, cv2.ROTATE_90_CLOCKWISE)
    output_quantity = cv2.flip(output_quantity, 1)
    cv2.imwrite(f'./checkpoints/output_images/epoch_{ep}.jpg', output_quantity)


BATCH_SIZE = 512
LEARNING_RATE = 1e-3
USE_CUDA = torch.cuda.is_available()
N_EPOCHS = 100

data_set = DL_2DNR()
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)

model = NeuralRendererModel2D(w0=5.)
print(model)

if USE_CUDA:
    model = model.cuda()

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lambda1 = lambda epoch: 0.65 ** epoch
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

loss = torch.nn.MSELoss()
if USE_CUDA:
    loss = loss.cuda()

for p in model.parameters():
    p.requires_grad = True

for epoch in range(N_EPOCHS + 1):
    data_iter = iter(data_loader)
    i = 0
    epoch_errors = []
    while i < len(data_loader):
        sample = next(data_iter)
        s_input_coordinates, s_output_quantity = sample['input_coordinates'], sample['output_quantity']

        model.zero_grad()
        if USE_CUDA:
            s_input_coordinates = s_input_coordinates.cuda()
            s_output_quantity = s_output_quantity.cuda()

        s_input_coordinates_v = Variable(s_input_coordinates)
        s_output_quantity_v = Variable(s_output_quantity)

        s_quantity_output = model(s_input_coordinates_v)
        err = loss(s_quantity_output, s_output_quantity_v)

        err.backward()
        optimizer.step()

        i += 1
        errr = err.cpu().data.numpy()
        epoch_errors.append(errr)

        # print(f'[Epoch: {epoch}/{N_EPOCHS}], [It: {i}/{len(data_loader)}], [Err_label: {errr}]')

    scheduler.step()
    print(f'[Epoch: {epoch}/{N_EPOCHS}], [Err_label: {np.mean(epoch_errors)}]')
    # torch.save(model, f'./checkpoints/epoch_{epoch}.pth')
    torch.save(model, f'./checkpoints/weights.pth')
    generate_image(epoch, model)



