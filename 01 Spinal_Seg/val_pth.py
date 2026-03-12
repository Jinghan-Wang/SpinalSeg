import torch
import numpy as np
import os
import SimpleITK as sitk
from utils import crop_data_size
import imageio
from model.CAUnet import CAUNet
import torch.nn.functional as F
from PIL import Image

def read_dicom_as_array(dicom_dir):
    image = sitk.ReadImage(dicom_dir)
    image_array = sitk.GetArrayFromImage(image)
    return image_array
def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask*20).astype(np.uint8))
    elif mask.ndim == 3:
        a = np.argmax(mask, axis=0)
        a = np.array(a)
        a = a.astype(np.uint8)
        return a

ct_dir = r'/data/Project/TensorRT Deployment Process/01 Spinal_Seg/test_data/3173607/1'

ct_list = os.listdir(ct_dir)

# for i in  range(len(ct_list)):
file_path = os.path.join(ct_dir,ct_list[1])
print(file_path)
input_ = read_dicom_as_array(file_path)
input_ = np.squeeze(input_)
input_,input__= crop_data_size(input_,input_,256,256)
input_[input_ < -50] = 0
input_[input_ > 1450] = 0
save_SliceData = (input_ - np.min(input_)) / (np.max(input_) - np.min(input_)) * 255
save_SliceData = save_SliceData.astype(np.uint8)
imageio.imsave("/data/Project/TensorRT Deployment Process/01 Spinal_Seg/test_result/val_pth/input.png", save_SliceData)
SliceData = input_ / 1500

input = SliceData
"""加载模型"""
# 模型路径
model_path = "/data/Project/TensorRT Deployment Process/01 Spinal_Seg/save_pth/unet_finetuning500.pth"
net = CAUNet(n_channels=1, n_classes=2, bilinear=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
net.load_state_dict(torch.load(model_path, map_location=device))
# 预测
net.eval()
img = torch.from_numpy(input)
img = img.unsqueeze(0)
img = img.unsqueeze(0)
print(img.shape)
img = img.to(device=device, dtype=torch.float32)
with torch.no_grad():
    output = net(img)
    if net.n_classes > 1:
        print("softmax")
        print(type(output))
        probs = F.softmax(output, dim=1)[0]
    else:
        probs = torch.sigmoid(output)[0]
    print("probs.cpu().shape ",probs.cpu().shape)
    full_mask = probs.cpu().squeeze()
    output = mask_to_image(full_mask)
    print(np.max(output))
    print(np.min(output))
    print(np.unique(output))
    print(np.sum(output == 1))   #1436
    imageio.imsave('/data/Project/TensorRT Deployment Process/01 Spinal_Seg/test_result/val_pth/output.png', output * 255)