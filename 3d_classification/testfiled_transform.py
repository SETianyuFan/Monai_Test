import os
import numpy as np
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, RandFlip, RandZoom, SpatialPad, Resize
import nibabel as nib
from monai.transforms import LoadImage


# directory = '/home/tianyu/Desktop/data_base/imagesTr_cropped'  # data directory
# images = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith('.nii.gz')]
# images = np.array(images)

nii_img = nib.load('/home/tianyu/Desktop/data_base/imagesTr_cropped/testSec_001.nii.gz')
data_matrix = nii_img.get_fdata()
data_with_channel = np.expand_dims(data_matrix, axis=0)

train_transforms = Compose(
    [ScaleIntensity(), EnsureChannelFirst(),
     # RandRotate90(prob=0.1),  # 10%的概率随机旋转90度
     RandFlip(spatial_axis=0, prob=0.1),  # 10%的概率进行随机翻转
     RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.1),  # 10%的概率进行随机缩放
     SpatialPad((128, 128, 128), mode='constant'),
     Resize((96, 96, 96))])

transformed_image = train_transforms(data_with_channel).numpy()
print(np.size(transformed_image))
transformed_nifti = nib.Nifti1Image(transformed_image[0], affine=np.eye(4))
nib.save(transformed_nifti, '/home/tianyu/Desktop/transformed_image.nii.gz')

