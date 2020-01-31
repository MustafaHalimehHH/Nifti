from __future__ import print_function
import os
import nibabel
import numpy
import medpy
import medpy.io
import medpy.features


print('nibabel', nibabel.__version__)
print('numpy', numpy.__version__)
print('medpy', medpy.__version__)


DATA_PATH = 'D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr\prostate_02.nii.gz'

image_data, image_header = medpy.io.load(DATA_PATH)
print('image_data', type(image_data), image_data.shape, image_data.dtype)
print('image_header', type(image_header), image_header)
print('pixel_spacing', medpy.io.header.get_pixel_spacing(image_header))
print('voxel_spacing', medpy.io.header.get_voxel_spacing(image_header))
print('offset', medpy.io.header.get_offset(image_header))
print(image_header)

print('FEATURES EXTRACTION')
f1 = medpy.features.intensity.intensities(image_data)
f2 = medpy.features.intensity.centerdistance(image_data)
print('f1', f1.shape)
print('f2', f2.shape)