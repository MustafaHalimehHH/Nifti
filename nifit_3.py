import os
import nibabel
import matplotlib.pyplot as plt

path = 'D:\Halimeh\Datasets\MSD\Task05_Prostate\imagesTr'
obj = nibabel.load(os.path.join(path, 'prostate_02.nii.gz'))
print('obj', type(obj))
print(obj)
data = obj.get_data()
print('data', type(data), data.shape)

def show_slices(image):
    data = image.get_data()
    fig, axs = plt.subplots(4, 5)
    cnt = 0
    for i in range(4):
        for j in range(5):
            axs[i, j].imshow(data[:, :, cnt, 0], cmap='gray')
            # axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1

    fig, axs = plt.subplots(4, 5)
    cnt = 0
    for i in range(4):
        for j in range(5):
            axs[i, j].imshow(data[:, :, cnt, 1], cmap='gray')
            # axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()

show_slices(obj)