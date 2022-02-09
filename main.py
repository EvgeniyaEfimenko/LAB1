import json
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage.io import imread, imshow, show
from skimage.exposure import histogram

settings= {
    'parameter_0':'img5.jpg',
    'parameter_1': -40, # shift x
    'parameter_2': -23, # shift y
    'parameter_3': 0.7, # scale x
    'parameter_4': 1.7, # scale y
    'parameter_5': 56, # angle of rotation
}

# пишем в файл
with open('settings.json', 'w') as fp:
     json.dump(settings, fp)

# читаем из файла
with open('settings.json') as json_file:
    json_data = json.load(json_file)

fig = plt.figure(figsize=(10, 5))

fig.add_subplot(2,2,1)
imshow(json_data['parameter_0'])

img = imread(json_data['parameter_0'])
tform = AffineTransform(scale=[json_data['parameter_3'],json_data['parameter_4']], rotation=json_data['parameter_5'], translation=[json_data['parameter_1'],json_data['parameter_2']])
print(type(tform))
new_img = warp(img, tform.inverse) # Деформируйте изображение в соответствии с заданным преобразованием координат.
fig.add_subplot(2,2,2)
print(new_img[1,1,:])
new_img*=255
print(type(new_img.dtype))
imshow(new_img.astype(np.uint8))

hist_red_num_1, bins_red_num_1 = histogram(img[:,:,0])
hist_green_num_1, bins_green_num_1 = histogram(img[:,:,1])
hist_blue_num_1, bins_blue_num_1 = histogram(img[:,:,2])
fig.add_subplot(2,2,3)
plt.ylabel('число отсчетов')
plt.xlabel('значение яркости')
plt.title('Гистограмма распределения яркостей по каждому каналу')
plt.plot(bins_green_num_1,hist_green_num_1, color='green', linestyle = '-', linewidth=1)
plt.plot(bins_red_num_1,hist_red_num_1, color='red', linestyle = '-', linewidth=1)
plt.plot(bins_blue_num_1,hist_blue_num_1, color='blue', linestyle = '-', linewidth=1)
plt.legend(['green','red','blue'])

hist_red_num_1, bins_red_num_1 = histogram(new_img[:,:,0])
hist_green_num_1, bins_green_num_1 = histogram(new_img[:,:,1])
hist_blue_num_1, bins_blue_num_1 = histogram(new_img[:,:,2])
fig.add_subplot(2,2,4)
plt.ylabel('число отсчетов')
plt.xlabel('значение яркости')
plt.title('Гистограмма распределения яркостей по каждому каналу')
plt.plot(bins_green_num_1,hist_green_num_1, color='green', linestyle = '-', linewidth=1)
plt.plot(bins_red_num_1,hist_red_num_1, color='red', linestyle = '-', linewidth=1)
plt.plot(bins_blue_num_1,hist_blue_num_1, color='blue', linestyle = '-', linewidth=1)
plt.legend(['green','red','blue'])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

show()