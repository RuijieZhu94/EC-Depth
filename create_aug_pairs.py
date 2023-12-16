import glob
from imagecorruptions import corrupt
from random import choice, sample
import PIL
from PIL import Image
from datasets.mono_dataset import pil_loader
import os
import numpy as np
from tqdm import tqdm

# custom config 
data_path = "./data/kitti"
split_path = "splits/eigen_zhou/train_files.txt" # train or val

corrupt_list = ["brightness", "color_quant", "contrast", "dark", "defocus_blur", 
                "elastic_transform", "fog", "frost", "gaussian_noise", "glass_blur", 
                "impulse_noise", "iso_noise", "jpeg_compression", "motion_blur", 
                "pixelate", "shot_noise", "snow", "zoom_blur"]
custom = ["color_quant", "dark", "iso_noise"]
side_map = {"2": 2, "3": 3, "l": 2, "r": 3}


def low_light(x, severity):
    c = [0.60, 0.50, 0.40, 0.30, 0.20][severity-1]
    x = np.array(x) / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2.) * 255
    x_scaled = poisson_gaussian_noise(x_scaled, severity=severity-1)
    return x_scaled


def imadjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def poisson_gaussian_noise(x, severity):
    c_poisson = 10 * [60, 25, 12, 5, 3][severity]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
    c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
    return Image.fromarray(np.uint8(x))


def color_quant(x, severity):
    bits = 5 - severity + 1
    x = PIL.ImageOps.posterize(x, bits)
    return x


def iso_noise(x, severity):
    c_poisson = 25
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.
    c_gauss = 0.7 * [.08, .12, 0.18, 0.26, 0.38][severity-1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255.
    return Image.fromarray(np.uint8(x))


def index_to_folder_and_frame_idx(path_info):
    """Convert index in the dataset to a folder name, frame_idx and any other bits
    """
    line = path_info.split()
    folder = line[0]

    if len(line) == 3:
        frame_index = int(line[1])
    else:
        frame_index = 0

    if len(line) == 3:
        side = line[2]
    else:
        side = None

    return folder, frame_index, side

def get_image_path(folder, frame_index, side):
    f_str = "{:010d}{}".format(frame_index, ".png")
    image_path = os.path.join(
        data_path, folder, "image_0{}/data".format(side_map[side]), f_str)
    return image_path

def image_corruption(img, corruption_name):
    corruption_severity = choice([1,2,3,4,5])
    if corruption_name == "color_quant":
        img_aug = color_quant(img, severity=corruption_severity)
    elif corruption_name == "dark":
        img_aug = low_light(img, severity=corruption_severity)
    elif corruption_name == "iso_noise":
        img_aug = iso_noise(img, severity=corruption_severity)
    else:
        img_aug = corrupt(np.array(img), corruption_name=corruption_name, severity=corruption_severity)
        img_aug = Image.fromarray(img_aug)
    return img_aug

def main():
    with open(split_path, "r") as f:
        path_infos = f.readlines()
        
    for path_info in tqdm(path_infos):
        folder, frame_index, side = index_to_folder_and_frame_idx(path_info)
        path = get_image_path(folder, frame_index, side)
        img = pil_loader(path)
        corruption_name = sample(corrupt_list, 2)
    
        img_aug1 = image_corruption(img, corruption_name[0])
        img_aug2 = image_corruption(img, corruption_name[1])
        save_folder = os.path.dirname(path) + "/../"
        imgname = os.path.split(path)[-1]
        save_dir1 = os.path.join(save_folder, "aug1")
        save_dir2 = os.path.join(save_folder, "aug2")
        if not os.path.exists(save_dir1):
            os.mkdir(save_dir1)
            os.mkdir(save_dir2)
        img_aug1.save(os.path.join(save_dir1, imgname))
        img_aug2.save(os.path.join(save_dir2, imgname))

if __name__ == "__main__":
    main()

    

