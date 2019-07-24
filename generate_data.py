import cv2
import glob
import numpy as np
import os
import random
import sys
import subprocess

from tqdm import tqdm

from image_augmentation import ImageAug
from utils import para_func,mkdir


# def except_catcher(func):
#     '''封面/视频帧识别结果装饰器
#     '''
#     def warpper(*args, **kw):
#         try:
#             res = func(*args, **kw)
#         except Exception as e:
#             msg = str(e).replace('\'', '').replace('"', '').replace('\\', '')
#             print(msg)
#             res = None
#         return res
#     return warpper

# def show(image):
#         cv2.imshow('image', image)
#         cv2.waitKey(0)

def pixelate(input_path, output_path):
    # try:
    image = cv2.imread(input_path)
    aug, pos = ImageAug.random_pixelate(image)
    if not pos:
        return None, None, None
    cv2.imwrite(output_path, aug)
    return output_path, pos, 0
    # except Exception as e:
    #     print(e)
    #     print(input_path)
    #     return None, None, None

def delogo(input_path, output_path):
    def random_position(h, w):
        n_h = random.randint(int(0.1*h), int(0.4*h))
        n_w = random.randint(int(0.1*w), int(0.4*w))
        top = random.randint(0, h - n_h )
        left = random.randint(0, w - n_w )
        position = (top, left, n_h, n_w)
        position_norm = (left/w, top/h, (left+n_w)/w, (top+n_h)/h)
        return position, position_norm
    try:
        image = cv2.imread(input_path)
        h, w = image.shape[:2]
        position, position_norm = random_position(h, w)
        t, l, h, w = position
        cmd = 'ffmpeg -nostats -loglevel 1 -y -i {} -vf delogo=x={}:y={}:w={}:h={} -c:a copy {}'\
                .format(input_path, l,t,w,h, output_path)
        subprocess.run(cmd.split(' '))
        return output_path, position_norm, 1
    except Exception as e:
        print(e)
        print(input_path)
        return None, None, None

def single_data(input_path, output_path):
    if random.uniform(0, 1) > 0.7:
        path, pos, label = pixelate(input_path, output_path)
    else:
        path, pos, label = delogo(input_path, output_path)
    return path, pos, label

def dir2dir(input_dir, output_dir, num_multi=10):
    output_dir = os.path.abspath(output_dir)
    data_path = glob.glob(input_dir + '/*.jpg')
    print(data_path[:10])
    if isinstance(num_multi, int):
        data_path = data_path * num_multi
        random.shuffle(data_path)
    else:
        random.shuffle(data_path)
        data_path = data_path[:int(len(data_path)* num_multi)]
    output_path = [os.path.join(output_dir, str(p)+'.jpg') 
                    for p in range(len(data_path))]
    mkdir(output_dir)
    params = list(zip(data_path, output_path))
    result = para_func(single_data, params) 
    # result = para_func(pixelate, params, False)
    print(output_dir, 'finished')
    # result = []
    # for input_path, output_path in tqdm(params):
    #     res = single_data(input_path, output_path)
    #     result.append(res)
    return result

def main():
    if sys.platform == 'linux':
        input_dir = '/home/admin-seu/hugh/images/images'
        data_dir = '/home/admin-seu/hugh/yolov3-tf2/data_native'
        nums_multi = [0.5, 0.1, 0.1]
    else:
        input_dir = '/home/admin-seu/hugh/images/images'
        data_dir = '/home/admin-seu/hugh/yolov3-tf2/data_native'
        nums_multi = [3, 0.1, 0.1]

    def gen(data_type, num_multi):
        output_dir = '{}/{}'.format(data_dir, data_type)
        txt = '{}/{}.txt'.format(data_dir, data_type)
        try:
            result = dir2dir(input_dir, output_dir, num_multi)
        except:
            return
        # result = [r for r in result if r[0] is not None]
        class_label = 0
        with open(txt, 'w', encoding='utf-8') as f:
            content = []
            for path, position, label in result:
                if path is None or not os.path.exists(path) :
                    continue
                content.append('{} {},{}'.format(path, ','.join(map(str, position)), label))
            random.shuffle(content)
            f.write('\n'.join(content))

    for dtype, num in zip(['train', 'eval', 'valid'], nums_multi):
        gen(dtype, num)


if __name__ == '__main__':
    main()
    # print(list(range(4)))
    # np_bar()  
