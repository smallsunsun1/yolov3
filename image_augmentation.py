import cv2
import numpy as np
import random


class ImageAug:
    '''
    Shape Related Operation
    '''
    def fliplr(image):
        return np.fliplr(image)

    def flipud(image):
        return np.flipud(image)

    def rot(image):
        return np.rot90(image)

    def compress(image, num_quality=90):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY) , num_quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)

    def resize(image, ratio=1):
        h,w = image.shape[:2]
        h = int(ratio * h)
        w = int(ratio * w)
        return cv2.resize(image, (w, h))

    def random_crop(image, pad=True):
        h,w = image.shape[:2]
        c_h = random.randint(int(0.6*h),h)
        c_w = random.randint(int(0.6 *w),w)
        if pad:
            padding = random.randint(max(10,int(0.1*h)), max(20, int(0.2*h)))
            nh = random.randint(0, h + 2*padding - c_h)
            nw = random.randint(0, w + 2*padding - c_w)
            npad = ((padding, padding), (padding, padding), (0, 0))
            image = np.lib.pad(image, pad_width=npad, mode='constant', constant_values=0)
        else:
            nh = random.randint(0, h - c_h)
            nw = random.randint(0, w - c_w)
        image_crop = image[nh:nh + c_h, nw:nw + c_w]
        return image_crop

    '''
    Quality Related Operation
    '''
    def blur(image, kernel_size = (3, 3), sigma = 1):
        return cv2.GaussianBlur(image, kernel_size, sigma);

    def scale(image, ratio=0.8):
        image = (image * ratio)
        return np.clip(image, 0, 255).astype(np.uint8)

    def hue(image, shift=10):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0].astype(np.int)
        h = (h + shift) % 180
        hsv[..., 0 ] = h.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def saturation(image, shift=10):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1].astype(np.int)
        s = np.clip(s+shift, 0, 255).astype(np.uint8)
        hsv[..., 1] = s
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def brightness(image, shift=10):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2].astype(np.int)
        v = np.clip(v+shift, 0, 255).astype(np.uint8)
        hsv[..., 2] = v
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    '''
    Ohter Operation
    '''
    def pixelate(image, position, block_size=8):
        '''
        Args:
            position: (x, y, h, w)
            block_size: 8 default
        Returns:
            result: image with part
        '''
        x, y, h, w = position
        mosaic_part = image[x:x+h, y:y+w, ...]
        small_part = cv2.resize(mosaic_part, (w//block_size, h//block_size), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small_part, (w, h), interpolation=cv2.INTER_NEAREST)
        result = image.copy()
        result[x:x+h, y:y+w, ...] = mosaic
        return result

    def random_position(h, w):
        '''
        Args:
            h: height
            w: width
        Return:
            position: top, left, height, width
            position_norm: normed(left, top, right, bottom)
        '''
        n_h = random.randint(int(0.1*h), int(0.4*h))
        n_w = random.randint(int(0.1*w), int(0.4*w))
        top = random.randint(0, h - n_h )
        left = random.randint(0, w - n_w )
        position = (top, left, n_h, n_w)
        position_norm = (left/w, top/h, (left+n_w)/w, (top+n_h)/h)
        return position, position_norm

    '''
    Synthesized Operation
    '''
    def random_shape(image):
        if random.choice([0,1]):
            ratio = random.uniform(0.8, 1.2)
            image = ImageAug.resize(image, ratio)

        for i in range(random.randint(0, 3)):
            image = ImageAug.rot(image)

        if random.choice([0,1]):
            image = ImageAug.fliplr(image)

        if random.choice([0,1]):
            image = ImageAug.flipud(image)

        if random.choice([0,1]):
            pad = random.choice([0,1])
            image = ImageAug.random_crop(image, pad)
        return image

    def random_quality(image):
        if random.choice([0,1]):
            num_quality = random.randint(3, 10)
            image = ImageAug.compress(image, num_quality*10)

        if not random.choice(range(10)):
            image = ImageAug.blur(image)

        if random.choice([0,1]):
            shift = random.randint(-18, 18)
            image = ImageAug.hue(image, shift)

        if random.choice([0,1]):
            shift = random.randint(-25, 25)
            image = ImageAug.saturation(image, shift)

        if random.choice([0,1]):
            shift = random.randint(-25, 25)
            image = ImageAug.brightness(image, shift)

        if random.choice([0,1]):
            ratio = random.uniform(0.6, 1.8)
            image = ImageAug.scale(image, ratio)
        return image

    def random_pixelate(image):
        # image = ImageAug.random_shape(image)

        h, w = image.shape[:2]
        
        position, position_norm = ImageAug.random_position(h, w)
        block = random.randint(12, max(12, min(position[2:])//3))

        image = ImageAug.pixelate(image, position, block)
        # image = ImageAug.random_quality(image)
        return image, position_norm

    def random_multiple_pixelate(image, max_mosaic=1):
        # image = ImageAug.random_shape(image)

        h, w = image.shape[:2]

        position_all = []
        label_all = []
        # num_mosaic = random.randint(0, max_mosaic)
        num_mosaic = 1

        if not num_mosaic:
            return None, None
        else:
            for i in range(num_mosaic):
                position, position_norm = ImageAug.random_position(h, w)
                position_all.append(position_norm)
                block = random.randint(12, max(12, min(position[2:])//3))
                image = ImageAug.pixelate(image, position, block)
            # image = ImageAug.random_quality(image)
        return image, position_all