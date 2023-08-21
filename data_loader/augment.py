from io import BytesIO

import numpy as np
import skimage as sk
from PIL import Image, ImageOps
import cv2
from PIL import Image, ImageOps
from wand.image import Image as WandImage


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        coords = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        coords = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(coords, coords)
    aliased_disk = np.asarray((x ** 2 + y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


'''
    PIL resize (W,H)
    cv2 image is BGR
    PIL image is RGB
'''


class DefocusBlur(object):
    def __init__(self, rng=None, mag=-1, prob=1., seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        # c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
        c = [(2, 0.1), (3, 0.1), (4, 0.1)]  # , (6, 0.5)] #prev 2 levels only
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        img = np.asarray(img) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
            n_channels = 3
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(n_channels):
            channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
        channels = np.asarray(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        # if isgray:
        #    img = img[:,:,0]
        #    img = np.squeeze(img)

        img = np.clip(channels, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img


class MotionBlur(object):
    def __init__(self, rng=None, mag=-1, prob=1., seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        # c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]
        c = [(10, 3), (12, 4), (14, 5)]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        output = BytesIO()
        img.save(output, format='PNG')
        img = WandImage(blob=output.getvalue())

        img.motion_blur(radius=c[0], sigma=c[1], angle=self.rng.uniform(-45, 45))
        img = cv2.imdecode(np.frombuffer(img.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img.astype(np.uint8))

        if isgray:
            img = ImageOps.grayscale(img)

        return img

class Brightness(object):
    
    def __init__(self, rng=None, mag=-1, prob=1., seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        # W, H = img.size
        # c = [.1, .2, .3, .4, .5]
        c = [.1, .2, .3]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = sk.color.rgb2hsv(img)
        img[:, :, 2] = np.clip(img[:, :, 2] + c, 0, 1)
        img = sk.color.hsv2rgb(img)

        # if isgray:
        #    img = img[:,:,0]
        #    img = np.squeeze(img)

        img = np.clip(img, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img
        # if isgray:
        # if isgray:
        #    img = color.rgb2gray(img)

        # return Image.fromarray(img.astype(np.uint8))


class JpegCompression(object):
    
    def __init__(self, rng=None, mag=-1, prob=1., seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        # c = [25, 18, 15, 10, 7]
        c = [25, 18, 15]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]
        output = BytesIO()
        img.save(output, 'JPEG', quality=c)
        return Image.open(output)


class Pixelate(object):
    def __init__(self, rng=None, mag=-1, prob=1., seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.mag = mag
        self.prob = prob

    def __call__(self, img):
        if self.rng.uniform(0, 1) > self.prob:
            return img

        w, h = img.size
        # c = [0.6, 0.5, 0.4, 0.3, 0.25]
        c = [0.6, 0.5, 0.4]
        if self.mag < 0 or self.mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = self.mag
        c = c[index]
        img = img.resize((int(w * c), int(h * c)), Image.BOX)
        return img.resize((w, h), Image.BOX)
