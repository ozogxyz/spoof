import random

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import UnivariateSpline
from torchvision.transforms import Compose as TorchvisionCompose
from torchvision.transforms import Lambda
from torchvision.transforms import functional as TVF


def adjust_brightness(src, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        src (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    dst = np.array(
        TVF.adjust_brightness(Image.fromarray(src), brightness_factor)
    )
    return dst


def adjust_contrast(src, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        src (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    dst = np.array(TVF.adjust_contrast(Image.fromarray(src), contrast_factor))
    return dst


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy ndarray: Hue adjusted image.
    """
    dst = np.array(TVF.adjust_hue(Image.fromarray(img), hue_factor))
    return dst


def adjust_gamma(img, gamma, gain=1):
    r"""Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
    See `Gamma Correction`_ for more details.
    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")
    # from here
    # https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python/41061351
    dst = np.array(TVF.adjust_gamma(Image.fromarray(img), gamma, gain))
    return dst


def adjust_saturation(img, factor=0.0):
    """
    Image saturation adjustment.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        factor (float): real number in [-1,1], saturation (S-channel in HSV model) is varied according to it.
        -1: grayscale, 0: no change, 1: unnaturally colorful img
    """
    dst = np.array(TVF.adjust_saturation(Image.fromarray(img), factor))
    return dst


def create_LUT_8UC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))


def adjust_temperature(img, factor=1):
    """Adjust color temperature of BGR image by changing color histogram of B and R channels.

    The image temperature is changed by modifying image color histogram.
     Source code is inspired by:
     http://www.askaswiss.com/2016/02/how-to-manipulate-color-temperature-opencv-python.html

    hue_factor is the amount of histogram center shift and must be in the
    interval [-1, 1].
    Args:
        img (np.ndarray): np.ndarray to be adjusted.
        factor (float):  How much to shift the R and B histogram channels. Should be in
            [-1, 1]. 1 == 'warm' colors, -1 == 'cold' colors.
    Returns:
        np.ndarray: Temperature adjusted image.
    """
    if not (-1 <= factor <= 1):
        raise ValueError(f"{factor} is not in [-1, 1].")

    if factor == 0:
        return img

    inp = [0, 64, 128, 192, 256]
    dest_warm = [0, 70, 140, 210, 256]
    dest_cold = [0, 30, 80, 120, 192]

    abs_factor = abs(factor)
    delta = np.array(dest_warm) - np.array(inp)
    dest_warm = (np.array(inp) + np.array(delta) * abs_factor).tolist()
    delta = np.array(dest_cold) - np.array(inp)
    dest_cold = (np.array(inp) + np.array(delta) * abs_factor).tolist()

    incr_ch_lut = create_LUT_8UC1(inp, dest_warm)
    decr_ch_lut = create_LUT_8UC1(inp, dest_cold)

    if factor > 0:
        incr_ch_lut, decr_ch_lut = decr_ch_lut, incr_ch_lut

    # factor > 0 --> make warm --> increase R and desrease B.
    img[:, :, 2] = cv2.LUT(img[:, :, 2], incr_ch_lut).astype(np.uint8)
    img[:, :, 0] = cv2.LUT(img[:, :, 0], decr_ch_lut).astype(np.uint8)

    # factor > 0 --> make warm --> increase saturation, or vise-versa
    hsv_lut = incr_ch_lut if factor < 0 else decr_ch_lut
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = cv2.LUT(img_hsv[:, :, 1], hsv_lut).astype(np.uint8)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img


class RandomHorizontalFlip:
    """Horizontally flip the given np.ndarray randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        tags (tuple of str): names of image data keys from sample dict
    """

    def __init__(self, p=0.5, tags=("image",)):
        self.p = p
        self.tags = tags

    def __call__(self, sample_dict):
        to_flip = random.random() < self.p
        common_tags = set(self.tags).intersection(set(sample_dict.keys()))
        for tag in common_tags:
            if to_flip:
                sample_dict[tag] = np.fliplr(sample_dict[tag].copy())

        return sample_dict

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class ColorJitterCV:
    """Randomly change the brightness, contrast, color temperature, saturation or apply gamma transform to an image.
    Each arg is list of length 1 or 2; if passed list of length 1 then factors are calculated as follows:

    Args:
        brightness (list with length either 1 or 2): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness[0]), 1 + brightness[0]].
        contrast (list with length either 1 or 2): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast[0]), 1 + contrast[0]].
        saturation (list with length either 1 or 2): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation[0]), 1 + saturation[0]].
        hue(list with length either 1 or 2): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue[0], hue[0]]. Should be >=0 and <= 0.5.

        gamma(list with length either 1 or 2): How much to jitter gamma. gamma is chosen uniformly from
            [1-gamma[0], 1+gamma[0]]. Should be >=0 and <= 0.5.
        temp(list with length either 1 or 2): How much to jitter color temperature. Value is chosen uniformly from
            [-temp[0], temp[0]]. Should be >=0 and <= 1.
        p(float from 0.0 to 1.0): probability of applying color jitter, default: 0.75
    If list of length 2 is passed (e.g. brightness=[0.9, 1.5]) then this exact (except clipping) interval is used to
    pick from in random uniform fashion.
    Clipping boundaries for intervals are:
        brightness in (0, inf)
        contrast in (0, inf)
        saturation in (0, inf)
        hue in (-0.5, 0.5)
        gamma in (0.5, 1.5)
        gain in (0.5, 1.5)
        temp in (-1, 1)
        sharpness in (-inf, inf)
    """

    def __init__(
        self,
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0,
        gamma=0,
        # gain=0,
        temp=0,
        p=0.75,
        tags=("image",),
    ):
        self.brightness = (
            brightness
            if self.check_param(brightness)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if self.check_param(contrast)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if self.check_param(saturation)
            else [-saturation, saturation]
        )
        self.hue = hue if self.check_param(hue) else [-hue, hue]
        self.gamma = (
            gamma if self.check_param(gamma) else [1 - gamma, 1 + gamma]
        )
        # self.gain = gain if self.check_param(gain) else [1 - gain, 1 + gain]
        self.temp = temp if self.check_param(temp) else [-temp, temp]

        # Proper interval clippings:
        self.brightness = np.clip(self.brightness, 0, None)
        self.contrast = np.clip(self.contrast, 0, None)
        self.hue = np.clip(self.hue, -0.5, 0.5)
        self.gamma = np.clip(self.gamma, 0.5, 1.5)
        # self.gain = np.clip(self.gain, 0.5, 1.5)
        self.temp = np.clip(self.temp, -1, 1)

        # helper parameters
        self.p = p
        self.tags = tags

    @staticmethod
    def check_param(param):
        return hasattr(param, "__len__") and len(param) == 2

    def __repr__(self):
        format_string = f"{self.__class__.__name__}("
        format_string += f"brightness={self.brightness},"
        format_string += f"contrast={self.contrast},"
        format_string += f"saturation={self.saturation},"
        format_string += f"hue={self.hue},"
        format_string += f"gamma={self.gamma},"
        # format_string += f"gain={self.gain},"
        format_string += f"temp={self.temp}"
        format_string += ")"
        return format_string

    @staticmethod
    def get_params(brightness, contrast, saturation, hue, gamma, temp):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        # handling gamma correction / brightness params
        gamma_factor = 1
        if not np.allclose(gamma, 1):
            gain_factor = 1
            gamma_factor = np.clip(
                random.uniform(gamma[0], gamma[1]), 0.5, 1.5
            )
            transforms.append(
                Lambda(
                    lambda img: adjust_gamma(
                        np.array(img), gamma_factor, gain_factor
                    )
                )
            )
        if not np.allclose(brightness, 1):
            if gamma_factor < 1 and brightness[1] > 1:
                brightness_factor = random.uniform(1, brightness[1])
            elif gamma_factor > 1 and brightness[0] < 1:
                brightness_factor = random.uniform(brightness[0], 1)
            elif gamma_factor == 1:
                brightness_factor = random.uniform(
                    brightness[0], brightness[1]
                )
            else:
                brightness_factor = 1
            transforms.append(
                Lambda(lambda img: adjust_brightness(img, brightness_factor))
            )

        if not np.allclose(contrast, 1):
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: adjust_contrast(img, contrast_factor))
            )
        if not np.allclose(saturation, 0):
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: adjust_saturation(img, saturation_factor))
            )
        if not np.allclose(temp, 0):
            temp_factor = random.uniform(temp[0], temp[1])
            transforms.append(
                Lambda(lambda img: adjust_temperature(img, temp_factor))
            )
        if not np.allclose(hue, 1):
            hue_factor = float(
                np.clip(random.uniform(hue[0], hue[1]), -0.5, 0.5)
            )
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = TorchvisionCompose(transforms)

        return transform

    def __call__(self, sample_dict):
        transform_func = self.get_params(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue,
            self.gamma,
            # self.gain,
            self.temp,
            # self.sharpness,
        )
        if np.random.random() < self.p:
            for tag in self.tags:
                sample_dict[tag] = transform_func(sample_dict[tag].squeeze())

        return sample_dict


class RandomGaussianBlur:
    """
    Args:
        radius (int, (int, int)): If radius is a single int, the range
            will be (3, radius). Default: (3, 8).
        p (float, from 0.0 to 1.0): probability of applying blur, default: 0.5
        tags (list of str): list of sample dict keys to apply image transform to
    """

    def __init__(
        self,
        radius=(1, 3),
        tags=("image",),
        p=0.5,
    ):
        self.p = p
        self.tags = tags
        assert (
            type(radius) is tuple
            and len(radius) == 2
            or type(radius) is int
            and radius > 1
        ), "Wrong input: radius"

        if isinstance(radius, float):
            self.radius = (1, radius)
        else:
            self.radius = radius

    def get_params(self):
        return {
            "sigmaX": self.radius[0]
            + random.random() * (self.radius[1] - self.radius[1])
        }

    def __repr__(self):
        format_string = (
            f"{self.__class__.__name__}(p={self.p}, radius={self.radius}"
        )
        return format_string

    def __call__(self, sample_dict):
        params = self.get_params()
        if random.random() > self.p:
            for tag in self.tags:
                sample_dict[tag] = cv2.GaussianBlur(
                    sample_dict[tag].copy(), None, **params
                )
        return sample_dict
