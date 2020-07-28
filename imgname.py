import numpy as np
import os
import cv2
import copy

# copyright @ Hamed Kiani Galoogahi- 2018
# email: hamedkg@gmail.com


class HIP:
    def __init__(self, image=None, path=None):
        self.image = image
        self.path = path
        self.original_image = None
        if os.path.isfile(path):
            image = cv2.imread(path)
            self.image = image
            self.original_image = copy.deepcopy(image)
        else:
            print("the path {} does not exist".format(path))

    def load(self, image=None, path=None):
        self.image = image
        self.original_image = None
        self.path = path
        if os.path.isfile(path):
            image = cv2.imread(path)
            self.image = image
            self.original_image = copy.deepcopy(image)
        else:
            print("the path {} does not exist".format(path))

    def reset(self):
        self.image = copy.deepcopy(self.original_image)

    def imshow(self, tag="image"):
        if self.image is not None:
            cv2.imshow(tag, self.image)
        else:
            print("the image is not loaded!")

    def height(self):
        if self.image is not None:
            return self.image.shape[1]
        else:
            return 0

    def width(self):
        if self.image is not None:
            return self.image.shape[0]
        else:
            return 0

    def channel(self):
        if self.image is not None:
            return self.image.ndim
        else:
            return 0

    def save_as(self, path=None):
        if self.image is None:
            print("there is not image to be saved!")
            return False
        cv2.imwrite(path, self.image)

    def get_image(self):
        return self.image

    def rgb_to_gray(self):
        if self.channel() == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        return self

    def binarize(self, threshold=128):
        if self.channel() == 3:
            self.rgb_to_gray()
        self.image[self.image >= threshold] = 255
        self.image[self.image < threshold] = 0
        return self

    def resize(self, new_size=None, ratio=(0.5, 0.5), interpolation=cv2.INTER_AREA, use_ratio=False):
        if self.image is not None:
            if use_ratio:
                assert(np.all([r >= 0 for r in ratio]))
                tmp_size = (int(ratio[0]*self.image.shape[0]), int(ratio[1]*self.image.shape[1]))
            elif new_size is not None:
                assert(np.all([s >= 0 for s in new_size]))
                tmp_size = new_size
            else:
                print("no resize ration neither a new size is provided.")
        if tmp_size is not None:
            self.image = cv2.resize(self.image, tmp_size, interpolation=interpolation)
        return self

    def crop(self, crop_size):
        # crop_size = [top, left, height, width]
        crop_size[0] = max(0, crop_size[0])
        crop_size[0] = min(self.image.shape[0] - 1, crop_size[0])
        crop_size[1] = max(0, crop_size[1])
        crop_size[1] = min(self.image.shape[1] - 1, crop_size[1])
        if (crop_size[0] + crop_size[2] > self.image.shape[0]):
            crop_size[2] = self.image.shape[0] - crop_size[0]
        if (crop_size[1] + crop_size[3] > self.image.shape[1]):
            crop_size[3] = self.image.shape[1] - crop_size[1]
        self.image = self.image[crop_size[0]:crop_size[0]+crop_size[2], crop_size[1]:crop_size[1]+crop_size[3]]
        return self

    def flip(self, mode=0):
        # mode 0 (horizontal), 1(vertical), -1 (both)
        if mode not in set([0, 1, -1]):
            print("the mode must be either horizontal = 0, vertical = 1, both = -1")
            return self
        self.image = cv2.flip(self.image, mode)
        return self

    def rotate(self, degree=0, rotate_center=None):
        row, col = self.image.shape[0], self.image.shape[1]
        if rotate_center is None:
            center = tuple(np.array([row, col]) / 2)
        else:
            map(int, rotate_center)
            center = tuple(rotate_center)
        rot_mat = cv2.getRotationMatrix2D(center, degree, 1.0)
        self.image = cv2.warpAffine(self.image, rot_mat, (col, row))
        return self

    def zero_padd(self, padd_size=[10, 10, 30, 30]):
        padd_size = list(map(int, padd_size))
        assert (np.alltrue([step >= 0 for step in padd_size]))
        # padd_size=[left, right, top, bottom]
        channel = self.channel()
        left_zeros = np.ones((self.image.shape[0], padd_size[0], channel), np.uint8)
        self.image = np.concatenate((left_zeros, self.image), axis=1)

        right_zeros = np.ones((self.image.shape[0], padd_size[1], channel), np.uint8)
        self.image = np.concatenate((self.image, right_zeros), axis=1)

        top_zeros = np.ones((padd_size[2], self.image.shape[1], channel), np.uint8)
        self.image = np.concatenate((top_zeros, self.image), axis=0)

        bottom_zeros = np.ones((padd_size[3], self.image.shape[1], channel), np.uint8)
        self.image = np.concatenate((self.image, bottom_zeros), axis=0)
        return self

    def add_noise(self, noise="salt&pepper"):
        if self.channel() > 2:
            row, col, ch = self.image.shape
        else:
            row, col = self.image.shape
            ch = 0
        if noise == "salt&pepper":
            salt_rate = 0.5
            pepper_rate = 1. - salt_rate
            amount = 0.01
            num_salt = np.ceil(amount * self.image.size * salt_rate)
            num_pepper = np.ceil(amount * self.image.size * pepper_rate)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in self.image.shape]
            coords_p = [np.random.randint(0, i - 1, int(num_pepper))
                        for i in self.image.shape]
            if self.image.ndim == 3:
                self.image[coords[0], coords[1],:] = (255, 255, 255)
                self.image[coords_p[0], coords_p[1], :] = (0, 0, 0)
            else:
                self.image[coords] = 255
                self.image[coords_p] = 0

        elif noise == "gauss":
            mean = 0.5
            var = 0.5
            sigma = var ** 0.5
            if ch:
                gauss = np.random.normal(mean, sigma, (row, col, ch))
            else:
                gauss = np.random.normal(mean, sigma, (row, col))
            gauss.reshape(self.image.shape)
            noisy_image = self.image + gauss
            self.image = noisy_image.astype(np.uint8)
        return self

    def gradient(self):
        image = self.image
        if image.ndim > 2:
            self.rgb_to_gray()
            image = self.image
        dx = np.zeros(shape=image.shape, dtype=np.uint8)
        dy = np.zeros(shape=image.shape, dtype=np.uint8)

        rows = [row + 1 for row in range(image.shape[0] - 2)]
        cols = [col + 1 for col in range(image.shape[1] - 2)]

        for row in rows:
            for col in cols:
                dx[row][col] = abs(-1.0 * image[row][col-1] + 1.0 * image[row][col+1])
                dy[row][col] = abs(-1.0 * image[row-1][col] + 1.0 * image[row+1][col])
        return dx, dy

    def magnitude(self):
        dx, dy = self.gradient()
        mag = np.sqrt(dx**2 + dy**2)
        mag = (mag - np.amin(mag) / (np.amax(mag)) - np.amin(mag))
        return mag.astype(np.uint8)

    def edge(self):
        gx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
        gy = np.transpose(gx)
        image = self.image
        if image.ndim > 2:
            image = self.rgb_to_gray()
        rows = [row + 1 for row in range(image.shape[0] - 2)]
        cols = [col + 1 for col in range(image.shape[1] - 2)]

        edge_image = np.zeros(shape=image.shape, dtype=np.uint8)
        for row in rows:
            for col in cols:
                dx = abs(
                    gx[0][0] * image[row - 1][col - 1] + gx[0][1] * image[row - 1][col] + gx[0][2] * image[row - 1][
                        col + 1]
                    + gx[1][0] * image[row][col - 1] + gx[1][1] * image[row][col] + gx[1][2] * image[row][col + 1]
                    + gx[2][0] * image[row + 1][col] + gx[2][1] * image[row + 1][col] + gx[2][2] * image[row + 1][
                        col + 1])

                dy = abs(
                    gy[0][0] * image[row - 1][col - 1] + gy[0][1] * image[row - 1][col] + gy[0][2] * image[row - 1][
                        col + 1]
                    + gy[1][0] * image[row][col - 1] + gy[1][1] * image[row][col] + gy[1][2] * image[row][col + 1]
                    + gy[2][0] * image[row + 1][col] + gy[2][1] * image[row + 1][col] + gy[2][2] * image[row + 1][
                        col + 1])

                edge_image[row][col] = np.sqrt(dx**2 + dy**2)
        return edge_image

    def get_path(self):
        return self.path
