import multiprocessing
import sys

import cv2
import numpy as np
from joblib import Parallel, delayed


class SGBMStereo:

    def __init__(self, img_l, img_r, max_disparity, k_size):
        """
        Class Init
        :param img_l: left image
        :param img_r: right image
        :param max_disparity: max disparity
        :param k_size: window size
        """
        self.imgL = img_l
        self.imgR = img_r
        self.k_size = k_size
        self.max_disparity = max_disparity

    def compute(self):
        min_disp = 0
        num_disp = self.max_disparity - min_disp
        channels = 1
        print("Initialising stereo object...")
        left_matcher = cv2.StereoSGBM_create(minDisparity=min_disp,
                                             numDisparities=num_disp,
                                             blockSize=self.k_size,
                                             P1=8000,#* channels * self.k_size ** 2,
                                             P2=32000,# * channels * self.k_size ** 2,
                                             disp12MaxDiff=1,
                                             uniquenessRatio=10,
                                             speckleWindowSize=100,
                                             speckleRange=32)

        print('Computing disparity...')
        # compute disparity and convert type
        disp = left_matcher.compute(self.imgL, self.imgR).astype(np.float32) / 16.0
        # normalize
        # disp_norm = np.uint8(disp * (255 / num_disp))
        # dont normalize
        disp_norm = np.uint8(disp)

        disp_norm[disp == -1] = 0

        return disp_norm


class SAD:

    def __init__(self, img_l, img_r, max_disp, k_size, normalized=False):
        n = 1
        self.imgR = img_r[::n, ::n]
        self.imgL = img_l[::n, ::n]

        self.max_disparity = max_disp
        self.kernel_size = k_size

        self.normalized = normalized

    def compute_parallel(self):

        num_cores = multiprocessing.cpu_count()

        # Load in both images, assumed to be RGBA 8bit per channel images
        left = np.asarray(self.imgL)
        right = np.asarray(self.imgR)
        h, w = left.shape  # assume that both images are same size

        # Depth (or disparity) map
        depth = np.zeros((w, h), np.uint8)
        depth.shape = h, w

        kernel_half = int(self.kernel_size / 2)

        max_sad = 255 ** 2 * self.kernel_size ** 2

        def calc_line(x, y_curr, kh, l_img, r_img, max_d, norm):

            best_offset = 0
            prev_sad = max_sad

            window_left = np.int16(l_img[y_curr - kh:y_curr + kh + 1, x - kh:x + kh + 1])

            if norm:
                left_std = np.std(window_left)
                left_mean = np.mean(window_left)
            else:
                left_std, left_mean = None, None

            for offset in range(max_d):

                if x - kh - offset <= 0:
                    offset = 0

                # v and u are the x,y of our local window search, used to ensure a good
                # match- going by the squared differences of two pixels alone is insufficient,
                # we want to go by the squared differences of the neighbouring pixels too

                # print(x, y, ((x-kernel_half)-offset),((x+kernel_half)-offset))
                window_right = np.int16(r_img[y_curr - kh:y_curr + kh + 1, (x - kh) - offset:(x + kh) - offset + 1])

                if norm:
                    sad = np.sum(np.abs(((window_left - left_mean) / left_std) - (
                            (window_right - np.mean(window_right)) / np.std(window_right))))
                else:
                    sad = np.sum(np.abs(window_left - window_right))

                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if sad < prev_sad:
                    prev_sad = sad
                    best_offset = offset

            # set depth output for this x,y location to the best match
            return best_offset

        for y in range(kernel_half, h - kernel_half):
            depth[y, kernel_half: w - kernel_half] = np.array(Parallel(n_jobs=num_cores)(
                delayed(calc_line)(x, y, kernel_half, left, right, self.max_disparity, self.normalized) for x in
                range(kernel_half, w - kernel_half)))

            cv2.imshow("SAD", np.uint8(depth))
            cv2.waitKey(1)

        return np.uint8(depth)


class NCC:

    def __init__(self, img_l, img_r, max_disp, k_size):
        n = 1
        self.imgR = img_r[::n, ::n]
        self.imgL = img_l[::n, ::n]

        self.max_disparity = max_disp
        self.kernel_size = k_size

        self.histo = list()

    def compute_parallel(self):

        from joblib import Parallel, delayed
        import multiprocessing

        num_cores = multiprocessing.cpu_count()

        # Load in both images, assumed to be RGBA 8bit per channel images
        left = np.asarray(self.imgL)
        right = np.asarray(self.imgR)
        h, w = left.shape  # assume that both images are same size

        # Depth (or disparity) map
        depth = np.zeros((w, h), np.uint8)
        depth.shape = h, w

        kernel_half = int(self.kernel_size / 2)

        def calc_line(x, y_curr, kh, l_img, r_img, max_d):

            np.seterr(divide='ignore', invalid='ignore')

            best_offset = 0
            prev_ncc = 0

            window_left = np.int16(left[y_curr - kernel_half:y_curr + kernel_half + 1, x - kernel_half:x + kernel_half + 1])
            left_top_factor = window_left - np.mean(window_left)
            left_std = np.std(window_left)

            for offset in range(self.max_disparity):

                if x - kernel_half - offset <= 0:
                    offset = 0

                window_right = np.int16(right[y_curr - kernel_half:y_curr + kernel_half + 1,
                                        (x - kernel_half) - offset:(x + kernel_half) - offset + 1])
                ncc = np.mean(
                    (left_top_factor * (window_right - np.mean(window_right))) / (np.std(window_right) * left_std))

                # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if abs(ncc) > abs(prev_ncc):
                    prev_ncc = ncc
                    best_offset = offset

            return best_offset

        for y in range(kernel_half, h - kernel_half):
            depth[y, kernel_half: w - kernel_half] = np.array(Parallel(n_jobs=num_cores)(
                delayed(calc_line)(x, y, kernel_half, left, right, self.max_disparity) for x in
                range(kernel_half, w - kernel_half)))

            cv2.imshow("NCC", np.uint8(depth))
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        return np.uint8(depth)


# Getting Arguments
imgl_path = sys.argv[1]
imgr_path = sys.argv[2]
img_out_path = sys.argv[3]
kernel_size = sys.argv[4]
max_disparity_range = sys.argv[5]
method = sys.argv[6]

# Reading images
imgl = cv2.imread(imgl_path, 0)
imgr = cv2.imread(imgr_path, 0)

# selection method and run stereo algorithm
if method == "NCC":
    result_img = NCC(imgl, imgr, int(max_disparity_range), int(kernel_size)).compute_parallel()
elif method == "SAD":
    result_img = SAD(imgl, imgr, int(max_disparity_range), int(kernel_size)).compute_parallel()
elif method == "SGBM":
    result_img = SGBMStereo(imgl, imgr, int(max_disparity_range), int(kernel_size)).compute()
else:
    result_img = None
    print("Method unknown")
    exit(1)
# save image
cv2.imwrite(img_out_path, result_img)
