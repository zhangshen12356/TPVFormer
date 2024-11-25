import numpy as np
import cv2


def impad_to_multiple(img, divisor):
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    shape = (pad_h, pad_w)
    width = max(shape[1] - img.shape[1], 0)  # 0
    height = max(shape[0] - img.shape[0], 0)  # 28
    padding = (0, 0, width, height)
    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(img, padding[1], padding[3], padding[0], padding[2],  border_type['constant'], value=0)
    return img

if __name__ == "__main__":
    img_path = "/home/zs/Code/TPVFormer_new/out/tpv_occupancy/frames/4509/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984245412460.jpg"
    image = cv2.imread(img_path)
    iim = impad_to_multiple(image, 32)
    cv2.imwrite("111.jpg", iim)