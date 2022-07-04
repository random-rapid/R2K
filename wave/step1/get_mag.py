import cv2
import numpy as np

# 画像を読み込み、モノクロに変換する
def img2gray(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray_img

# モノクロにした画像を、2値化する（閾値を決めない場合、大津の方法を使う）
def thresh(path, thresh=-1):
    gray = img2gray(path)
    if thresh == -1:
        retval_otsu, dst_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        return dst_otsu, retval_otsu
        
    elif 0 < thresh < 255:
        retval, dst = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return dst, retval
        
    else:
        print("thresh error")

# 高速フーリエ変換後、2次元パワースペクトルを出力する
def fft(img):
    f_uv = np.fft.fft2(img)
    shifted_f_uv = np.fft.fftshift(f_uv)
    magnitude_spectrum2d = 20 * np.log(np.absolute(shifted_f_uv)) #type numpy.ndarray
    return magnitude_spectrum2d