import cv2
import numpy as np


# 画像を読み込みモノクロに変換
def img2gray(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray_img

# モノクロ画像を2値化（閾値を決めない場合、大津の方法を使う(thresh = -1）
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

# 高速フーリエ変換後、象限入れ替え
def fft(img):
    f = np.fft.fft2(img)
    shifted_f = np.fft.fftshift(f)
    return shifted_f

# 2次元パワースペクトルを出力
def magnitude(shifted_f):
    magnitude_spectrum2d = 20 * np.log(np.absolute(shifted_f))
    return magnitude_spectrum2d

# 複素平面を構成する値の配列を生成(原点は中心)
def complex_plane(width):
    half_width = width // 2
    re = np.array(range(width)) - half_width
    im = - re
    re, im = np.meshgrid(re, im)
    return re + im * 1j

# スペクトルを与えられた角度の範囲内だけ集計
def aggregate_in_angle(agg_fun, spectrum, min_angle, max_angle):
    width = spectrum.shape[0]
    min_radius, max_radius = 0, width // 2
    cp = complex_plane(width)
    cp_mag = np.abs(cp)
    in_radius = np.logical_and(min_radius < cp_mag, cp_mag < max_radius)
    cp_angle = np.angle(cp, deg=True)
    in_angle = np.logical_and(min_angle <= cp_angle, cp_angle < max_angle)
    return agg_fun(spectrum[np.logical_and(in_radius, in_angle)])
  

# 指定角度ごとの平均を求める
def average_angle(spectrum, angle_gap):
    means = [aggregate_in_angle(np.mean, spectrum, angle, angle + angle_gap)
             for angle in range(-180, 180, angle_gap)]
    return means
    
# スペクトルを与えられた半径の範囲内だけ集計
def aggregate_in_radius(agg_fun, spectrum, min_radius, max_radius):
    width = spectrum.shape[0]
    cp = complex_plane(width)
    cp_mag = np.abs(cp)
    in_radius = np.logical_and(min_radius <= cp_mag, cp_mag < max_radius)
    return agg_fun(spectrum[in_radius])
  

# 指定半径ごとの平均を求める
def average_radius(spectrum, radius_gap):
    max_radius_ = spectrum.shape[0] // 2
    means = [aggregate_in_radius(np.mean, spectrum, radius, radius + radius_gap)
             for radius in range(0, max_radius_, radius_gap)]
    return means


# バンドパスフィルター
def band_pass_filter(magnitude, inner_radius, outer_radius):
    width = magnitude.shape[0]
    cp = complex_plane(width)
    cp_mag = np.abs(cp)
    on_band_ring = np.logical_and(inner_radius <= cp_mag, cp_mag <= outer_radius)
    return on_band_ring