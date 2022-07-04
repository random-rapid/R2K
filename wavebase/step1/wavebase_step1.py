import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込みモノクロに変換
def img2gray(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray_img

# モノクロ画像を2値化（閾値を決めない場合、大津の方法を使う）
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


# -------------------- メイン --------------------

#画像のパス
path = r".\test4az.png"

band_param = [5, 15]

# 必要な形式に画像を変換
img = img2gray(path)
# img, retval = thresh(path)

#2次元パワースペクトル取得
shifted_f = fft(img)
mag = magnitude(shifted_f)

# 動径方向分布の表示 (Radial Distribution Function)
rdf = average_radius(mag, 1)
rdf_max = max(rdf)
rdf_min = min(rdf)
rdf_margin = (rdf_max - rdf_min) // 10
rdf_begin = rdf_min - rdf_margin
rdf_end = rdf_max + rdf_margin

# 1次元スペクトルのx軸を作成
x = [i for i in range(len(rdf))]
x_max = max(x)
x_min = min(x)
x_margin = (x_max - x_min) // 10
x_begin = x_min - x_margin
x_end = x_max + x_margin

# バンドパスフィルターでマスク
filter_ = band_pass_filter(shifted_f, band_param[0], band_param[1])
filtered_f = np.multiply(shifted_f, filter_)

masked_mag = magnitude(filtered_f + 1)

# マスクされた2次元パワースペクトルを逆FFT
f_ishift = np.fft.ifftshift(filtered_f)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# プロット
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0][0].imshow(img, 'gray')
ax[0][0].set_xticks([])
ax[0][0].set_yticks([])

ax[0][1].imshow(mag, 'gray')
ax[0][1].set_xticks([])
ax[0][1].set_yticks([])

ax[0][2].plot(x, rdf)
ax[0][2].set_xlim(x_begin, x_end)
ax[0][2].set_ylim(rdf_begin, rdf_end)

ax[1][0].imshow(img_back, 'gray')
ax[1][0].set_xticks([])
ax[1][0].set_yticks([])

ax[1][1].imshow(masked_mag, 'gray')
ax[1][1].set_xticks([])
ax[1][1].set_yticks([])

ax[1][2].plot(x[band_param[0]:band_param[1]], rdf[band_param[0]:band_param[1]])
ax[1][2].set_xlim(x_begin, x_end)
ax[1][2].set_ylim(rdf_begin, rdf_end)

plt.tight_layout()
plt.show()
plt.close()