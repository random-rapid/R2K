import numpy as np
from matplotlib import pyplot as plt 
import power_spectrum as sp


#画像のパス
path = r".\pictures\test4az.png"

#抽出する帯域
band_param = [5, 15]

# 必要な形式に画像を変換
img = sp.img2gray(path)
# img, retval = sp.thresh(path)

#2次元パワースペクトル取得
shifted_f = sp.fft(img)
mag = sp.magnitude(shifted_f)

# 動径方向分布の表示 (Radial Distribution Function)
rdf = sp.average_radius(mag, 1)
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
filter_ = sp.band_pass_filter(shifted_f, band_param[0], band_param[1])
filtered_f = np.multiply(shifted_f, filter_)

masked_mag = sp.magnitude(filtered_f + 1)

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
