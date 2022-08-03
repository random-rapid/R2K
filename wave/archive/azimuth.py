import numpy as np


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
    means = [aggregate_in_radius(np.mean, spectrum, radius, radius + radius_gap)
             for radius in range(0, 1000, radius_gap)]
    return means