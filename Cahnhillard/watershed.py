import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import pandas as pd
import os
from datetime import datetime
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation
# --- Process Parameters ---
magnification = 300
threhold_maginification = 1.30

# SNOW parameters (Default values generally work well)
# r_max: Maximum radius of the structuring element (adjust based on pore size)
# sigma: Smoothing parameter for the distance transform
snow_r_max = 20
snow_sigma = 0.01
# --------------------------

# Create output directories
output_dir = 'output_data'
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_subdir = os.path.join(output_dir, timestamp)
os.makedirs(output_subdir, exist_ok=True)

# Load the image
image_path = '/Users/fukushimatakeru/IDE/Github/poromet/input/専1_0.5Pa系/array/TGB0.5Pa系/TGB0.5Pa系630C/202408080.5Pa 50W TGB 1um Si,620C 24h,8N-NaOH 30C 4h,1e-5N-HCl 30C 4h/0.5Pa250WTGB1umSi620C24h8N-NaOH30C4h1e-5N-HCl30C1.5h_m006.jpg'
image = skimage.io.imread(image_path, as_gray=True)
image_height, image_width = image.shape[:2]

# --- Pixel Size Definition ---
pixel_data = {
    (2560, 1920): {
        10: {'px_per_nm': 1008 / 5000, 'unit': 'nm'},
        20: {'px_per_nm': 807 / 2000, 'unit': 'nm'},
        50: {'px_per_nm': 1022 / 1000, 'unit': 'nm'},
        100: {'px_per_nm': 1018 / 500, 'unit': 'nm'},
    },
    (1280, 960): {
        200: {'px_per_nm': 406 / 200, 'unit': 'nm'},
        300: {'px_per_nm': 303 / 100, 'unit': 'nm'},
    },
    (554, 416): {
        200: {'px_per_nm': 174 / 200, 'unit': 'nm'},
    }
}

resolution_pixel_data = pixel_data.get((image_width, image_height))
if resolution_pixel_data is None:
    raise ValueError(f"Pixel data for image resolution {image_width}x{image_height} is not defined.")

magnification_pixel_data = resolution_pixel_data.get(magnification)
if magnification_pixel_data is None:
    raise ValueError(f"Pixel data for magnification {magnification} and resolution {image_width}x{image_height} is not defined.")

px_per_nm = magnification_pixel_data['px_per_nm']
unit = magnification_pixel_data['unit']
pixel_size_nm = 1/px_per_nm

# --- Image Processing ---

# Thresholding
thresh = threshold_otsu(image)
adjusted_thresh = thresh * threhold_maginification
thresholded_image = image < adjusted_thresh  # True = Pore

print(f"Image loaded. Size: {image.shape}. Starting SNOW partitioning...")

# --- 1. SNOW Partitioning (Corrected based on documentation) ---
# ps.filters.snow_partitioning を使用します
snow_output = ps.filters.snow_partitioning(
    im=thresholded_image,
    r_max=snow_r_max,
    sigma=snow_sigma
)

# snow_output contains: .im (binary), .dt (distance transform), .regions (labeled regions)
regions = snow_output.regions

print("Creating debug visualization (debug_snow_steps.png)...")
fig, ax = plt.subplots(2, 2, figsize=(12, 12))

# (1) Binary Image: アルゴリズムに入力された白黒画像
ax[0, 0].imshow(snow_output.im, cmap='gray', origin='lower')
ax[0, 0].set_title("1. Binary Image (Input)")

# (2) Distance Transform: 「壁からの距離」を示した地図
# 明るい場所＝細孔の中心（広い場所）。ここがくっついていると分割されません。
ax[0, 1].imshow(snow_output.dt, cmap='viridis', origin='lower')
ax[0, 1].set_title("2. Distance Transform (DT)")

# (3) Peaks: 検出された「細孔の中心点」
# 白い点が「1つの細孔の中に2つ以上」あれば、そこで分割されます。
dt_peak = snow_output.dt.copy()
# ピーク位置を少し太らせて見やすくする
peaks_dilated = binary_dilation(snow_output.peaks > 0, footprint=np.ones((3,3))) 
dt_peak[peaks_dilated > 0] = np.nan # ピーク位置を白抜き（NaN）にする
ax[1, 0].imshow(dt_peak, cmap='viridis', origin='lower')
ax[1, 0].set_title("3. Detected Peaks (White Dots)")

# (4) Segmentation: 最終結果
ax[1, 1].imshow(regions, cmap='nipy_spectral', origin='lower')
ax[1, 1].set_title("4. Final Segmentation")

plt.tight_layout()
debug_filename = os.path.join(output_subdir, 'debug_snow_steps.png')
plt.savefig(debug_filename, dpi=150)
plt.close()
print(f"Debug image saved: {debug_filename}")

# --- 2. Region Analysis (Regionprops) ---
print("Calculating region properties...")

# Calculate properties for each region
props = ps.metrics.regionprops_3D(regions)

# ドキュメントにある props_to_DataFrame を使用してデータを整理
df = ps.metrics.props_to_DataFrame(props)

# --- 3. Add Calculated Metrics to DataFrame ---

# Convert Equivalent Diameter from pixels to nm
# equivalent_diameter_area は「同じ面積を持つ円の直径」
df['diameter_nm'] = df['equivalent_diameter_area'] * pixel_size_nm

# Calculate Circularity (2D Sphericity approximation)
# Circularity = 4 * pi * Area / Perimeter^2
# 1.0 = Perfect Circle, < 0.5 = Elongated or Irregular
# Note: 'area' and 'perimeter' are in pixels
df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)

# Handle cases where perimeter is 0 (single pixel pores) to avoid NaN
df['circularity'] = df['circularity'].fillna(0)

# 統計量の計算
total_pores = len(df)
avg_diameter = df['diameter_nm'].mean()
median_diameter = df['diameter_nm'].median()
std_diameter = df['diameter_nm'].std()
avg_circularity = df['circularity'].mean()

print(f"Analysis Complete. Found {total_pores} pores.")

# --- 4. Save Results ---

# Text Report
txt_filename = os.path.join(output_subdir, 'pore_analysis_report.txt')
with open(txt_filename, 'w') as f:
    f.write("Pore Size & Shape Analysis (Watershed/SNOW Method)\n")
    f.write("==================================================\n")
    f.write(f"Image: {os.path.basename(image_path)}\n")
    f.write(f"Resolution: {image_width}x{image_height}\n")
    f.write(f"Pixel Size: {pixel_size_nm:.4f} nm/px\n")
    f.write("\nSummary Statistics:\n")
    f.write(f"Total Pores Detected: {total_pores}\n")
    f.write(f"Average Pore Diameter: {avg_diameter:.4f} nm\n")
    f.write(f"Median Pore Diameter:  {median_diameter:.4f} nm\n")
    f.write(f"Std Dev Diameter:      {std_diameter:.4f} nm\n")
    f.write(f"Average Circularity:   {avg_circularity:.4f} (1.0=Circle)\n")
    f.write("\nNote: 'Diameter' is the equivalent diameter of a circle with the same area.\n")
    f.write("'Circularity' indicates shape irregularity (low value = irregular/elongated).\n")

print(f"Report saved to: {txt_filename}")

# CSV Export (All Data)
csv_filename = os.path.join(output_subdir, 'all_pores_data.csv')
# 必要なカラムだけ選んで保存（全部だと多すぎる場合があるため）
cols_to_save = ['label', 'diameter_nm', 'circularity', 'area', 'perimeter', 'solidity', 'eccentricity']
# 存在するカラムのみを選択
available_cols = [c for c in cols_to_save if c in df.columns]
df[available_cols].to_csv(csv_filename, index=False)
print(f"CSV data saved to: {csv_filename}")

# --- 5. Visualization ---

# Histogram: Pore Size Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['diameter_nm'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(avg_diameter, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_diameter:.1f} nm')
plt.xlabel('Equivalent Pore Diameter (nm)')
plt.ylabel('Count')
plt.title('Pore Size Distribution (Watershed Segmentation)')
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_subdir, 'hist_size_distribution.png'))
plt.close()

# Histogram: Shape (Circularity) Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['circularity'], bins=50, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('Circularity (1.0 = Perfect Circle)')
plt.ylabel('Count')
plt.title('Pore Shape Distribution\n(Values < 0.6 indicate irregular or elongated pores)')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_subdir, 'hist_shape_distribution.png'))
plt.close()

# --- 6. Segmentation Map Visualization ---
print("Creating random color segmentation map...")

# 1. ラベルの最大値を取得
max_label = regions.max()

# 2. ランダムな色を作成 (行数=ラベル数+1, 列数=3(RGB))
# np.random.rand は 0.0~1.0 の乱数を生成します
np.random.seed(42) # 毎回同じランダム色になるように種を固定
random_colors = np.random.rand(max_label + 1, 3)

# 3. 背景(ID=0)は必ず「黒」にする
random_colors[0] = [0, 0, 0]

# 4. ラベル画像を、色の配列を使ってRGB画像に変換
# regionsの数字(ID)に対応する色を random_colors から引っ張ってきます
colored_regions = random_colors[regions]

plt.figure(figsize=(12, 10))
# imshowに直接 RGB画像 を渡すと、カラーマップ設定(cmap)は無視してその色で表示されます
plt.imshow(colored_regions)
plt.title(f'Random Color Segmentation (Detected {total_pores} regions)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_subdir, 'segmentation_random_colors.png'), dpi=300)
plt.close()

# Image: Original Image
plt.figure(figsize=(12, 10))
plt.imshow(image, cmap='gray')
plt.colorbar(label='Intensity')
plt.title('Original Image')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_subdir, 'original_image.png'), dpi=300)
plt.close()


# Image: Thresholded Binary
skimage.io.imsave(os.path.join(output_subdir, 'binary_threshold.png'), (thresholded_image * 255).astype(np.uint8))

print("All processing finished successfully.")