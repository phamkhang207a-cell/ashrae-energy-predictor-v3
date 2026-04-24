import pandas as pd
import os

# --- Cấu hình đường dẫn ---
path_raw = r'D:\ashrae-energy-prediction\data\raw'
path_proc = r'D:\ashrae-energy-prediction\data\processed'

print("Đang load dữ liệu...")
# (1) Bảng train đọc từ parquet nên timestamp đã tự động là datetime
train = pd.read_parquet(os.path.join(path_proc, 'train_subset.parquet'))
meta = pd.read_csv(os.path.join(path_raw, 'building_metadata.csv'))
weather = pd.read_csv(os.path.join(path_raw, 'weather_train.csv'))

# (2) Tiền xử lý weather: Ép kiểu timestamp trước khi merge
weather['timestamp'] = pd.to_datetime(weather['timestamp'])

# (3) Merge: Kết hợp thông tin
df = train.merge(meta, on='building_id', how='left')
df = df.merge(weather, on=['site_id', 'timestamp'], how='left')

# (4) Cleaning cơ bản: Nội suy missing value cho CÁC cột thời tiết quan trọng
weather_cols_to_fill = ['air_temperature', 'dew_temperature', 'wind_speed']
print(f"Đang xử lý missing values cho thời tiết: {weather_cols_to_fill}...")

for col in weather_cols_to_fill:
    # Nội suy theo thời gian cho từng site_id riêng biệt
    df[col] = df.groupby('site_id')[col].transform(lambda x: x.interpolate(limit_direction='both'))

# (5) Lưu kết quả cuối cùng ra Parquet để dùng cho Notebook
df.to_parquet(os.path.join(path_proc, 'train_merged.parquet'), index=False)
print("Merge và Clean hoàn tất! Đã lưu file train_merged.parquet")