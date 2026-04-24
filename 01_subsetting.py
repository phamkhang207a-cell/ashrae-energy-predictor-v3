import pandas as pd
import os

# --- Cấu hình đường dẫn ---
path_raw = r'D:\ashrae-energy-prediction\data\raw'
path_proc = r'D:\ashrae-energy-prediction\data\processed'

# Tạo thư mục processed nếu chưa có
os.makedirs(path_proc, exist_ok=True)

print("--- Bắt đầu lọc dữ liệu (Subsetting) ---")

# 1. Load Metadata & Chọn Building thuộc Site 0, 1, 2
df_meta = pd.read_csv(os.path.join(path_raw, 'building_metadata.csv'))
selected_buildings = df_meta[df_meta['site_id'].isin([0, 1, 2])]['building_id'].unique()

# 2. Load Train data (chỉ lấy cột cần thiết)
train_cols = ['building_id', 'meter', 'timestamp', 'meter_reading']
df_train = pd.read_csv(os.path.join(path_raw, 'train.csv'), usecols=train_cols)

# 3. Lọc: Site 0,1,2 + electricity (meter=0)
subset = df_train[(df_train['building_id'].isin(selected_buildings)) & (df_train['meter'] == 0)].copy()

# 4. CHUYỂN DATETIME VÀ LỌC THEO THÁNG (Tháng 5 -> Tháng 10)
subset['timestamp'] = pd.to_datetime(subset['timestamp'])
subset = subset[(subset['timestamp'].dt.month >= 5) & (subset['timestamp'].dt.month <= 10)]

# 5. Lưu lại dưới dạng Parquet (Tối ưu I/O và dung lượng)
subset.to_parquet(os.path.join(path_proc, 'train_subset.parquet'), index=False)
print(f"Hoàn thành! Subset có {len(subset)} dòng. Đã lưu file train_subset.parquet")