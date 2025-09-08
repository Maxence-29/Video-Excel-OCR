import pandas as pd
import os

file_dir = "./ocr_align/csv_corrected"
file_list = [os.path.join(file_dir, f) for f in sorted(os.listdir(file_dir)) if f.endswith('.csv')]

dfs = [pd.read_csv(f) for f in file_list]

merged_df = pd.concat(dfs, axis=1)

merged_df.to_csv("./merged_table.csv", index=False)
print("[✓] Merge finished, saved to: ./merged_table.csv")
