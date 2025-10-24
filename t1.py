import pandas as pd

df = pd.read_csv("combined_choice_data_v2.csv", low_memory=False)
mah_df = df[df['Market'] == 'Maharashtra']
print("Unique values in FC:", mah_df['FC'].unique())
print("Unique values in herd_size:", mah_df['herd_size'].unique())
print("Unique values in cattle_type:", mah_df['cattle_type'].unique())
print("FC for None brand:", mah_df[mah_df['Brand'] == 'None']['FC'].value_counts())

mah_df = df[df['Market'] == 'Maharashtra']
print("Unique values in AH:", mah_df['AH'].unique())
print("Unique values in VAS:", mah_df['VAS'].unique())
print("Unique values in Credit:", mah_df['Credit'].unique())