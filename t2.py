import pandas as pd

df = pd.read_csv("combined_choice_data_v2.csv", low_memory=False)
mah_df = df[df['Market'] == 'Maharashtra']
print("FC distribution:", mah_df['FC'].value_counts())
print("AH distribution:", mah_df['AH'].value_counts())
print("VAS distribution:", mah_df['VAS'].value_counts())
print("Credit distribution:", mah_df['Credit'].value_counts())
print("Price variation:", mah_df['Price'].nunique(), "std:", mah_df['Price'].std())
print("Sample size:", len(mah_df), "None choices:", len(mah_df[mah_df['chosen_profile'] == 99]))

encoded = pd.get_dummies(mah_df[['FC', 'AH', 'VAS', 'Credit']], drop_first=True)
print(encoded.corr())