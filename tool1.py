import pandas as pd
df = pd.read_excel("CBC_Data_Final_09Jun25.xlsx")
# unique_profs = df['profiles_presented'].unique()
# print(f"Total profiles found {len(unique_profs)}")
# print(f"Total unique respondents:{len(df['respondent_id'].unique())}")

# # df = pd.read_excel("profiles.xlsx")
# # adv_features = df['Advanced_Feature'].unique()
# # tdf = pd.DataFrame({"Advanced_Features":adv_features})
# # tdf.to_excel("adv features.xlsx",index=False)

# df_groups = pd.read_excel("groups.xlsx")
# resp_id = df['respondent_id'].unique()
# segments = df_groups['group'].values

# tdf = pd.DataFrame({"respondent_id":resp_id,'group':segments})
# tdf.to_excel("temp_groups.xlsx",index=False)
print(df.groupby("respondent_id").size())
tasks = df.groupby("respondent_id").size()
tasks.to_excel("task_count.xlsx")