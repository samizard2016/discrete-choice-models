import pandas as pd
import numpy as np

df = pd.read_csv("cattle_feed_data_final.csv")
df1 = df[df["Chosen"]==1]
df1['Total'] = 1

cts = []
for var in ["Brand","CP","FC","AH","VAS","Credit","herd_size","cattle_type"]:
    varx = df1[var].apply(str)
    ct = pd.crosstab(varx,df1["State"])
    indx = ct.index
    indx = [f"{var}-{item}" for item in indx]
    ct = ct.values
    ctb = pd.crosstab(df1['Total'],df1["State"]).values
    ct = ct/ctb
    df_ct = pd.DataFrame(ct,index=indx)
    cts.append(df_ct)

df_cts = pd.concat(cts,axis=0)
df_cts.columns = df1["State"].unique()
df_cts.to_excel("Feature distributions_state.xlsx")