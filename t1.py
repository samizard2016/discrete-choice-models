import pandas as pd
def main():
    df = pd.read_excel("data_final.xlsx")
    df["valid"] = df['profiles_presented'].apply(lambda x: False if "14" in x and "15" in x else True)
    df.to_excel("data checked.xlsx",index=False)
    print(df.head())
    print("Done")
    
if __name__=="__main__":
    main()
