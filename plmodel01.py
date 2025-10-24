import pandas as pd
import numpy as np
import collections
from collections.abc import Iterable
collections.Iterable = Iterable  # Patch for pylogit Python 3.10 compatibility
import pylogit as pl
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import openpyxl

class DiscreteChoiceModel:
    def __init__(self, markets=None, use_pylogit=True):
        self.markets = markets if markets else ['Maharashtra', 'Gujarat', 'Punjab', 'South']
        self.models = {}
        self.coef_dfs = {}
        self.use_pylogit = use_pylogit
    
    def fit(self, df, target='Chosen', features=None):
        if features is None:
            features = ['Price', 'CP', 'FC', 'AH', 'herd_size', 'cattle_type']  # Exclude VAS, Credit
        
        for market in self.markets:
            market_df = df[df['Market'] == market].copy()
            if market_df.empty:
                print(f"Warning: No data for {market}")
                self.coef_dfs[market] = pd.DataFrame(columns=['Coefficient', 'P-value'])
                continue
            
            # Verify columns
            missing_cols = [col for col in features if col not in market_df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} for {market}")
                self.coef_dfs[market] = pd.DataFrame(columns=['Coefficient', 'P-value'])
                continue
            
            # Clean categorical columns
            for col in ['FC', 'AH', 'herd_size', 'cattle_type']:
                market_df[col] = market_df[col].fillna('None').astype(str)
            
            if self.use_pylogit:
                try:
                    # Pylogit setup
                    pylogit_df = market_df.copy()
                    pylogit_df['obs_id'] = pylogit_df['ChoiceID']
                    pylogit_df['alt_id'] = pylogit_df['Brand']
                    pylogit_df['choice'] = pylogit_df[target]
                    
                    # Specification: Treat CP as continuous, others as categorical
                    spec = {
                        'Price': 'all_same',
                        'CP': 'all_same',  # Numeric
                        'FC': 'all_different',
                        'AH': 'all_different',
                        'herd_size': 'all_same',
                        'cattle_type': 'all_same'
                    }
                    varnames = features
                    name_dict = {var: var for var in varnames}
                    
                    # Fit pylogit
                    model = pl.create_choice_model(
                        data=pylogit_df,
                        alt_id_col='alt_id',
                        obs_id_col='obs_id',
                        choice_col='choice',
                        specification=spec,
                        model_type='MNL',
                        names=name_dict
                    )
                    results = model.fit_mle(np.zeros(len(varnames)), print_res=False)
                    coef_df = pd.DataFrame({
                        'Coefficient': results.params,
                        'P-value': results.pvalues
                    }, index=varnames)
                    self.models[market] = results
                    self.coef_dfs[market] = coef_df
                    print(f"Pylogit model fitted for {market}")
                except Exception as e:
                    print(f"Pylogit failed for {market}: {e}. Switching to LogisticRegression.")
                    self.use_pylogit = False
            
            if not self.use_pylogit:
                try:
                    # LogisticRegression setup
                    categorical_cols = [col for col in ['FC', 'AH', 'herd_size', 'cattle_type'] if col in market_df.columns]
                    market_df_encoded = pd.get_dummies(market_df, columns=categorical_cols, drop_first=True)
                    feature_cols = [col for col in market_df_encoded.columns if col.startswith(('Price', 'CP') + tuple(categorical_cols))]
                    if not feature_cols:
                        print(f"Warning: No valid features after encoding for {market}")
                        self.coef_dfs[market] = pd.DataFrame(columns=['Coefficient', 'P-value'])
                        continue
                    X = market_df_encoded[feature_cols].astype(float)
                    y = market_df_encoded[target]
                    
                    # Fit LogisticRegression
                    model = LogisticRegression(multi_class='multinomial', C=100.0, max_iter=1000)
                    model.fit(X, y)
                    coef_df = pd.DataFrame({
                        'Coefficient': model.coef_[0],
                        'P-value': np.nan
                    }, index=feature_cols)
                    self.models[market] = model
                    self.coef_dfs[market] = coef_df
                    print(f"LogisticRegression model fitted for {market}")
                except Exception as e:
                    print(f"LogisticRegression failed for {market}: {e}")
                    self.coef_dfs[market] = pd.DataFrame(columns=['Coefficient', 'P-value'])
        
        return self
    
    def get_attribute_utilities(self, market):
        coef_df = self.coef_dfs.get(market, pd.DataFrame(columns=['Coefficient', 'P-value']))
        return coef_df['Coefficient'] if 'Coefficient' in coef_df.columns else pd.Series(dtype=float)
    
    def get_feature_importance(self, market):
        utilities = self.get_attribute_utilities(market)
        if utilities.empty:
            return pd.Series(dtype=float)
        importance = utilities.abs() / utilities.abs().sum() if utilities.abs().sum() > 0 else utilities
        return importance
    
    def add_model_summaries_to_workbook(self, workbook):
        for market in self.markets:
            sheet = workbook.create_sheet(title=f"{market}_summary")
            coef_df = self.coef_dfs.get(market, pd.DataFrame(columns=['Coefficient', 'P-value']))
            for r, (index, row) in enumerate(coef_df.iterrows(), start=1):
                sheet[f'A{r}'] = index
                sheet[f'B{r}'] = row.get('Coefficient', float('nan'))
                sheet[f'C{r}'] = row.get('P-value', float('nan'))
if __name__=="__main__":
    # Load data
    df = pd.read_csv("combined_choice_data_v2.csv")
    model = DiscreteChoiceModel(markets=['Maharashtra'])
    model.fit(df)
    print(model.get_attribute_utilities('Maharashtra'))
    workbook = openpyxl.Workbook()
    model.add_model_summaries_to_workbook(workbook)
    workbook.save('cargill_findings_v4.xlsx')