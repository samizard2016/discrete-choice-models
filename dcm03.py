import pandas as pd
import numpy as np
import random
from itertools import combinations
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def generate_choice_sets(n_profiles, set_size, n_sets):
    """
    Generate choice sets with specified size and number, ensuring balance.
    """
    profile_indices = list(range(n_profiles))
    all_combinations = list(combinations(profile_indices, set_size))
    
    selected_sets = []
    profile_counts = {i: 0 for i in profile_indices}
    
    random.seed(42)
    while len(selected_sets) < n_sets:
        candidate = random.choice(all_combinations)
        temp_counts = profile_counts.copy()
        for p in candidate:
            temp_counts[p] += 1
        if max(temp_counts.values()) - min(temp_counts.values()) <= 2:
            selected_sets.append(list(candidate))
            for p in candidate:
                profile_counts[p] += 1
            all_combinations.remove(candidate)
    
    choice_sets = pd.DataFrame({
        'choice_set': range(n_sets),
        'profiles_presented': selected_sets
    })
    
    return choice_sets, profile_counts

class DiscreteChoiceAnalyzer:
    def __init__(self, profiles, choices, groups):
        self.profiles = profiles
        self.choices = choices
        self.groups = groups
        self.model = None
        self.utilities = None
        self.feature_importance = None
        self.choice_data = choices
    def prepare_data(self):
        profiles = self.profiles.copy()
        profiles['Price'] = profiles['Price'] / 1000
    
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=int
        )
        
        # Rename columns
        rename_dict = {
            'Size_Performance_High performance Ics=Icu=Icw 66kA for 1sec': 'Size_Perf_High',
            'Higher Electrical life from 6,000 to 7,500 operations without maintenance': 'Extended Electrical Life',
            'Visible Health indication (Breaker Status, trip cause Indication - OL,SC,GF)': 'Breaker Health Indicator',
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, contorl, terminal temp alarm)': 'Basic Modbus Connectivity',
            'Current Measuremnt and time stamped fault records on mobile app': 'Mobile Fault Recording',
            'Access trip unit data during tripping events without supply': 'Trip Data Access',
            'Scalable connectivity at breaker level- Modbus Ethernet': 'Modbus Ethernet Connectivity',
            'Operator Safety - Arc Flash reduction during maintenance': 'Arc Flash Safety',
            'Terminal Temperature threshold monitoring': 'Terminal Temp Monitoring'
            }
        profiles_encoded.rename(columns=rename_dict, inplace=True)
        
        # print("profiles_encoded columns:", profiles_encoded.columns.tolist())
        # print("profiles_encoded shape:", profiles_encoded.shape)     
        
        # Validate choices
        if not all(self.choices['chosen_profile'].isin(profiles_encoded.index)):
            raise ValueError("Invalid chosen_profile values")
        
        # Merge
        choice_data = self.choice_data.merge(profiles_encoded, left_on='chosen_profile', right_index=True, validate="many_to_one")
        choice_data = choice_data.merge(self.groups, on='respondent_id', validate="many_to_one")
        
        # Debug: Print choice_data columns after merge
        # print("choice_data columns after merge:", choice_data.columns.tolist())
        # Identify columns with 'Unnamed' in their names
        unnamed_cols = [col for col in choice_data.columns if 'Unnamed' in col]
        # Drop those columns
        choice_data = choice_data.drop(columns=unnamed_cols)
        
        # # Create choice set data
        # choice_sets = []
        # for choice_set in choice_data['choice_set'].unique():
        #     respondents = choice_data[choice_data['choice_set'] == choice_set]
        #     for _, respondent in respondents.iterrows():
        #         profiles_presented = respondent['profiles_presented']
        #         try:
        #             profiles_presented = [int(v) for v in profiles_presented[1:-1].split(',')]
        #         except Exception as e:
        #             raise ValueError(f"Failed to parse profiles_presented: {str(e)}")
        #         if not isinstance(profiles_presented, list):
        #             raise ValueError(f"profiles_presented must be a list, got {type(profiles_presented)}")
        #         for idx in profiles_presented:
        #             profile = profiles_encoded.loc[idx]
        #             row = {
        #                 **{k: v for k, v in profile.items()},
        #                 'respondent_id': int(respondent['respondent_id']),
        #                 'choice_set': int(choice_set),
        #                 'group': str(respondent['group']),
        #                 'chosen': 1 if idx == respondent['chosen_profile'] else 0
        #             }
        #             choice_sets.append(row)
        
        # self.choice_data = pd.DataFrame(choice_sets)
        
        # Convert any bool columns to int
        # for col in self.choice_data.columns:
        #     if self.choice_data[col].dtype == bool:
        #         self.choice_data[col] = self.choice_data[col].astype(int)
        
        # Debug: Print final choice_data columns and shape
        # print("Final choice_data columns:", choice_data.columns.tolist())
        # print("Final choice_data shape:", choice_data.shape)

        
        # Verify numeric columns
        numeric_cols = [col for col in choice_data.columns if col not in ['respondent_id', 'choice_set', 'group', 'chosen']]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(choice_data[col]):
                raise ValueError(f"Non-numeric column {col}: {choice_data[col].dtype}")
        self.choice_data = choice_data        
        return choice_data

    # def xprepare_data(self):
        """
        Prepare data for conditional logit model, ensuring numeric output.
        """
        # self.profiles.to_excel("dcm02_profiles.xlsx", index=True)
        
        # Scale Price and encode dummies as int
        profiles = self.profiles.copy()
        profiles['Price'] = profiles['Price'] / 1000
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['Size_Performance', 'Advanced_Feature'],
            drop_first=True,
            dtype=int  # Ensure dummy variables are int64, not bool
        )
        
        # Validate choices
        if not all(self.choices['chosen_profile'].isin(profiles_encoded.index)):
            raise ValueError("Invalid chosen_profile values")
        
        # Merge
        choice_data = self.choices.merge(profiles_encoded, left_on='chosen_profile', right_index=True, validate="many_to_one")
        choice_data = choice_data.merge(self.groups, on='respondent_id', validate="many_to_one")
        
        # Create choice set data
        choice_sets = []
        for choice_set in choice_data['choice_set'].unique():
            respondents = choice_data[choice_data['choice_set'] == choice_set]
            for _, respondent in respondents.iterrows():
                profiles_presented = respondent['profiles_presented']
                profiles_presented = [int(v) for v in profiles_presented[1:-1].split(',')]
                if not isinstance(profiles_presented, list):
                    raise ValueError(f"profiles_presented must be a list, got {type(profiles_presented)}")
                for idx in profiles_presented:
                    profile = profiles_encoded.loc[idx]
                    row = {
                        **{k: v for k, v in profile.items()},
                        'respondent_id': int(respondent['respondent_id']),
                        'choice_set': int(choice_set),
                        'group': str(respondent['group']),
                        'chosen': 1 if idx == respondent['chosen_profile'] else 0
                    }
                    choice_sets.append(row)
        
        self.choice_data = pd.DataFrame(choice_sets)
        
        # Convert any bool columns to int
        for col in self.choice_data.columns:
            if self.choice_data[col].dtype == bool:
                self.choice_data[col] = self.choice_data[col].astype(int)
        
        # Verify numeric columns
        numeric_cols = [col for col in self.choice_data.columns if col not in ['Unnamed: 0','respondent_id', 'choice_set', 'group', 'chosen']]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.choice_data[col]):
                raise ValueError(f"Non-numeric column {col}: {self.choice_data[col].dtype}")
        
        return self.choice_data

    def fit_model(self):
        """
        Fit conditional logit model using statsmodels.
        """
        X_cols = [col for col in self.choice_data.columns if col not in ['respondent_id', 'choice_set', 'chosen', 'group']]
        # # Explicitly define predictor columns based on provided output
        # X_cols = [
        #     'Price',
        #     'Size_Performance_High performance Ics=Icu=Icw 66kA for 1sec',
        #     'Advanced_Feature_Current Measurement and time stamped fault records on mobile app',
        #     'Advanced_Feature_Higher Electrical life from 6,000 to 7,500 operations without maintenance',
        #     'Advanced_Feature_Operator Safety - Arc Flash reduction during maintenance',
        #     'Advanced_Feature_Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)',
        #     'Advanced_Feature_Scalable connectivity at breaker level- Modbus Ethernet',
        #     'Advanced_Feature_Terminal Temperature threshold monitoring',
        #     'Advanced_Feature_Visible Health indication (Breaker Status, trip cause Indication - OL,SC,GF)'
        # ]
        if not X_cols:
            raise ValueError("No predictor columns found")        
        X = self.choice_data[X_cols]
        y = self.choice_data['chosen']
        
        # Debug: Print columns and shape
        print("X_cols:", X_cols)
        print("X shape before constant:", X.shape)
        print("X dtypes:\n", X.dtypes)
        
        # Ensure X and y are numeric
        if not all(X.dtypes.apply(pd.api.types.is_numeric_dtype)):
            raise ValueError(f"Non-numeric columns in X: {X.dtypes}")
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Non-numeric y: {y.dtype}")
        
        X = sm.add_constant(X, has_constant='add')
        print("X shape after constant:", X.shape)
        
        try:
            self.model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        except Exception as e:
            raise RuntimeError(f"Logit model fitting failed: {str(e)}")
        self.utilities = self.model.params
        return self.utilities

    def calculate_feature_importance(self):
        """
        Calculate feature importance.
        """
        size_utils = self.utilities.filter(like='SP_').abs()
        adv_feature_utils = self.utilities.filter(like='AF_').abs()
        price_util = abs(self.utilities['Price']) * (self.profiles['Price'].max() - self.profiles['Price'].min())
        
        importance = {
            'Size_Performance': size_utils.max() - size_utils.min() if not size_utils.empty else 0,
            'Advanced_Feature': adv_feature_utils.max() - adv_feature_utils.min() if not adv_feature_utils.empty else 0,
            'Price': price_util
        }
        
        total = sum(importance.values())
        if total == 0:
            raise ValueError("Total importance is zero")
        self.feature_importance = {k: v / total for k, v in importance.items()}
        return self.feature_importance

    def plot_utilities(self):
        """Plot the utilities of the features."""
        if self.utilities is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        utilities = self.utilities.drop('const', errors='ignore')  # Exclude constant if present
        plt.figure(figsize=(10, 6))
        utilities.plot(kind='bar')
        plt.title('Feature Utilities')
        plt.ylabel('Utility')
        plt.xlabel('Features')
        plt.savefig('utilities.png')
        plt.close()

    def plot_feature_importance(self):
        """Plot the feature importance."""
        if self.feature_importance is None:
            raise ValueError("Feature importance is not calculated yet. Call calculate_feature_importance() first.")
        
        importance_df = pd.DataFrame(list(self.feature_importance.items()), columns=['Feature', 'Importance'])
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=importance_df.sort_values('Importance', ascending=False))
        plt.title('Feature Importance')
        plt.savefig('feature_importance.png')
        plt.close()

    def plot_price_elasticity(self):
        """Plot price elasticity based on the fitted model."""
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        X_cols = [col for col in self.choice_data.columns if col not in ['respondent_id', 'choice_set', 'chosen', 'group']]
        X = self.choice_data[X_cols]
        X = sm.add_constant(X, has_constant='add')
        V = self.model.predict(X, linear=True)
        
        temp_data = self.choice_data.copy()
        temp_data['V'] = V
        temp_data['exp_V'] = np.exp(V)
        temp_data['sum_exp_V'] = temp_data.groupby(['respondent_id', 'choice_set'])['exp_V'].transform('sum')
        temp_data['P_i'] = temp_data['exp_V'] / temp_data['sum_exp_V']
        temp_data['E_ii'] = (1 - temp_data['P_i']) * self.utilities['Price'] * temp_data['Price']
        temp_data['original_price'] = temp_data['Price'] * 1000
        
        # Compute mean elasticity per unique price
        mean_elasticity = temp_data.groupby('original_price')['E_ii'].mean().reset_index()
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='original_price', y='E_ii', data=mean_elasticity, marker='o')
        plt.title('Average Price Elasticity vs. Price')
        plt.xlabel('Price')
        plt.ylabel('Elasticity')
        plt.savefig('price_elasticity.png')
        plt.close()

if __name__ == "__main__":  
    # Load data
    profiles = pd.read_excel('profiles.xlsx')
    choices = pd.read_excel("CBC_Data_Final_06Jun25.xlsx")
    groups = pd.read_excel("groups_updated.xlsx")
   

    # Run analysis
    try:
        analyzer = DiscreteChoiceAnalyzer(profiles, choices, groups)
        choice_data = analyzer.prepare_data()
        
        # Debug
        print("choice_data columns:", choice_data.columns)
        print("choice_data dtypes:\n", choice_data.dtypes)
      
        
        utilities = analyzer.fit_model()
        importance = analyzer.calculate_feature_importance()

        print("Utilities:\n", utilities)
        print("Feature Importance:\n", importance)

        # Plotting
        analyzer.plot_utilities()
        analyzer.plot_feature_importance()
        analyzer.plot_price_elasticity()
        print("Done")

        # Save data
        # choice_data.to_excel("choice_data_updated.xlsx")
        # profiles.to_excel("profiles_updated.xlsx", index=True)
        # choices.to_excel("choices_updated.xlsx")
        # groups.to_excel("groups_updated.xlsx")
    except Exception as e:
        print(f"Error occurred: {str(e)}")