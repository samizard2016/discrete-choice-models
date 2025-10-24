import pandas as pd
import numpy as np
import random
from itertools import combinations
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def generate_choice_sets(n_profiles, set_size, n_sets):
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
        self.choice_data = None
        self.price_mean = None
        self.price_std = None

    def validate_data(self):
        """Check for data anomalies, choice set correlations, and respondent consistency."""
        choice_data = self.choice_data.copy()
        choice_data['Price'] = (choice_data['Price'] + self.price_mean / 1000) * 1000  # Uncenter
        price_choice = choice_data.groupby('chosen')['Price'].mean()
        print("Mean Price by Chosen Status:\n", price_choice)
        if price_choice[1] > price_choice[0]:
            print("Warning: Chosen profiles have higher mean price, possible data issue.")

        # Analyze choice sets for price-feature correlations
        choice_sets = self.choices.copy()
        correlations_modbus = []
        for _, row in choice_sets.iterrows():
            try:
                profiles = [int(v) for v in row['profiles_presented'][1:-1].split(',')]
            except:
                continue
            if not all(p in self.profiles.index for p in profiles):
                continue
            set_data = self.profiles.loc[profiles].copy()
            set_data['Price'] = set_data['Price'] / 1000
            set_encoded = pd.get_dummies(set_data, columns=['SP', 'AF'], drop_first=True, dtype=float)
            rename_dict = {
                'SP_High performance Ics=Icu=Icw 66kA for 1sec': 'Size_Perf_High',
                'AF_Higher Electrical life from 6,000 to 7,500 operations without maintenance': 'Adv_Feat_ElecLife',
                'AF_Visible Health indication (Breaker Status, trip cause Indication - OL,SC,GF)': 'Adv_Feat_Health',
                'AF_Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)': 'Adv_Feat_ModbusBasic',
                'AF_Current Measurement and time stamped fault records on mobile app': 'Adv_Feat_Current',
                'AF_Scalable connectivity at breaker level- Modbus Ethernet': 'Adv_Feat_ModbusEth',
                'AF_Operator Safety - Arc Flash reduction during maintenance': 'Adv_Feat_Safety',
                'AF_Terminal Temperature threshold monitoring': 'Adv_Feat_TempMon'
            }
            set_encoded.rename(columns=rename_dict, inplace=True)
            if 'Adv_Feat_ModbusBasic' in set_encoded.columns:
                corr_modbus = set_encoded[['Price', 'Adv_Feat_ModbusBasic']].corr().iloc[0, 1]
                if not np.isnan(corr_modbus):
                    correlations_modbus.append(corr_modbus)
        if correlations_modbus:
            mean_corr_modbus = np.mean(correlations_modbus)
            print(f"Mean Price-Adv_Feat_ModbusBasic correlation in choice sets: {mean_corr_modbus:.4f}")
            if np.abs(mean_corr_modbus) > 0.5:
                print("Warning: High Price-Adv_Feat_ModbusBasic correlation in choice sets.")

        # Check respondent consistency and choice set counts
        respondent_choices = choice_data.groupby('respondent_id')['chosen'].sum()
        print("Respondent choice counts (should be number of choice sets per respondent):\n", respondent_choices.value_counts())
        if (respondent_choices == 0).any():
            print("Warning: Some respondents never chose a profile (possible inattention).")
        choice_set_counts = choice_data.groupby('respondent_id')['choice_set'].nunique()
        print("Choice sets per respondent:\n", choice_set_counts.value_counts())
        if choice_set_counts.max() < 16:
            print(f"Warning: Maximum choice sets per respondent ({choice_set_counts.max()}) is below expected (16).")

        return price_choice

    def prepare_data(self):
        profiles = self.profiles.copy()
        profiles['Price'] = pd.to_numeric(profiles['Price'], errors='coerce')
        profiles['Price'] = profiles['Price'] / 1000  # Scale to thousands
        self.price_mean = profiles['Price'].mean() * 1000
        self.price_std = profiles['Price'].std() * 1000
        
        profiles['AF'] = profiles['AF'].replace({
            'Current Measuremnt and time stamped fault records on mobile app':
            'Current Measurement and time stamped fault records on mobile app',
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, contorl, terminal temp alarm)':
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)'
        })
        
        print("Unique SP values:", profiles['SP'].unique())
        print("Unique AF values:", sorted(profiles['AF'].unique()))
        print("Dropped AF level (first alphabetically):", sorted(profiles['AF'].unique())[0])
        
        # Exclude profiles with Adv_Feat_TempMon
        profiles = profiles[profiles['AF'] != 'Terminal Temperature threshold monitoring']
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=float
        )
        
        print("profiles_encoded columns before renaming:", profiles_encoded.columns.tolist())
        
        rename_dict = {
            'SP_High performance Ics=Icu=Icw 66kA for 1sec': 'Size_Perf_High',
            'AF_Higher Electrical life from 6,000 to 7,500 operations without maintenance': 'Adv_Feat_ElecLife',
            'AF_Visible Health indication (Breaker Status, trip cause Indication - OL,SC,GF)': 'Adv_Feat_Health',
            'AF_Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)': 'Adv_Feat_ModbusBasic',
            'AF_Current Measurement and time stamped fault records on mobile app': 'Adv_Feat_Current',
            'AF_Scalable connectivity at breaker level- Modbus Ethernet': 'Adv_Feat_ModbusEth',
            'AF_Operator Safety - Arc Flash reduction during maintenance': 'Adv_Feat_Safety'
        }
        profiles_encoded.rename(columns=rename_dict, inplace=True)
        
        print("profiles_encoded columns after renaming:", profiles_encoded.columns.tolist())
        print("profiles_encoded shape:", profiles_encoded.shape)
        
        # Filter choices to exclude invalid profiles
        valid_profiles = profiles_encoded.index.tolist()
        choices_filtered = self.choices.copy()
        initial_rows = len(choices_filtered)
        invalid_chosen = choices_filtered[~choices_filtered['chosen_profile'].isin(valid_profiles)]
        print(f"Dropping {len(invalid_chosen)} choices with invalid chosen_profile: {invalid_chosen['chosen_profile'].unique()}")
        choices_filtered = choices_filtered[choices_filtered['chosen_profile'].isin(valid_profiles)]
        
        def filter_profiles_presented(profiles_str, valid_profiles):
            try:
                profiles = [int(v) for v in profiles_str[1:-1].split(',')]
                valid = [p for p in profiles if p in valid_profiles]
                return str(valid) if len(valid) >= 2 else None
            except:
                return None
        
        choices_filtered['profiles_presented'] = choices_filtered['profiles_presented'].apply(
            lambda x: filter_profiles_presented(x, valid_profiles)
        )
        invalid_sets = choices_filtered[choices_filtered['profiles_presented'].isnull()]
        print(f"Dropping {len(invalid_sets)} choice sets with fewer than 2 valid profiles")
        choices_filtered = choices_filtered[choices_filtered['profiles_presented'].notnull()]
        final_rows = len(choices_filtered)
        print(f"Choices rows: {initial_rows} initial, {final_rows} after filtering ({initial_rows - final_rows} dropped)")
        
        if not all(choices_filtered['respondent_id'].isin(self.groups['respondent_id'])):
            missing_ids = choices_filtered[~choices_filtered['respondent_id'].isin(self.groups['respondent_id'])]['respondent_id'].unique()
            raise ValueError(f"respondent_id values in choices not found in groups: {missing_ids}")
        
        print("validation done. Attempting to create choice set data ...")
        
        choice_sets = []
        for choice_set in choices_filtered['choice_set'].unique():
            respondents = choices_filtered[choices_filtered['choice_set'] == choice_set]
            for _, respondent in respondents.iterrows():
                profiles_presented = respondent['profiles_presented']
                try:
                    profiles_presented = [int(v) for v in profiles_presented[1:-1].split(',')]
                except Exception as e:
                    print(f"Skipping malformed profiles_presented for respondent {respondent['respondent_id']}: {str(e)}")
                    continue
                if not isinstance(profiles_presented, list):
                    print(f"Skipping invalid profiles_presented for respondent {respondent['respondent_id']}: not a list")
                    continue
                for idx in profiles_presented:
                    if idx not in profiles_encoded.index:
                        print(f"Skipping invalid profile {idx} for respondent {respondent['respondent_id']}")
                        continue
                    profile = profiles_encoded.loc[idx]
                    group_row = self.groups[self.groups['respondent_id'] == respondent['respondent_id']]
                    if group_row.empty:
                        raise ValueError(f"No group found for respondent_id {respondent['respondent_id']}")
                    group = group_row['group'].iloc[0]
                    group = 'Panel builder' if group == 'Panel builder' else 'Others'
                    row = {
                        **{k: v for k, v in profile.items()},
                        'respondent_id': int(respondent['respondent_id']),
                        'choice_set': int(choice_set),
                        'group': str(group),
                        'chosen': int(1 if idx == respondent['chosen_profile'] else 0)
                    }
                    choice_sets.append(row)
        
        self.choice_data = pd.DataFrame(choice_sets)
        
        for col in self.choice_data.columns:
            if self.choice_data[col].dtype == bool:
                self.choice_data[col] = self.choice_data[col].astype(float)
        
        unnamed_cols = [col for col in self.choice_data.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"Dropping Unnamed columns from choice_data: {unnamed_cols}")
            self.choice_data = self.choice_data.drop(columns=unnamed_cols)
        
        # Center Price
        self.choice_data['Price'] = -(self.choice_data['Price'] - self.choice_data['Price'].mean())  # Negative price
        print("Price range (centered, thousands):\n", self.choice_data['Price'].describe())
        
        print("Group value counts:\n", self.choice_data['group'].value_counts())
        print("Sample of choice_data:\n", self.choice_data[['respondent_id', 'choice_set', 'group', 'chosen']].head())
        print("Final choice_data columns:", self.choice_data.columns.tolist())
        print("Final choice_data dtypes:\n", self.choice_data.dtypes)
        
        numeric_cols = [col for col in self.choice_data.columns if col not in ['respondent_id', 'choice_set', 'group', 'chosen']]
        if self.choice_data[numeric_cols].isna().any().any():
            raise ValueError("Missing values in numeric columns")
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.choice_data[col]):
                raise ValueError(f"Non-numeric column {col}: {self.choice_data[col].dtype}")
        
        self.validate_data()
        return self.choice_data

    def fit_model(self, group=None):
        X_cols = [
            'Price', 'Size_Perf_High', 'Adv_Feat_ElecLife', 'Adv_Feat_Current',
            'Adv_Feat_Safety', 'Adv_Feat_Health'
        ]
        
        if not all(col in self.choice_data.columns for col in X_cols):
            missing_cols = [col for col in X_cols if col not in self.choice_data.columns]
            raise ValueError(f"Missing columns in choice_data: {missing_cols}")
        
        data = self.choice_data if group is None else self.choice_data[self.choice_data['group'] == group]
        X = data[X_cols]
        y = data['chosen']
        
        print(f"Fitting model for {'all groups' if group is None else group}")
        print("X_cols:", X_cols)
        print("X shape:", X.shape)
        print("X dtypes:\n", X.dtypes)
        print("Correlation matrix:\n", X.corr())
        
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X_cols
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print("VIF:\n", vif_data)
        
        non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        if non_numeric_cols:
            raise ValueError(f"Non-numeric columns in X: {non_numeric_cols}")
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Non-numeric y: {y.dtype}")
        
        X = sm.add_constant(X, has_constant='add')
        try:
            self.model = sm.Logit(y, X).fit(disp=0, maxiter=2000, method='bfgs')
            if not self.model.mle_retvals['converged']:
                print("bfgs failed to converge, trying newton...")
                self.model = sm.Logit(y, X).fit(disp=0, maxiter=2000, method='newton')
        except Exception as e:
            raise RuntimeError(f"Logit model fitting failed: {str(e)}")
        
        print("Model Converged:", self.model.mle_retvals['converged'])
        print("Model Summary:\n", self.model.summary())
        
        self.utilities = self.model.params
        return self.utilities

    def calculate_feature_importance(self):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        X_cols = [col for col in self.utilities.index if col != 'const']
        X = self.choice_data[X_cols]
        
        coef = self.model.params[X_cols].values
        intercept = self.model.params['const']
        explainer = shap.LinearExplainer((coef, intercept), shap.maskers.Independent(X))
        shap_values = explainer.shap_values(X)
        
        importance = np.abs(shap_values).mean(axis=0)
        total = importance.sum()
        self.feature_importance = {col: imp / total for col, imp in zip(X_cols, importance)}
        
        shap.summary_plot(shap_values, X, feature_names=X_cols, plot_type="bar", show=False)
        plt.savefig('shap_importance.png')
        plt.close()
        
        return self.feature_importance

    def plot_utilities(self, custom_labels=None):
        if self.utilities is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        default_labels = {
            'Price': 'Price (thousands, centered)',
            'Size_Perf_High': 'High Performance Size (66kA)',
            'Adv_Feat_ElecLife': 'Higher Electrical Life',
            'Adv_Feat_Current': 'Current Measurement & Fault Records',
            'Adv_Feat_Safety': 'Operator Safety',
            'Adv_Feat_Health': 'Visible Health Indication'
        }
        
        label_map = custom_labels if custom_labels is not None else default_labels
        
        utilities = self.utilities.drop('const', errors='ignore').copy()
        utilities.index = [label_map.get(idx, idx) for idx in utilities.index]
        
        plt.figure(figsize=(10, 6))
        utilities.plot(kind='bar')
        plt.title('Feature Utilities')
        plt.ylabel('Utility')
        plt.xlabel('Features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('utilities.png')
        plt.close()

    def plot_feature_importance(self):
        if self.feature_importance is None:
            raise ValueError("Feature importance is not calculated yet. Call calculate_feature_importance() first.")
        
        importance_df = pd.DataFrame(list(self.feature_importance.items()), columns=['Feature', 'Importance'])
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=importance_df.sort_values('Importance', ascending=False))
        plt.title('Feature Importance (SHAP)')
        plt.savefig('feature_importance_dcm.png')
        plt.close()

    def plot_price_elasticity(self):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        X_cols = [col for col in self.utilities.index if col != 'const']
        if not all(col in self.choice_data.columns for col in X_cols):
            missing_cols = [col for col in X_cols if col not in self.choice_data.columns]
            raise ValueError(f"Missing columns in choice_data: {missing_cols}")
        
        X = self.choice_data[X_cols]
        X = sm.add_constant(X, has_constant='add')
        V = self.model.predict(X, which="linear")
        
        temp_data = self.choice_data.copy()
        temp_data['V'] = V
        temp_data['exp_V'] = np.exp(V)
        temp_data['sum_exp_V'] = temp_data.groupby(['respondent_id', 'choice_set'])['exp_V'].transform('sum')
        temp_data['P_i'] = temp_data['exp_V'] / temp_data['sum_exp_V']
        temp_data['E_ii'] = (1 - temp_data['P_i']) * self.utilities['Price'] * temp_data['Price']
        temp_data['original_price'] = (temp_data['Price'] + self.price_mean / 1000) * 1000
        
        mean_elasticity = temp_data.groupby('original_price')['E_ii'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='original_price', y='E_ii', data=mean_elasticity, marker='o')
        plt.title('Average Price Elasticity vs. Price')
        plt.xlabel('Price ($)')
        plt.ylabel('Elasticity')
        plt.savefig('price_elasticity.png')
        plt.close()

    def evaluate_price_scenario_lp(self, price_scenarios, profile_labels=None, plot=True):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        profiles = self.profiles.copy()
        profiles['Price'] = pd.to_numeric(profiles['Price'], errors='coerce')
        profiles = profiles[profiles['AF'] != 'Terminal Temperature threshold monitoring']
        
        profiles['AF'] = profiles['AF'].replace({
            'Current Measuremnt and time stamped fault records on mobile app': 'Current Measurement and time stamped fault records on mobile app',
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, contorl, terminal temp alarm)': 'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)'
        })
        
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=float
        )
        
        rename_dict = {
            'SP_High performance Ics=Icu=Icw 66kA for 1sec': 'Size_Perf_High',
            'AF_Higher Electrical life from 6,000 to 7,500 operations without maintenance': 'Adv_Feat_ElecLife',
            'AF_Visible Health indication (Breaker Status, trip cause Indication - OL,SC,GF)': 'Adv_Feat_Health',
            'AF_Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)': 'Adv_Feat_ModbusBasic',
            'AF_Current Measurement and time stamped fault records on mobile app': 'Adv_Feat_Current',
            'AF_Scalable connectivity at breaker level- Modbus Ethernet': 'Adv_Feat_ModbusEth',
            'AF_Operator Safety - Arc Flash reduction during maintenance': 'Adv_Feat_Safety'
        }
        profiles_encoded.rename(columns=rename_dict, inplace=True)
        
        X_cols = [col for col in self.utilities.index if col != 'const']
        for col in X_cols:
            if col not in profiles_encoded.columns:
                profiles_encoded[col] = 0.0
        
        profiles_encoded['Price'] = -profiles['Price'].values / 1000  # Negative price
        
        if isinstance(price_scenarios, dict):
            scenarios_df = pd.DataFrame(price_scenarios, dtype=float)
        else:
            scenarios_df = price_scenarios.astype(float)
        
        shares = pd.DataFrame(index=profiles_encoded.index, columns=scenarios_df.columns)
        
        for scenario in scenarios_df.columns:
            temp_profiles = profiles_encoded.copy()
            for profile_idx in scenarios_df.index:
                if profile_idx in temp_profiles.index:
                    temp_profiles.loc[profile_idx, 'Price'] = -scenarios_df.loc[profile_idx, scenario] / 1000
            X = temp_profiles[X_cols]
            X = sm.add_constant(X, has_constant='add')
            V = X @ self.utilities
            exp_V = np.exp(V)
            sum_exp_V = exp_V.sum()
            probabilities = exp_V / sum_exp_V
            shares[scenario] = probabilities
        
        if profile_labels is None:
            profile_labels = [f'Profile {i}' for i in profiles_encoded.index]
        elif isinstance(profile_labels, dict):
            profile_labels = [profile_labels.get(i, f'Profile {i}') for i in profiles_encoded.index]
        elif isinstance(profile_labels, list):
            if len(profile_labels) != len(profiles_encoded.index):
                raise ValueError("profile_labels list length must match number of profiles")
        
        if plot:
            plot_data = shares.reset_index().melt(id_vars='index', var_name='Scenario', value_name='Share')
            plot_data['Profile'] = plot_data['index'].map({i: label for i, label in enumerate(profile_labels)})
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=plot_data, x='Profile', y='Share', hue='Scenario', style='Scenario', markers=True, dashes=False)
            plt.title('Profile Shares Across Price Scenarios')
            plt.xlabel('Profile')
            plt.ylabel('Choice Probability (Share, %)')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('profile_shares_line.png')
            plt.close()
        
        return shares * 100

if __name__ == "__main__":
    profiles = pd.read_excel('profiles.xlsx', index_col=0)
    choices = pd.read_excel("CBC_Data_Final_09Jun25.xlsx")
    groups = pd.read_excel("A2_9Jun25.xlsx")
    
    choices['respondent_id'] = pd.to_numeric(choices['respondent_id'], errors='coerce')
    choices['choice_set'] = pd.to_numeric(choices['choice_set'], errors='coerce')
    choices['chosen_profile'] = pd.to_numeric(choices['chosen_profile'], errors='coerce')
    
    for df in [profiles, choices, groups]:
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"Dropping Unnamed columns from {df}: {unnamed_cols}")
            df.drop(columns=unnamed_cols, inplace=True)
    
    print("Profiles dtypes:\n", profiles.dtypes)
    print("Choices dtypes:\n", choices.dtypes)
    print("Groups dtypes:\n", groups.dtypes)
    
    try:
        analyzer = DiscreteChoiceAnalyzer(profiles, choices, groups)
        choice_data = analyzer.prepare_data()
        
        # Fit overall model
        utilities = analyzer.fit_model()
        importance = analyzer.calculate_feature_importance()
        print("Utilities:\n", utilities)
        print("Feature Importance (SHAP):\n", importance)
        analyzer.plot_utilities()
        analyzer.plot_feature_importance()
        analyzer.plot_price_elasticity()
        
        price_scenarios = {
            'Baseline': profiles['Price'].to_dict(),
            '20% Increase': (profiles['Price'] * 1.2).to_dict(),
            '20% Decrease': (profiles['Price'] * 0.8).to_dict(),
            'Custom': {i: profiles['Price'].mean() * 1.1 for i in profiles.index}
        }
        
        shares = analyzer.evaluate_price_scenario_lp(price_scenarios, plot=True)
        print("Profile Shares (%):\n", shares.round(2))
        
        # Fit group-specific models
        print("\nFitting group-specific models...")
        for group in choice_data['group'].unique():
            print(f"\nFitting model for {group}...")
            utilities = analyzer.fit_model(group=group)
            print(f"Utilities for {group}:\n", utilities)
            shares = analyzer.evaluate_price_scenario_lp(price_scenarios, plot=True)
            print(f"Profile Shares for {group} (%):\n", shares.round(2))
        
        print("Done")
    except Exception as e:
        print(f"Error occurred: {str(e)}")