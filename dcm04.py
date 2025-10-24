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
        self.choice_data = None

    def prepare_data(self):
        """
        Prepare data for conditional logit model, ensuring numeric output.
        """
        profiles = self.profiles.copy()
        profiles['Price'] = pd.to_numeric(profiles['Price'], errors='coerce') / 1000
        
        # Fix typos in AF column
        profiles['AF'] = profiles['AF'].replace({
            'Current Measuremnt and time stamped fault records on mobile app':
            'Current Measurement and time stamped fault records on mobile app',
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, contorl, terminal temp alarm)':
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)'
        })
        
        print("Unique SP values:", profiles['SP'].unique())
        print("Unique AF values:", sorted(profiles['AF'].unique()))  # Sorted for clarity
        print("Dropped AF level (first alphabetically):", sorted(profiles['AF'].unique())[0])
        
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=int
        )
        
        print("profiles_encoded columns before renaming:", profiles_encoded.columns.tolist())
        
        # Rename dummy columns
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
        profiles_encoded.rename(columns=rename_dict, inplace=True)
        
        print("profiles_encoded columns after renaming:", profiles_encoded.columns.tolist())
        print("profiles_encoded shape:", profiles_encoded.shape)
        
        # Validate choices
        if not all(self.choices['chosen_profile'].isin(profiles_encoded.index)):
            raise ValueError("Invalid chosen_profile values in choices")
        print(f"choice data columns: {self.choices.columns}")
        print(f"groups columns: {self.groups.columns}")
        
        # Validate groups merge
        if not all(self.choices['respondent_id'].isin(self.groups['respondent_id'])):
            missing_ids = self.choices[~self.choices['respondent_id'].isin(self.groups['respondent_id'])]['respondent_id'].unique()
            raise ValueError(f"respondent_id values in choices not found in groups: {missing_ids}")
        
        print("validation done. Attempting to create choice set data ...")
        
        # Create choice set data
        choice_sets = []
        for choice_set in self.choices['choice_set'].unique():
            respondents = self.choices[self.choices['choice_set'] == choice_set]
            for _, respondent in respondents.iterrows():
                profiles_presented = respondent['profiles_presented']
                try:
                    profiles_presented = [int(v) for v in profiles_presented[1:-1].split(',')]
                except Exception as e:
                    raise ValueError(f"Failed to parse profiles_presented for respondent {respondent['respondent_id']}: {str(e)}")
                if not isinstance(profiles_presented, list):
                    raise ValueError(f"profiles_presented must be a list, got {type(profiles_presented)}")
                for idx in profiles_presented:
                    if idx not in profiles_encoded.index:
                        raise ValueError(f"Profile index {idx} not found in profiles_encoded")
                    profile = profiles_encoded.loc[idx]
                    # Merge group information
                    group_row = self.groups[self.groups['respondent_id'] == respondent['respondent_id']]
                    if group_row.empty:
                        raise ValueError(f"No group found for respondent_id {respondent['respondent_id']}")
                    group = group_row['group'].iloc[0]
                    row = {
                        **{k: v for k, v in profile.items()},
                        'respondent_id': int(respondent['respondent_id']),
                        'choice_set': int(choice_set),
                        'group': str(group),
                        'chosen': int(1 if idx == respondent['chosen_profile'] else 0)
                    }
                    choice_sets.append(row)
        
        self.choice_data = pd.DataFrame(choice_sets)
        
        # Convert any bool columns to int
        for col in self.choice_data.columns:
            if self.choice_data[col].dtype == bool:
                self.choice_data[col] = self.choice_data[col].astype(int)
        
        # Remove Unnamed columns
        unnamed_cols = [col for col in self.choice_data.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"Dropping Unnamed columns from choice_data: {unnamed_cols}")
            self.choice_data = self.choice_data.drop(columns=unnamed_cols)
        
        # Debug: Print group value counts and sample
        print("Group value counts:\n", self.choice_data['group'].value_counts())
        print("Sample of choice_data:\n", self.choice_data[['respondent_id', 'choice_set', 'group', 'chosen']].head())
        print("Final choice_data columns:", self.choice_data.columns.tolist())
        print("Final choice_data dtypes:\n", self.choice_data.dtypes)
        
        # Verify numeric columns
        numeric_cols = [col for col in self.choice_data.columns if col not in ['respondent_id', 'choice_set', 'group', 'chosen']]
        if self.choice_data[numeric_cols].isna().any().any():
            raise ValueError("Missing values in numeric columns")
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.choice_data[col]):
                raise ValueError(f"Non-numeric column {col}: {self.choice_data[col].dtype}")
        
        return self.choice_data

    def fit_model(self):
        """
        Fit conditional logit model using statsmodels.
        """
        X_cols = [
            'Price',
            'Size_Perf_High',
            'Adv_Feat_ElecLife',
            'Adv_Feat_Health',
            'Adv_Feat_ModbusBasic',
            'Adv_Feat_Current',
            'Adv_Feat_ModbusEth',
            'Adv_Feat_Safety',
            'Adv_Feat_TempMon'
        ]
        if not all(col in self.choice_data.columns for col in X_cols):
            missing_cols = [col for col in X_cols if col not in self.choice_data.columns]
            raise ValueError(f"Missing columns in choice_data: {missing_cols}")
        
        X = self.choice_data[X_cols]
        y = self.choice_data['chosen']
        
        print("X_cols:", X_cols)
        print("X shape:", X.shape)
        print("X dtypes:\n", X.dtypes)
        print("y dtype:", y.dtype)
        
        non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        if non_numeric_cols:
            raise ValueError(f"Non-numeric columns in X: {non_numeric_cols}")
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(f"Non-numeric y: {y.dtype}")
        
        X = sm.add_constant(X, has_constant='add')
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
        size_utils = self.utilities.filter(like='Size_Perf_').abs()
        adv_feature_utils = self.utilities.filter(like='Adv_Feat_').abs()
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
    def calculate_feature_importancex(self):
        """
        Calculate feature importance.
        """
        size_utils = self.utilities.filter(like='Size_Perf_').abs()
        adv_feature_utils = self.utilities.filter(like='Adv_Feat_').abs()
        # Scale price range to model units (thousands)
        price_range = (self.profiles['Price'].max() - self.profiles['Price'].min()) / 1000
        price_util = abs(self.utilities['Price']) * price_range
        
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

    def plot_utilities(self, custom_labels=None):
        """Plot the utilities of the features with optional custom labels.
        
        Parameters:
        - custom_labels: dict, optional
            Dictionary mapping model feature names to custom display names for the plot.
            If None, uses the renamed feature names from the model.
        """
        if self.utilities is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        # Default custom labels for more readable feature names
        default_labels = {
            'Price': 'Price (in thousands)',
            'Size_Perf_High': 'High Performance Size (66kA)',
            'Adv_Feat_ElecLife': 'Higher Electrical Life',
            'Adv_Feat_Health': 'Visible Health Indication',
            'Adv_Feat_ModbusBasic': 'Basic Modbus Connectivity',
            'Adv_Feat_Current': 'Current Measurement & Fault Records',
            'Adv_Feat_ModbusEth': 'Modbus Ethernet Connectivity',
            'Adv_Feat_Safety': 'Operator Safety (Arc Flash Reduction)',
            'Adv_Feat_TempMon': 'Terminal Temperature Monitoring'
        }
        
        # Use provided custom_labels if available, otherwise use default_labels
        label_map = custom_labels if custom_labels is not None else default_labels
        
        # Prepare utilities for plotting
        utilities = self.utilities.drop('const', errors='ignore').copy()
        
        # Rename index for plotting
        utilities.index = [label_map.get(idx, idx) for idx in utilities.index]
        
        # Create bar plot
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
        """Plot the feature importance."""
        if self.feature_importance is None:
            raise ValueError("Feature importance is not calculated yet. Call calculate_feature_importance() first.")
        
        importance_df = pd.DataFrame(list(self.feature_importance.items()), columns=['Feature', 'Importance'])
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=importance_df.sort_values('Importance', ascending=False))
        plt.title('Feature Importance')
        plt.savefig('feature_importance_dcm.png')
        plt.close()

    def plot_price_elasticity(self):
        """Plot price elasticity based on the fitted model."""
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        X_cols = [
            'Price',
            'Size_Perf_High',
            'Adv_Feat_ElecLife',
            'Adv_Feat_Health',
            'Adv_Feat_ModbusBasic',
            'Adv_Feat_Current',
            'Adv_Feat_ModbusEth',
            'Adv_Feat_Safety',
            'Adv_Feat_TempMon'
        ]
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
        temp_data['original_price'] = temp_data['Price'] * 1000
        
        mean_elasticity = temp_data.groupby('original_price')['E_ii'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='original_price', y='E_ii', data=mean_elasticity, marker='o')
        plt.title('Average Price Elasticity vs. Price')
        plt.xlabel('Price')
        plt.ylabel('Elasticity')
        plt.savefig('price_elasticity.png')
        plt.close()
    # Add this method to the DiscreteChoiceAnalyzer class
    def evaluate_price_scenario(self, price_scenarios, plot=True):
        """
        Evaluate profile shares for given price scenarios and optionally plot results.
        
        Parameters:
        - price_scenarios: dict or pd.DataFrame
            If dict: {scenario_name: {profile_idx: new_price, ...}, ...}
            If DataFrame: columns are scenario names, index is profile indices, values are prices (in thousands)
        - plot: bool, whether to generate a stacked bar chart (default: True)
        
        Returns:
        - pd.DataFrame: Profile shares for each scenario (rows: profiles, columns: scenarios)
        """
        if self.model is None or self.utilities is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        # Prepare profiles data
        profiles = self.profiles.copy()
        profiles['Price'] = pd.to_numeric(profiles['Price'], errors='coerce') / 1000
        
        # Fix typos in AF column (consistent with prepare_data)
        profiles['AF'] = profiles['AF'].replace({
            'Current Measuremnt and time stamped fault records on mobile app': 'Current Measurement and time stamped fault records on mobile app',
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, contorl, terminal temp alarm)': 'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)'
        })
        
        # Encode profiles
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=int
        )
        
        # Rename columns (consistent with prepare_data)
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
        profiles_encoded.rename(columns=rename_dict, inplace=True)
        
        # Ensure all X_cols are present
        X_cols = [
            'Price',
            'Size_Perf_High',
            'Adv_Feat_ElecLife',
            'Adv_Feat_Health',
            'Adv_Feat_ModbusBasic',
            'Adv_Feat_Current',
            'Adv_Feat_ModbusEth',
            'Adv_Feat_Safety',
            'Adv_Feat_TempMon'
        ]
        for col in X_cols:
            if col not in profiles_encoded.columns:
                profiles_encoded[col] = 0  # Add missing dummy columns as zeros
        
        # Convert price_scenarios to DataFrame if dict
        if isinstance(price_scenarios, dict):
            scenarios_df = pd.DataFrame(price_scenarios)
        else:
            scenarios_df = price_scenarios.copy()
        
        # Ensure prices are in thousands
        scenarios_df = scenarios_df / 1000
        
        # Initialize shares DataFrame
        shares = pd.DataFrame(index=profiles_encoded.index, columns=scenarios_df.columns)
        
        # Compute utilities and shares for each scenario
        for scenario in scenarios_df.columns:
            # Copy profiles_encoded
            temp_profiles = profiles_encoded.copy()
            # Update prices for the scenario
            for profile_idx in scenarios_df.index:
                if profile_idx in temp_profiles.index:
                    temp_profiles.loc[profile_idx, 'Price'] = scenarios_df.loc[profile_idx, scenario]
            
            # Compute utilities
            X = temp_profiles[X_cols]
            X = sm.add_constant(X, has_constant='add')
            V = X @ self.utilities  # Linear predictor (utility)
            
            # Compute choice probabilities (softmax)
            exp_V = np.exp(V)
            sum_exp_V = exp_V.sum()
            probabilities = exp_V / sum_exp_V
            
            # Store shares
            shares[scenario] = probabilities
        
        # Plot stacked bar chart
        if plot:
            plt.figure(figsize=(12, 6))
            shares.T.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title('Profile Shares Across Price Scenarios')
            plt.xlabel('Price Scenario')
            plt.ylabel('Choice Probability (Share)')
            plt.legend(title='Profile Index', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('profile_shares.png')
            plt.close()
        
        return shares * 100  # Return shares as percentages
    def evaluate_price_scenario_lp(self, price_scenarios, profile_labels=None, plot=True):
        """
        Evaluate profile shares for given price scenarios and optionally plot results as a line plot.
        
        Parameters:
        - price_scenarios: dict or pd.DataFrame
            If dict: {scenario_name: {profile_idx: new_price, ...}, ...}
            If DataFrame: columns are scenario names, index is profile indices, values are prices (in thousands)
        - profile_labels: dict or list, optional
            If dict: {profile_idx: label, ...}
            If list: [label for profile_idx 0, label for profile_idx 1, ...]
            If None: Use profile indices (e.g., 'Profile 0')
        - plot: bool, whether to generate a line plot (default: True)
        
        Returns:
        - pd.DataFrame: Profile shares for each scenario (rows: profiles, columns: scenarios)
        """
        if self.model is None or self.utilities is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        # Prepare profiles data
        profiles = self.profiles.copy()
        profiles['Price'] = pd.to_numeric(profiles['Price'], errors='coerce') / 1000
        
        # Fix typos in AF column
        profiles['AF'] = profiles['AF'].replace({
            'Current Measuremnt and time stamped fault records on mobile app': 'Current Measurement and time stamped fault records on mobile app',
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, contorl, terminal temp alarm)': 'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)'
        })
        
        # Encode profiles
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=int
        )
        
        # Rename columns
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
        profiles_encoded.rename(columns=rename_dict, inplace=True)
        
        # Ensure all X_cols are present
        X_cols = [
            'Price',
            'Size_Perf_High',
            'Adv_Feat_ElecLife',
            'Adv_Feat_Health',
            'Adv_Feat_ModbusBasic',
            'Adv_Feat_Current',
            'Adv_Feat_ModbusEth',
            'Adv_Feat_Safety',
            'Adv_Feat_TempMon'
        ]
        for col in X_cols:
            if col not in profiles_encoded.columns:
                profiles_encoded[col] = 0
        
        # Convert price_scenarios to DataFrame if dict
        if isinstance(price_scenarios, dict):
            scenarios_df = pd.DataFrame(price_scenarios)
        else:
            scenarios_df = price_scenarios.copy()
        
        # Ensure prices are in thousands
        scenarios_df = scenarios_df / 1000
        
        # Initialize shares DataFrame
        shares = pd.DataFrame(index=profiles_encoded.index, columns=scenarios_df.columns)
        
        # Compute utilities and shares for each scenario
        for scenario in scenarios_df.columns:
            temp_profiles = profiles_encoded.copy()
            for profile_idx in scenarios_df.index:
                if profile_idx in temp_profiles.index:
                    temp_profiles.loc[profile_idx, 'Price'] = scenarios_df.loc[profile_idx, scenario]
            
            X = temp_profiles[X_cols]
            X = sm.add_constant(X, has_constant='add')
            V = X @ self.utilities
            exp_V = np.exp(V)
            sum_exp_V = exp_V.sum()
            probabilities = exp_V / sum_exp_V
            shares[scenario] = probabilities
        
        # Prepare profile labels for x-axis
        if profile_labels is None:
            profile_labels = [f'Profile {i}' for i in profiles_encoded.index]
        elif isinstance(profile_labels, dict):
            profile_labels = [profile_labels.get(i, f'Profile {i}') for i in profiles_encoded.index]
        elif isinstance(profile_labels, list):
            if len(profile_labels) != len(profiles_encoded.index):
                raise ValueError("profile_labels list length must match number of profiles")
        
        # Plot line plot
        if plot:
            # Prepare data for seaborn
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
        
        return shares * 100  # Return shares as percentages


# Update the main block to test the new method
if __name__ == "__main__":
    # Load data
    profiles = pd.read_excel('profiles.xlsx', index_col=0)
    choices = pd.read_excel("CBC_Data_Final_09Jun25.xlsx")
    groups = pd.read_excel("A2_9Jun25.xlsx")
    
    # Ensure numeric types for choices
    choices['respondent_id'] = pd.to_numeric(choices['respondent_id'], errors='coerce')
    choices['choice_set'] = pd.to_numeric(choices['choice_set'], errors='coerce')
    choices['chosen_profile'] = pd.to_numeric(choices['chosen_profile'], errors='coerce')
    
    # Remove Unnamed columns
    for df in [profiles, choices, groups]:
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"Dropping Unnamed columns from {df}: {unnamed_cols}")
            df.drop(columns=unnamed_cols, inplace=True)
    
    # Debug: Print column types
    print("Profiles dtypes:\n", profiles.dtypes)
    print("Choices dtypes:\n", choices.dtypes)
    print("Groups dtypes:\n", groups.dtypes)
    
    try:
        analyzer = DiscreteChoiceAnalyzer(profiles, choices, groups)
        choice_data = analyzer.prepare_data()
        utilities = analyzer.fit_model()
        importance = analyzer.calculate_feature_importance()
        print("Utilities:\n", utilities)
        print("Feature Importance:\n", importance)
        analyzer.plot_utilities()
        analyzer.plot_feature_importance()
        analyzer.plot_price_elasticity()
        
        # Define price scenarios (example)
        price_scenarios = {
            'Baseline': profiles['Price'].to_dict(),  # Original prices
            '10% Increase': (profiles['Price'] * 1.1).to_dict(),  # 10% price increase
            '10% Decrease': (profiles['Price'] * 0.9).to_dict(),  # 10% price decrease
            'Custom': {i: profiles['Price'].mean() for i in profiles.index}  # Same price for all
        }
        
        # Evaluate price scenarios
        shares = analyzer.evaluate_price_scenario_lp(price_scenarios, plot=True)
        print("Profile Shares (%):\n", shares.round(2))
        
        print("Done")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    # Add this method to the DiscreteChoiceAnalyzer class (replace the existing evaluate_price_scenario)

