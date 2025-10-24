import pandas as pd
import numpy as np
import random
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QToolTip
from PySide6.QtCharts import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis, QLineSeries, QScatterSeries
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtCore import Qt, QPointF

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
        self.trace = None  # PyMC posterior trace
        self.utilities = None  # Posterior means
        self.feature_importance = None
        self.choice_data = None

    def prepare_data(self):
        """
        Prepare data for Bayesian conditional logit model, ensuring numeric output.
        """
        profiles = self.profiles.copy()
        profiles['Price'] = pd.to_numeric(profiles['Price'], errors='coerce') / 1000
        
        # Fix typos in AF column
        profiles['AF'] = profiles['AF'].replace({
            'Current Measuremnt and time stamped fault records on mobile app': 'Current Measurement and time stamped fault records on mobile app',
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, contorl, terminal temp alarm)': 'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)'
        })
        
        print("Unique SP values:", profiles['SP'].unique())
        print("Unique AF values:", sorted(profiles['AF'].unique()))
        print("Dropped AF level (first alphabetically):", sorted(profiles['AF'].unique())[0])
        
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=int
        )
        
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
        
        print("profiles_encoded columns:", profiles_encoded.columns.tolist())
        print("profiles_encoded shape:", profiles_encoded.shape)
        
        # Validate choices
        if not all(self.choices['chosen_profile'].isin(profiles_encoded.index)):
            raise ValueError("Invalid chosen_profile values in choices")
        
        # Validate groups merge
        if not all(self.choices['respondent_id'].isin(self.groups['respondent_id'])):
            missing_ids = self.choices[~self.choices['respondent_id'].isin(self.groups['respondent_id'])]['respondent_id'].unique()
            raise ValueError(f"respondent_id values in choices not found in groups: {missing_ids}")
        
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
        
        # Convert bool to int
        for col in self.choice_data.columns:
            if self.choice_data[col].dtype == bool:
                self.choice_data[col] = self.choice_data[col].astype(int)
        
        # Remove Unnamed columns
        unnamed_cols = [col for col in self.choice_data.columns if 'Unnamed' in col]
        if unnamed_cols:
            print(f"Dropping Unnamed columns: {unnamed_cols}")
            self.choice_data.drop(columns=unnamed_cols, inplace=True)
        
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
        Fit Bayesian conditional logit model using PyMC with group-level effects.
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
        
        # Prepare data
        choice_data = self.choice_data
        X = choice_data[X_cols].values
        y = choice_data['chosen'].values
        respondent_ids = choice_data['respondent_id'].values
        choice_set_ids = choice_data['choice_set'].values
        groups = choice_data['group'].astype('category').cat.codes.values
        group_names = choice_data['group'].astype('category').cat.categories
        
        # Unique choice sets per respondent
        unique_choice_sets = choice_data.groupby(['respondent_id', 'choice_set']).size().reset_index()[['respondent_id', 'choice_set']]
        
        with pm.Model() as self.model:
            # Priors for group-level coefficients (mean and sd)
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=2, shape=len(X_cols))  # Mean for each feature
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=2, shape=len(X_cols))  # SD for group variation
            
            # Group-specific coefficients
            beta_group = pm.Normal('beta_group', 
                                  mu=mu_beta, 
                                  sigma=sigma_beta, 
                                  shape=(len(group_names), len(X_cols)))
            
            # Compute utilities for each alternative
            utilities = pm.Deterministic('utilities', 
                                       pm.math.dot(X, beta_group[groups].T))
            
            # Reshape utilities for each choice set
            choice_idx = []
            utility_values = []
            for _, row in unique_choice_sets.iterrows():
                mask = (choice_data['respondent_id'] == row['respondent_id']) & (choice_data['choice_set'] == row['choice_set'])
                choice_idx.append(np.where(mask)[0])
                utility_values.append(utilities[mask])
            
            # Softmax likelihood for choices
            for i, (idx, util) in enumerate(zip(choice_idx, utility_values)):
                pm.Categorical(f'choice_{i}', p=pm.math.softmax(util), observed=y[idx])
            
            # Sample from posterior
            self.trace = pm.sample(1000, tune=1000, chains=2, random_seed=42, return_inferencedata=True)
        
        # Extract posterior means for utilities
        self.utilities = pd.Series(self.trace.posterior['mu_beta'].mean(dim=['chain', 'draw']).values, 
                                 index=X_cols)
        
        print("Posterior summary:\n", az.summary(self.trace, var_names=['mu_beta'], hdi_prob=0.95))
        return self.utilities

    def calculate_feature_importance(self):
        """
        Calculate feature importance using posterior means.
        """
        if self.utilities is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        # Scale price range to model units (thousands)
        price_range = (self.profiles['Price'].max() - self.profiles['Price'].min()) / 1000
        price_util = abs(self.utilities['Price']) * price_range
        
        size_utils = self.utilities.filter(like='Size_Perf_').abs()
        adv_feature_utils = self.utilities.filter(like='Adv_Feat_').abs()
        
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
        """
        Plot posterior mean utilities with 95% HDI.
        """
        if self.trace is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        utilities = self.trace.posterior['mu_beta'].mean(dim=['chain', 'draw']).values
        hdi = az.hdi(self.trace, var_names=['mu_beta'], hdi_prob=0.95)['mu_beta'].values
        feature_names = [
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
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(range(len(feature_names)), utilities, 
                    yerr=[utilities - hdi[:, 0], hdi[:, 1] - utilities], 
                    fmt='o', capsize=5)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.title('Posterior Mean Utilities with 95% HDI')
        plt.ylabel('Utility')
        plt.xlabel('Features')
        plt.tight_layout()
        plt.savefig('bayesian_utilities.png')
        plt.close()

    def plot_feature_importance(self):
        """
        Plot feature importance based on posterior means.
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance is not calculated yet. Call calculate_feature_importance() first.")
        
        importance_df = pd.DataFrame(list(self.feature_importance.items()), 
                                   columns=['Feature', 'Importance'])
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Importance', y='Feature', data=importance_df.sort_values('Importance', ascending=False))
        plt.title('Feature Importance (Bayesian)')
        plt.savefig('bayesian_feature_importance.png')
        plt.close()

    def evaluate_price_scenario_lp(self, price_scenarios, profile_labels=None, plot=True):
        """
        Evaluate profile shares for price scenarios using posterior predictive distribution.
        """
        if self.trace is None:
            raise ValueError("Model is not fitted yet. Call fit_model() first.")
        
        profiles = self.profiles.copy()
        profiles['Price'] = pd.to_numeric(profiles['Price'], errors='coerce') / 1000
        
        # Fix typos in AF column
        profiles['AF'] = profiles['AF'].replace({
            'Current Measuremnt and time stamped fault records on mobile app': 'Current Measurement and time stamped fault records on mobile app',
            'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, contorl, terminal temp alarm)': 'Scalable connectivity at breaker level- Basic Modbus (Breaker Status, control, terminal temp alarm)'
        })
        
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=int
        )
        
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
        
        if isinstance(price_scenarios, dict):
            scenarios_df = pd.DataFrame(price_scenarios)
        else:
            scenarios_df = price_scenarios.copy()
        
        scenarios_df = scenarios_df / 1000
        shares = pd.DataFrame(index=profiles_encoded.index, columns=scenarios_df.columns)
        
        # Use posterior samples to compute shares
        beta_samples = self.trace.posterior['mu_beta'].stack(sample=('chain', 'draw')).values
        for scenario in scenarios_df.columns:
            temp_profiles = profiles_encoded.copy()
            for profile_idx in scenarios_df.index:
                if profile_idx in temp_profiles.index:
                    temp_profiles.loc[profile_idx, 'Price'] = scenarios_df.loc[profile_idx, scenario]
            
            X = temp_profiles[X_cols].values
            X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add constant
            # Compute utilities across samples
            utilities = np.dot(X, beta_samples)
            exp_utilities = np.exp(utilities)
            sum_exp_utilities = exp_utilities.sum(axis=0)
            probabilities = exp_utilities / sum_exp_utilities
            shares[scenario] = probabilities.mean(axis=1)  # Mean across posterior samples
        
        # Plotting
        if profile_labels is None:
            profile_labels = [f'Profile {i}' for i in profiles_encoded.index]
        elif isinstance(profile_labels, dict):
            profile_labels = [profile_labels.get(i, f'Profile {i}') for i in profiles_encoded.index]
        
        if plot:
            plot_data = shares.reset_index().melt(id_vars='index', var_name='Scenario', value_name='Share')
            plot_data['Profile'] = plot_data['index'].map({i: label for i, label in enumerate(profile_labels)})
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=plot_data, x='Profile', y='Share', hue='Scenario', style='Scenario', markers=True, dashes=False)
            plt.title('Profile Shares Across Price Scenarios (Bayesian)')
            plt.xlabel('Profile')
            plt.ylabel('Choice Probability (Share, %)')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('bayesian_profile_shares_line.png')
            plt.close()
        
        return shares * 100

    def plot_price_elasticity(self):
        """
        Plot price elasticity using posterior predictive distribution.
        """
        if self.trace is None:
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
        X = self.choice_data[X_cols].values
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        beta_samples = self.trace.posterior['mu_beta'].stack(sample=('chain', 'draw')).values
        beta_samples = np.vstack([np.zeros((1, beta_samples.shape[1])), beta_samples])  # Add constant
        
        # Compute utilities and probabilities
        V = np.dot(X, beta_samples)
        exp_V = np.exp(V)
        sum_exp_V = exp_V.groupby(self.choice_data[['respondent_id', 'choice_set']].values).sum()
        P_i = exp_V / sum_exp_V
        E_ii = (1 - P_i) * beta_samples[0] * self.choice_data['Price'].values[:, None]
        
        temp_data = self.choice_data.copy()
        temp_data['E_ii'] = E_ii.mean(axis=1)
        temp_data['original_price'] = temp_data['Price'] * 1000
        
        mean_elasticity = temp_data.groupby('original_price')['E_ii'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='original_price', y='E_ii', data=mean_elasticity, marker='o')
        plt.title('Average Price Elasticity vs. Price (Bayesian)')
        plt.xlabel('Price')
        plt.ylabel('Elasticity')
        plt.savefig('bayesian_price_elasticity.png')
        plt.close()

# QtCharts Visualization (Optional)
class ChartWindow(QMainWindow):
    def __init__(self, shares, profile_labels, title="Profile Shares Across Price Scenarios"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.chart = QChart()
        self.chart.setTitle(title)
        
        # Create line series for each scenario
        colors = ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0']
        for scenario, color in zip(shares.columns, colors[:len(shares.columns)]):
            series = QLineSeries()
            series.setName(scenario)
            for i, share in enumerate(shares[scenario]):
                series.append(i, share)
            series.setPen(QPen(QColor(color), 2))
            self.chart.addSeries(series)
        
        # X-axis (profiles)
        axis_x = QBarCategoryAxis()
        axis_x.append(profile_labels)
        self.chart.setAxisX(axis_x, series)
        
        # Y-axis
        self.axis_y = QValueAxis()
        self.axis_y.setRange(0, 100)
        self.axis_y.setTitleText("Choice Probability (%)")
        self.chart.setAxisY(self.axis_y, series)
        
        chart_view = QChartView(self.chart)
        chart_view.setStyleSheet("border: 2px solid red;")
        layout.addWidget(chart_view)
        
        print("Chart initialized with line plots for scenarios")

if __name__ == "__main__":
    # Load data
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
        utilities = analyzer.fit_model()
        importance = analyzer.calculate_feature_importance()
        print("Utilities (Posterior Means):\n", utilities)
        print("Feature Importance:\n", importance)
        analyzer.plot_utilities()
        analyzer.plot_feature_importance()
        analyzer.plot_price_elasticity()
        
        # Price scenarios
        price_scenarios = {
            'Baseline': profiles['Price'].to_dict(),
            '10% Increase': (profiles['Price'] * 1.1).to_dict(),
            '10% Decrease': (profiles['Price'] * 0.9).to_dict(),
            'Custom': {i: profiles['Price'].mean() for i in profiles.index}
        }
        
        # Evaluate price scenarios
        profile_labels = [f'Profile {i}' for i in profiles.index]
        shares = analyzer.evaluate_price_scenario_lp(price_scenarios, profile_labels=profile_labels, plot=True)
        print("Profile Shares (%):\n", shares.round(2))
        
        # QtCharts visualization (optional)
        app = QApplication(sys.argv)
        window = ChartWindow(shares, profile_labels, "Bayesian Profile Shares Across Price Scenarios")
        window.show()
        sys.exit(app.exec())
        
        print("Done")
    except Exception as e:
        print(f"Error occurred: {str(e)}")