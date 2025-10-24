import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import random
import textwrap

class EnhancedBayesianChoiceAnalyzer:
    def __init__(self, profiles, choices, groups):
        if not isinstance(profiles, pd.DataFrame):
            raise TypeError("profiles must be a pandas DataFrame")
        if not isinstance(choices, pd.DataFrame):
            raise TypeError("choices must be a pandas DataFrame")
        if not isinstance(groups, pd.DataFrame):
            raise TypeError("groups must be a pandas DataFrame")
        self.profiles = profiles
        self.choices = choices
        self.groups = groups
        self.model = None
        self.trace = None
        self.choice_data = None
        self.X_cols = None
    def prepare_data(self):
        """
        Prepare data for conditional logit model, ensuring numeric output.
        """
        # self.profiles.to_excel("dcm02_profiles.xlsx", index=True)
        
        # Scale Price and encode dummies as int
        profiles = self.profiles.copy()
        profiles['Price'] = profiles['Price'] / 1000
        profiles_encoded = pd.get_dummies(
            profiles,
            columns=['SP', 'AF'],
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
        numeric_cols = [col for col in self.choice_data.columns if col not in ['respondent_id', 'choice_set', 'group', 'chosen']]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.choice_data[col]):
                raise ValueError(f"Non-numeric column {col}: {self.choice_data[col].dtype}")
        self.choice_data.to_excel("choice_data_extended bdcm01.xlsx",index=False)
        self.X_cols = [col for col in self.choice_data.columns 
                      if col not in ['respondent_id', 'choice_set', 'chosen', 'group']]        
        return self.choice_data

    def ex_prepare_data(self):
        """Prepare data with additional feature engineering."""
        profiles = self.profiles.copy()
        profiles['Price'] = profiles['Price'] / 1000  # Scale price for better numerics
        
        # Create dummy variables with proper encoding
        profiles_encoded = pd.get_dummies(
                            profiles,
                            columns=['SP', 'AF'],
                            drop_first=True,
                            dtype=int
                        )
        
        # Merge with choices and groups
        choice_data = self.choices.merge(
                            profiles_encoded, 
                            left_on='chosen_profile', 
                            right_index=True, 
                            validate="many_to_one"
                        ).merge(
                            self.groups, 
                            on='respondent_id', 
                            validate="many_to_one"
                        )
        
        # Create long format choice data
        choice_sets = []
        for _, row in choice_data.iterrows():
            profiles_presented = list(row['profiles_presented'])
            if not isinstance(profiles_presented, list):
                    raise ValueError(f"profiles_presented must be a list, got {type(profiles_presented)}")
            for idx in profiles_presented:
                profile = profiles_encoded.loc[idx]
                choice_sets.append({
                    **profile.to_dict(),
                    'respondent_id': row['respondent_id'],
                    'choice_set': row['choice_set'],
                    'group': row['group'],
                    'chosen': 1 if idx == row['chosen_profile'] else 0
                })
        
        self.choice_data = pd.DataFrame(choice_sets)
        self.choice_data.to_excel("choice_data_extended bdcm01.xlsx",index=False)
        self.X_cols = [col for col in self.choice_data.columns 
                      if col not in ['respondent_id', 'choice_set', 'chosen', 'group']]
        
        return self.choice_data

    def build_hierarchical_model(self):
        """Build hierarchical Bayesian choice model with enhanced features."""
        if self.choice_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        # n_alternatives = len(self.choice_data['choice_set'].value_counts())
        # if n_alternatives != 3:  # Assuming 3 alternatives per set
        #     raise ValueError(f"Expected 3 alternatives per choice set, found {n_alternatives}")
        
        # Group indices for hierarchical modeling
        group_idx, groups = pd.factorize(self.choice_data['group'])
        respondent_idx, respondents = pd.factorize(self.choice_data['respondent_id'])
        
        coords = {
            "features": self.X_cols,
            "groups": groups,
            "respondents": respondents,
            "obs": np.arange(len(self.choice_data))
        }
        print(f"Coords: {coords}")
        
        with pm.Model(coords=coords) as self.model:
            # Hyperpriors for group-level parameters
            mu_beta = pm.Normal("mu_beta", 0, 10, dims=("features", "groups"))
            sigma_beta = pm.HalfNormal("sigma_beta", 5, dims=("features", "groups"))
            
            # Respondent-level random effects
            beta_offset = pm.Normal(
                "beta_offset", 
                0, 
                1, 
                dims=("features", "respondents")  # Fixed this line
            )
            
            # Combine to get individual coefficients
            betas = pm.Deterministic(
                "betas",
                mu_beta[:, group_idx[respondent_idx]] + 
                sigma_beta[:, group_idx[respondent_idx]] * beta_offset,
                dims=("features", "obs")
            )
            
            # Utility computation
            X_data = pm.MutableData("X_data", self.choice_data[self.X_cols].values)
            utilities = pm.math.dot(X_data, betas)
            
            # Choice probability (softmax)
            choice_prob = pm.math.softmax(
                utilities.reshape((-1, 3)),  # Assuming 3 alternatives per choice set
                axis=1
            )
            
            # Likelihood
            choice_idx = pm.MutableData("choice_idx", self.choice_data['chosen'].values)
            pm.Categorical(
                "choices", 
                p=choice_prob, 
                observed=choice_idx
            )
            
            # Willingness-to-pay calculations
            with self.model:
                for i, feat in enumerate(self.X_cols[1:], start=1):  # Skip price
                    pm.Deterministic(
                        f"wtp_{feat}",
                        -betas[i] / (betas[0] + 1e-10),  # Added small epsilon to prevent division by zero
                        dims="obs"
                    )
            
            # Price elasticity calculations
            price_elasticity = pm.Deterministic(
                "price_elasticity",
                betas[0] * X_data[:, 0] * (1 - choice_prob[:, 0]),  # Own-price elasticity
                dims="obs"
            )
            
        return self.model

    def fit_model(self, draws=2000, tune=1000, chains=4):
        """Fit the hierarchical model with enhanced diagnostics."""
        if self.model is None:
            self.build_hierarchical_model()
            
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.9,
                return_inferencedata=True
            )
            
            # Check convergence
            print(az.summary(self.trace, var_names=["mu_beta", "sigma_beta"]))
            az.plot_trace(self.trace, var_names=["mu_beta", "sigma_beta"])
            plt.tight_layout()
            plt.savefig('convergence_diagnostics.png', dpi=300)
            plt.close()
        if hasattr(self, 'trace'):
            self.trace = az.extract_dataset(self.trace)  # Reduce memory usage            
        return self.trace

    def analyze_results(self):
        """Comprehensive analysis of model results."""
        if self.trace is None:
            raise ValueError("Model not fitted yet. Call fit_model() first.")
            
        # 1. Plot utilities with uncertainty
        self._plot_utilities_with_uncertainty()
        
        # 2. Calculate and plot feature importance
        self._calculate_feature_importance()
        
        # 3. Analyze willingness-to-pay
        self._analyze_willingness_to_pay()
        
        # 4. Price elasticity analysis
        self._analyze_price_elasticity()
        
        # 5. Market segmentation analysis
        self._analyze_market_segments()

    def _plot_utilities_with_uncertainty(self):
        """Plot utilities with credible intervals."""
        az.plot_forest(
            self.trace, 
            var_names=["mu_beta"],
            combined=True,
            figsize=(10, 6),
            hdi_prob=0.95
        )
        plt.title("Feature Utilities with 95% Credible Intervals")
        plt.tight_layout()
        plt.savefig('utilities_with_uncertainty.png', dpi=300)
        plt.close()

    def _calculate_feature_importance(self):
        """Calculate feature importance with uncertainty."""
        # Get posterior samples of coefficients
        betas = az.extract(self.trace, var_names="mu_beta")
        
        # Calculate importance metrics
        importance = {}
        for i, feat in enumerate(self.X_cols):
            if feat == 'Price':
                # For price, use absolute coefficient scaled by price range
                price_range = (self.profiles['Price'].max() - self.profiles['Price'].min()) / 1000
                importance[feat] = np.abs(betas.sel(features=i)) * price_range
            else:
                # For other features, use range of utilities
                levels = [col for col in self.X_cols if feat in col]
                if levels:
                    level_indices = [self.X_cols.index(l) for l in levels]
                    max_util = betas.sel(features=level_indices).max(dim='features')
                    min_util = betas.sel(features=level_indices).min(dim='features')
                    importance[feat] = max_util - min_util
        
        # Normalize importance
        total_importance = sum(imp.mean() for imp in importance.values())
        self.feature_importance = {
            feat: (imp / total_importance).mean().item()
            for feat, imp in importance.items()
        }
        
        # Plot
        importance_df = pd.DataFrame({
            'Feature': list(self.feature_importance.keys()),
            'Importance': list(self.feature_importance.values())
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()
        
        return self.feature_importance

    def _analyze_willingness_to_pay(self, price_col='Price', hdi_prob=0.95):
        """
        Analyze and plot willingness-to-pay for features with enhanced robustness.
        
        Parameters:
        -----------
        price_col : str (default='Price')
            Name of the price column in the data
        hdi_prob : float (default=0.95)
            Probability for the highest density interval
        """
        wtp_results = {}
        
        # Get the actual price column name from X_cols
        price_col = next((col for col in self.X_cols if 'Price' in col), price_col)
        
        for i, feat in enumerate(self.X_cols):
            # Skip price feature and any derived price features
            if feat == price_col or 'Price' in feat:
                continue
                
            wtp_var = f"wtp_{feat}"
            
            # Check if WTP variable exists in trace
            if wtp_var not in self.trace.posterior:
                print(f"Warning: WTP variable {wtp_var} not found in trace")
                continue
                
            try:
                wtp = az.extract(self.trace, var_names=wtp_var)
                
                # Convert to original price scale (undo the /1000 scaling)
                wtp_values = wtp.values * 1000
                
                wtp_results[feat] = {
                    'mean': wtp_values.mean(),
                    'hdi_95': az.hdi(wtp_values, hdi_prob=hdi_prob),
                    'median': np.median(wtp_values),
                    'std': wtp_values.std()
                }
            except Exception as e:
                print(f"Error processing WTP for {feat}: {str(e)}")
                continue
        
        if not wtp_results:
            print("No valid WTP results to display")
            return None
        
        # Create formatted feature names for display
        display_names = {
            feat: textwrap.fill(feat.replace('_', ' '), 50) 
            for feat in wtp_results.keys()
        }
        
        # Plot WTP with enhanced formatting
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(wtp_results))
        
        for i, (feat, vals) in enumerate(wtp_results.items()):
            lower_error = vals['mean'] - vals['hdi_95'][0]
            upper_error = vals['hdi_95'][1] - vals['mean']
            xerr = [[lower_error], [upper_error]]  # Explicit 2x1 array

            plt.errorbar(
                x=vals['mean'],
                y=y_pos[i],
                xerr=xerr,  # Now using pre-calculated errors
                fmt='o',
                markersize=8,
                capsize=5,
                label=display_names[feat]
            )
            # Add median as a different marker
            plt.scatter(
                x=vals['median'],
                y=y_pos[i],
                marker='x',
                color='red',
                s=100
            )
        
        plt.yticks(y_pos, [display_names[feat] for feat in wtp_results.keys()])
        plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
        
        # Add reference lines for typical price points
        price_ref = self.profiles['Price'].median()
        plt.axvline(price_ref, color='green', linestyle=':', alpha=0.5, 
                label=f'Median Price ({price_ref:,.0f})')
        plt.axvline(-price_ref, color='green', linestyle=':', alpha=0.5)
        
        plt.title("Willingness-to-Pay Analysis\n(Mean ± 95% HDI, X=Median)")
        plt.xlabel("WTP (Currency Units)")
        plt.grid(True, axis='x', alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('wtp_analysis_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return results as DataFrame for further analysis
        wtp_df = pd.DataFrame.from_dict(wtp_results, orient='index')
        wtp_df.index.name = 'Feature'
        wtp_df.to_excel('wtp_results.xlsx')
        
        return wtp_df

    def _analyze_price_elasticity(self):
        """Analyze price elasticity across different price points."""
        if 'price_elasticity' not in self.trace.posterior:
            raise ValueError("Price elasticity not calculated in model")
            
        # Get elasticity samples
        elasticity = az.extract(self.trace, var_names="price_elasticity")
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        az.plot_posterior(elasticity, hdi_prob=0.95)
        plt.title("Price Elasticity Distribution")
        plt.xlabel("Elasticity")
        plt.tight_layout()
        plt.savefig('price_elasticity_distribution.png', dpi=300)
        plt.close()
        
        # Analyze by price level
        price_levels = pd.qcut(self.choice_data['Price'], q=4)
        elasticity_by_price = pd.DataFrame({
                'Price Level': price_levels,
                'Elasticity': elasticity.mean(dim='sample')
            }).groupby('Price Level')['Elasticity'].agg(['mean', 'std'])
        
        # Plot
        plt.figure(figsize=(10, 6))
        elasticity_by_price['mean'].plot(kind='bar', yerr=elasticity_by_price['std'])
        plt.title("Price Elasticity by Price Level")
        plt.ylabel("Mean Elasticity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('elasticity_by_price_level.png', dpi=300)
        plt.close()
        
        return elasticity_by_price

    def _analyze_market_segments(self):
        """Compare preferences across market segments."""
        if 'mu_beta' not in self.trace.posterior:
            raise ValueError("Hierarchical structure not found in model")
            
        # Get group-level parameters
        mu_beta = az.extract(self.trace, var_names="mu_beta")
        
        # Compare groups for each feature
        comparisons = {}
        for i, feat in enumerate(self.X_cols):
            group_comparison = {}
            for j, group in enumerate(mu_beta.coords['groups'].values):
                samples = mu_beta.sel(features=i, groups=group)
                group_comparison[group] = {
                    'mean': samples.mean().item(),
                    'hdi_95': az.hdi(samples.values, hdi_prob=0.95)
                }
            comparisons[feat] = group_comparison
        
        # Plot comparison for each feature
        for feat, group_data in comparisons.items():
            plt.figure(figsize=(10, 6))
            for i, (group, vals) in enumerate(group_data.items()):
                plt.errorbar(
                    vals['mean'], i,
                    xerr=[[vals['mean'] - vals['hdi_95'][0]], 
                        [vals['hdi_95'][1] - vals['mean']]],
                    fmt='o',
                    label=group)
            plt.yticks(range(len(group_data)), list(group_data.keys()))
            plt.axvline(0, color='gray', linestyle='--')
            plt.title(f"Utility Comparison for {feat}")
            plt.xlabel("Utility")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'group_comparison_{feat}.png', dpi=300)
            plt.close()
        
        return comparisons

    def predict_market_shares(self, new_profiles):
        """
            Predict market shares for new product profiles with uncertainty.
            Parameters:
            -----------
            new_profiles : pd.DataFrame
                DataFrame containing product profiles with same features as training data
                Must include columns: Size_Performance, Advanced_Feature, Price
                
            Returns:
            --------
            pd.DataFrame
                Contains columns:
                - profile: Index of input profiles
                - market_share: Predicted market share (mean)
                - share_lower: Lower bound of HDI
                - share_upper: Upper bound of HDI
        """
        if self.trace is None:
            raise ValueError("Model not fitted yet. Call fit_model() first.")
            
        # Prepare new profiles
        new_profiles = new_profiles.copy()
        new_profiles['Price'] = new_profiles['Price'] / 1000
        new_profiles_encoded = pd.get_dummies(
            new_profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=int
        )
        
        # Ensure same columns as training data
        missing_cols = set(self.X_cols) - set(new_profiles_encoded.columns)
        for col in missing_cols:
            new_profiles_encoded[col] = 0
        new_profiles_encoded = new_profiles_encoded[self.X_cols]
        
        # Predict for each posterior sample
        with self.model:
            pm.set_data({"X_data": new_profiles_encoded.values})
            ppc = pm.sample_posterior_predictive(
                self.trace,
                var_names=["choices"]
            )
        
        # Calculate market shares
        shares = ppc['choices'].mean(axis=0)  # Average over samples
        shares_hdi = az.hdi(ppc['choices'], hdi_prob=0.95)
        
        result = pd.DataFrame({
            'profile': new_profiles.index,
            'market_share': shares,
            'share_lower': shares_hdi[:, 0],
            'share_upper': shares_hdi[:, 1]
        })
        
        return result

    def simulate_scenarios(self, scenarios):
        """
        Simulate market response to different scenarios.
        
        Parameters:
        -----------
        scenarios : list of dict
            Each dict should contain:
            - 'name': scenario name
            - 'changes': dict of feature changes (e.g., {'Price': 1.1} for 10% price increase)
            - 'target_profiles': list of profile indices to apply changes to
        """
        results = []
        original_shares = self.predict_market_shares(self.profiles)
        
        for scenario in scenarios:
            # Apply changes to profiles
            modified_profiles = self.profiles.copy()
            for idx in scenario.get('target_profiles', self.profiles.index):
                for feature, change in scenario['changes'].items():
                    if feature == 'Price':
                        modified_profiles.loc[idx, feature] *= change
                    else:
                        modified_profiles.loc[idx, feature] = change
            
            # Predict shares
            modified_shares = self.predict_market_shares(modified_profiles)
            
            # Calculate changes
            comparison = original_shares.merge(
                modified_shares, 
                on='profile', 
                suffixes=('_original', '_modified')
            )
            comparison['scenario'] = scenario['name']
            results.append(comparison)
        
        return pd.concat(results)

if __name__=="__main__":
    # Profiles data        
    profiles = pd.read_excel('profiles.xlsx', index_col=0)
    choices = pd.read_excel("CBC_Data_Final_09Jun25.xlsx")
    groups = pd.read_excel("A2_9Jun25.xlsx")
    # Initialize and fit model
    analyzer = EnhancedBayesianChoiceAnalyzer(profiles, choices, groups)
    analyzer.prepare_data()
    trace = analyzer.fit_model()

    # Analyze results
    analyzer.analyze_results()

    # Scenario simulation
    scenarios = [
        {
            'name': '10% Price Increase on Premium Products',
            'changes': {'Price': 1.10},
            'target_profiles': [0, 2, 4, 6]  # Indices of premium products
        },
        {
            'name': 'New Feature Introduction',
            'changes': {'AF': 'Current Measurement and time stamped fault records on mobile app'},
            'target_profiles': [8, 9]
        }
    ]

    simulation_results = analyzer.simulate_scenarios(scenarios)
