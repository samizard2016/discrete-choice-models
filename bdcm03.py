import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from itertools import combinations
import textwrap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBayesianChoiceAnalyzer:
    def __init__(self, profiles, choices, groups, rename_dict=None):
        """
        Initialize the Bayesian choice analyzer with data validation and feature renaming.

        Args:
            profiles (pd.DataFrame): Product profiles with features.
            choices (pd.DataFrame): Respondent choices with respondent_id, choice_set, chosen_profile, profiles_presented.
            groups (pd.DataFrame): Respondent group assignments.
            rename_dict (dict, optional): Dictionary to rename feature columns for clarity.
        """
        if not all(isinstance(df, pd.DataFrame) for df in [profiles, choices, groups]):
            raise TypeError("All inputs (profiles, choices, groups) must be pandas DataFrames")
        
        self.profiles = profiles.copy()
        self.choices = choices.copy()
        self.groups = groups.copy()
        self.rename_dict = rename_dict or {}
        self.model = None
        self.trace = None
        self.choice_data = None
        self.X_cols = None
        self.n_alternatives = 3
        self.n_unique_choice_sets = 24
        self.n_respondents = 295
        self.tasks_per_respondent = 4
        self.n_choice_sets = self.n_respondents * self.tasks_per_respondent  # 400

    def prepare_data(self):
        """
        Prepare data for conditional logit model with vectorized operations and feature renaming.

        Returns:
            pd.DataFrame: Prepared choice data with numeric features.
        """
        logger.info("Preparing data for Bayesian choice model")
        profiles = self.profiles.copy()
        profiles['Price'] = profiles['Price'] / 1000  # Normalize price to thousands
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

        # Apply renaming dictionary to encoded columns
        if self.rename_dict:
            profiles_encoded = profiles_encoded.rename(columns=self.rename_dict)

        # Validate choices
        if not self.choices['chosen_profile'].isin(profiles_encoded.index).all():
            raise ValueError("Some chosen_profile values are not in profiles index")

        # Merge data
        choice_data = self.choices.merge(
            profiles_encoded, left_on='chosen_profile', right_index=True, validate="many_to_one"
        ).merge(self.groups, on='respondent_id', validate="many_to_one")

        # Validate data structure
        if len(choice_data['respondent_id'].unique()) != self.n_respondents:
            raise ValueError(f"Expected {self.n_respondents} respondents, found {len(choice_data['respondent_id'].unique())}")
        if not (choice_data.groupby('respondent_id').size() == self.tasks_per_respondent).all():
            raise ValueError(f"Expected {self.tasks_per_respondent} tasks per respondent")
        if len(choice_data['choice_set'].unique()) != self.n_unique_choice_sets:
            raise ValueError(f"Expected {self.n_unique_choice_sets} unique choice sets")
        if len(choice_data) != self.n_choice_sets:
            raise ValueError(f"Expected {self.n_choice_sets} respondent-choice set combinations")

        # Vectorized preparation of choice sets
        choice_sets = []
        for _, row in choice_data.iterrows():
            try:
                profiles_presented = [int(v) for v in row['profiles_presented'][1:-1].split(',')]
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid profiles_presented format: {row['profiles_presented']}")
            if len(profiles_presented) != self.n_alternatives:
                raise ValueError(f"Choice set {row['choice_set']} has {len(profiles_presented)} alternatives")

            for idx in profiles_presented:
                if idx not in profiles_encoded.index:
                    raise ValueError(f"Profile index {idx} not found in profiles")
                profile = profiles_encoded.loc[idx]
                choice_sets.append({
                    **profile.to_dict(),
                    'respondent_id': int(row['respondent_id']),
                    'choice_set': int(row['choice_set']),
                    'group': str(row['group']),
                    'chosen': 1 if idx == row['chosen_profile'] else 0
                })

        self.choice_data = pd.DataFrame(choice_sets)
        self.choice_data = self.choice_data.astype({col: int for col in self.choice_data.select_dtypes('bool').columns})

        # Validate numeric columns
        numeric_cols = [col for col in self.choice_data.columns if col not in ['respondent_id', 'choice_set', 'group', 'chosen']]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.choice_data[col]):
                raise ValueError(f"Non-numeric column {col}: {self.choice_data[col].dtype}")

        # Validate total observations
        expected_obs = self.n_choice_sets * self.n_alternatives
        if len(self.choice_data) != expected_obs:
            raise ValueError(f"Expected {expected_obs} observations, got {len(self.choice_data)}")

        logger.info(f"Data prepared: {self.n_respondents} respondents, {self.tasks_per_respondent} tasks, "
                    f"{self.n_unique_choice_sets} choice sets, {expected_obs} observations")

        self.choice_data.to_excel("choice_data_extended.xlsx", index=False)
        self.X_cols = [col for col in self.choice_data.columns 
                       if col not in ['respondent_id', 'choice_set', 'chosen', 'group', 'Unnamed: 0']]
        logger.info(f"Features ({len(self.X_cols)}): {self.X_cols}")

        return self.choice_data

    def build_model(self):
        """
        Build a non-hierarchical Bayesian choice model using PyMC.

        Returns:
            pm.Model: Configured PyMC model.
        """
        if self.choice_data is None:
            raise ValueError("Run prepare_data() first")

        logger.info("Building Bayesian choice model")
        choice_set_idx = self.choice_data.groupby(['respondent_id', 'choice_set']).ngroup()

        coords = {
            "features": self.X_cols,
            "obs": np.arange(len(self.choice_data)),
            "alternatives": np.arange(self.n_alternatives),
            "obs_id": np.arange(self.n_choice_sets)
        }

        with pm.Model(coords=coords) as self.model:
            betas = pm.Normal("betas", 0, 10, dims="features")
            X_data = pm.Data("X_data", self.choice_data[self.X_cols].values, dims=("obs", "features"))
            utilities = pm.math.dot(X_data, betas)
            utilities_reshaped = utilities.reshape((self.n_choice_sets, self.n_alternatives))
            choice_prob = pm.math.softmax(utilities_reshaped, axis=1)

            chosen_alternative = np.array([
                i % self.n_alternatives for i, chosen in enumerate(self.choice_data['chosen']) if chosen == 1
            ])

            pm.Categorical("choices", p=choice_prob, observed=chosen_alternative, dims="obs_id")

            # WTP calculations
            for i, feat in enumerate(self.X_cols[1:], start=1):
                pm.Deterministic(f"wtp_{feat}", -betas[i] / (betas[0] + 1e-10))

            pm.Deterministic(
                "price_elasticity",
                betas[0] * X_data[:, 0] * (1 - choice_prob.flatten()[:len(self.choice_data)]),
                dims="obs"
            )

        return self.model

    def fit_model(self, n_iterations=50000, method='advi'):
        """
        Fit the model using ADVI or MCMC.

        Args:
            n_iterations (int): Number of iterations for ADVI (default: 50000).
            method (str): Optimization method ('advi' or 'mcmc').

        Returns:
            az.InferenceData: Model trace.
        """
        if self.model is None:
            self.build_model()

        logger.info(f"Fitting model with {method.upper()} (iterations: {n_iterations})")
        with self.model:
            if method.lower() == 'advi':
                mean_field = pm.fit(n=n_iterations, method='advi', random_seed=42)
                self.trace = mean_field.sample(1000)
                self.trace = az.convert_to_inference_data(self.trace)
            else:
                self.trace = pm.sample(
                    draws=2000, tune=1000, chains=4, target_accept=0.9, return_inferencedata=True
                )

            summary = az.summary(self.trace, var_names=["betas"])
            logger.info(f"Model summary:\n{summary}")

            az.plot_trace(self.trace, var_names=["betas"], figsize=(12, len(self.X_cols) * 2))
            plt.savefig('convergence_diagnostics.png', dpi=300, bbox_inches='tight')
            plt.close()

        return self.trace

    def analyze_results(self):
        """
        Analyze model results, including utilities, feature importance, WTP, elasticity, and segments.
        """
        if self.trace is None:
            raise ValueError("Run fit_model() first")

        logger.info("Analyzing model results")
        self._plot_utilities_with_uncertainty()
        self._calculate_feature_importance()
        wtp_results = self._analyze_willingness_to_pay()
        self._analyze_price_elasticity()
        self._analyze_market_segments()

        # Generate Chart.js bar chart for WTP
        if wtp_results is not None:
            features = [self.rename_dict.setdefault(feat, feat) for feat in wtp_results.index]
            means = wtp_results['mean'].tolist()
            hdi_lower = wtp_results['hdi_95'].apply(lambda x: x[0]).tolist()
            hdi_upper = wtp_results['hdi_95'].apply(lambda x: x[1]).tolist()
            error_lower = [mean - lower for mean, lower in zip(means, hdi_lower)]
            error_upper = [upper - mean for mean, upper in zip(means, hdi_upper)]

            chart_config = {
                "type": "bar",
                "data": {
                    "labels": features,
                    "datasets": [{
                        "label": "Mean WTP",
                        "data": means,
                        "backgroundColor": "rgba(54, 162, 235, 0.6)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "indexAxis": "y",
                    "scales": {
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Willingness-to-Pay (Currency Units)",
                                "color": "#ffffff"
                            },
                            "grid": {
                                "color": "rgba(255, 255, 255, 0.2)"
                            }
                        },
                        "y": {
                            "title": {
                                "display": True,
                                "text": "Feature",
                                "color": "#ffffff"
                            }
                        }
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "Willingness-to-Pay by Feature (95% HDI)",
                            "color": "#ffffff"
                        },
                        "legend": {
                            "display": False
                        },
                        "tooltip": {
                            "enabled": True
                        }
                    },
                    "errorBars": {
                        "Mean WTP": {
                            "plus": error_upper,
                            "minus": error_lower,
                            "color": "rgba(255, 99, 132, 1)"
                        }
                    }
                }
            }
            logger.info("Generated Chart.js WTP visualization")

    def _plot_utilities_with_uncertainty(self, save_betas_to_excel=True, beta_excel_file='beta_coefficients.xlsx'):
        """
        Plot utilities with 95% credible intervals and optionally save betas to Excel.

        Args:
            save_betas_to_excel (bool): If True, save beta coefficients to Excel. Default is True.
            beta_excel_file (str): Path to the Excel file for saving betas. Default is 'beta_coefficients.xlsx'.

        Returns:
            pd.DataFrame: DataFrame containing beta coefficients and summary statistics if save_betas_to_excel is True.
        """
        logger.info("Plotting utilities with uncertainty and optionally saving betas")
        display_names = {feat: textwrap.fill(self.rename_dict.get(feat, feat).replace('_', ' '), 20) 
                        for feat in self.X_cols}

        # Plot forest plot
        forest_plot = az.plot_forest(
            self.trace,
            var_names=["betas"],
            combined=True,
            figsize=(10, max(6, len(self.X_cols) * 0.8)),
            hdi_prob=0.95,
            textsize=10,
            ess=False,
            r_hat=False,
            coords={"features": self.X_cols}
        )

        ax = forest_plot[0]
        ax.set_yticklabels([display_names[feat] for feat in self.X_cols])
        plt.title("Feature Utilities with 95% Credible Intervals")
        plt.savefig('utilities_with_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save betas to Excel if requested
        if save_betas_to_excel:
            logger.info(f"Extracting and saving beta coefficients to {beta_excel_file}")
            # Extract betas from trace
            betas = az.extract(self.trace, var_names="betas")
            # Compute summary statistics
            beta_summary = {
                'Feature': [],
                'Mean': [],
                'Median': [],
                'Std': [],
                'HDI_2.5%': [],
                'HDI_97.5%': []
            }

            for feat in self.X_cols:
                beta_values = betas.sel(features=feat).values
                hdi = az.hdi(beta_values, hdi_prob=0.95)
                beta_summary['Feature'].append(self.rename_dict.get(feat, feat))
                beta_summary['Mean'].append(beta_values.mean())
                beta_summary['Median'].append(np.median(beta_values))
                beta_summary['Std'].append(beta_values.std())
                beta_summary['HDI_2.5%'].append(hdi[0])
                beta_summary['HDI_97.5%'].append(hdi[1])

            # Create DataFrame
            beta_df = pd.DataFrame(beta_summary)
            # Save to Excel
            beta_df.to_excel(beta_excel_file, index=False)
            logger.info(f"Beta coefficients saved to {beta_excel_file}")

            return beta_df
    def _calculate_feature_importance(self, original_price_range=200):
        """
        Calculate and plot feature importance based on utility ranges, using raw price range for Price.

        Args:
            original_price_range (float): Range of original prices (max - min) in raw currency units.
                                Default is 200 (e.g., for prices [1400, 1500, 1600]).

        Returns:
            dict: Feature importance scores.
        """
        logger.info("Calculating feature importance")
        betas = az.extract(self.trace, var_names="betas")
        importance = {}

        for feat in self.X_cols:
            if feat == 'Price':
                # Use the original price range (in raw currency units, e.g., 200 for [1400, 1500, 1600])
                importance[feat] = np.abs(betas.sel(features=feat)) * (original_price_range / 1000)
                logger.debug(f"Price importance: beta mean = {betas.sel(features=feat).mean().item()}, "
                            f"range = {original_price_range / 1000}, "
                            f"importance mean = {importance[feat].mean().item()}")
            else:
                # For categorical features, use the range of utilities across levels
                levels = [col for col in self.X_cols if feat in col]
                if levels:
                    level_names = [self.rename_dict.get(lvl, lvl) for lvl in levels]
                    max_util = betas.sel(features=level_names).max(dim='features')
                    min_util = betas.sel(features=level_names).min(dim='features')
                    importance[feat] = max_util - min_util
                    logger.debug(f"{feat} importance: max_util = {max_util.mean().item()}, "
                                f"min_util = {min_util.mean().item()}, "
                                f"importance mean = {importance[feat].mean().item()}")

        # Normalize importance scores
        total_importance = sum(imp.mean() for imp in importance.values())
        self.feature_importance = {
            self.rename_dict.get(feat, feat): (imp / total_importance).mean().item()
            for feat, imp in importance.items()
        }

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': list(self.feature_importance.keys()),
            'Importance': list(self.feature_importance.values())
        }).sort_values('Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title("Feature Importance")
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()

        logger.info(f"Feature importance calculated: {self.feature_importance}")
        return self.feature_importance
    
    def _calculate_feature_importance_x(self):
        """Calculate and plot feature importance based on utility ranges."""
        betas = az.extract(self.trace, var_names="betas")
        importance = {}

        for feat in self.X_cols:
            if feat == 'Price':
                price_range = (self.profiles['Price'].max() - self.profiles['Price'].min()) / 1000
                importance[feat] = np.abs(betas.sel(features=feat)) * price_range
            else:
                levels = [col for col in self.X_cols if feat in col]
                if levels:
                    level_names = [self.rename_dict.get(lvl, lvl) for lvl in levels]
                    max_util = betas.sel(features=level_names).max(dim='features')
                    min_util = betas.sel(features=level_names).min(dim='features')
                    importance[feat] = max_util - min_util

        total_importance = sum(imp.mean() for imp in importance.values())
        self.feature_importance = {
            self.rename_dict.get(feat, feat): (imp / total_importance).mean().item()
            for feat, imp in importance.items()
        }

        importance_df = pd.DataFrame({
            'Feature': list(self.feature_importance.keys()),
            'Importance': list(self.feature_importance.values())
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title("Feature Importance")
        plt.savefig('feature_importance.png', dpi=300)
        plt.close()

        return self.feature_importance

    def _analyze_willingness_to_pay(self, price_col='Price', hdi_prob=0.95):
        """Analyze WTP with uncertainty intervals."""
        wtp_results = {}
        price_col = next((col for col in self.X_cols if 'Price' in col), price_col)

        for feat in self.X_cols:
            if feat == price_col:
                continue
            wtp_var = f"wtp_{feat}"
            if wtp_var not in self.trace.posterior:
                logger.warning(f"WTP variable {wtp_var} not found in trace")
                continue
            try:
                wtp = az.extract(self.trace, var_names=wtp_var)
                wtp_values = wtp.values * 1000
                wtp_results[feat] = {
                    'mean': wtp_values.mean(),
                    'hdi_95': az.hdi(wtp_values, hdi_prob=hdi_prob),
                    'median': np.median(wtp_values),
                    'std': wtp_values.std()
                }
            except Exception as e:
                logger.error(f"Error processing WTP for {feat}: {str(e)}")
                continue

        if not wtp_results:
            logger.warning("No valid WTP results to display")
            return None

        display_names = {feat: textwrap.fill(self.rename_dict.get(feat, feat).replace('_', ' '), 50) 
                         for feat in wtp_results.keys()}

        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(wtp_results))

        for i, (feat, vals) in enumerate(wtp_results.items()):
            lower_error = vals['mean'] - vals['hdi_95'][0]
            upper_error = vals['hdi_95'][1] - vals['mean']
            plt.errorbar(
                x=vals['mean'], y=y_pos[i], xerr=[[lower_error], [upper_error]],
                fmt='o', markersize=8, capsize=5, label=display_names[feat]
            )
            plt.scatter(x=vals['median'], y=y_pos[i], marker='x', color='red', s=100)

        plt.yticks(y_pos, [display_names[feat] for feat in wtp_results.keys()])
        plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
        price_ref = self.profiles['Price'].median()
        plt.axvline(price_ref, color='green', linestyle=':', alpha=0.5, 
                    label=f'Median Price ({price_ref:,.0f})')
        plt.axvline(-price_ref, color='green', linestyle=':', alpha=0.5)
        plt.title("Willingness-to-Pay Analysis\n(Mean ± 95% HDI, X=Median)")
        plt.xlabel("WTP (Currency Units)")
        plt.grid(True, axis='x', alpha=0.3)
        plt.legend()
        plt.savefig('wtp_analysis_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

        wtp_df = pd.DataFrame.from_dict(wtp_results, orient='index')
        wtp_df.index.name = 'Feature'
        wtp_df.to_excel('wtp_results.xlsx')
        return wtp_df

    def _analyze_price_elasticity(self):
        """Analyze price elasticity across price levels."""
        if 'price_elasticity' not in self.trace.posterior:
            raise ValueError("Price elasticity not calculated in model")

        elasticity = az.extract(self.trace, var_names="price_elasticity")
        plt.figure(figsize=(10, 6))
        az.plot_posterior(elasticity, hdi_prob=0.95)
        plt.title("Price Elasticity Distribution")
        plt.savefig('price_elasticity_distribution.png', dpi=300)
        plt.close()

        price_levels = pd.qcut(self.choice_data['Price'], q=4)
        elasticity_by_price = pd.DataFrame({
            'Price Level': price_levels,
            'Elasticity': elasticity.mean(dim='sample')
        }).groupby('Price Level')['Elasticity'].agg(['mean', 'std'])

        plt.figure(figsize=(10, 6))
        elasticity_by_price['mean'].plot(kind='bar', yerr=elasticity_by_price['std'])
        plt.title("Price Elasticity by Price Level")
        plt.ylabel("Mean Elasticity")
        plt.xticks(rotation=45)
        plt.savefig('elasticity_by_price_level.png', dpi=300)
        plt.close()

        return elasticity_by_price

    def _analyze_market_segments(self):
        """Compare preferences across market segments."""
        if 'betas' not in self.trace.posterior:
            raise ValueError("Coefficients not found in model")

        betas = az.extract(self.trace, var_names="betas")
        comparisons = {}
        for feat in self.X_cols:
            comparisons[self.rename_dict.get(feat, feat)] = {
                'mean': betas.sel(features=feat).mean().item(),
                'hdi_95': az.hdi(betas.sel(features=feat).values, hdi_prob=0.95)
            }

        for feat, vals in comparisons.items():
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                vals['mean'], 0,
                xerr=[[vals['mean'] - vals['hdi_95'][0]], [vals['hdi_95'][1] - vals['mean']]],
                fmt='o',
                label=textwrap.fill(feat.replace('_', ' '), 20)
            )
            plt.yticks([])
            plt.axvline(0, color='gray', linestyle='--')
            plt.title(f"Utility for {textwrap.fill(feat.replace('_', ' '), 20)}")
            plt.xlabel("Utility")
            plt.legend()
            plt.savefig(f'comparison_{feat}.png', dpi=300, bbox_inches='tight')
            plt.close()

        return comparisons

    def predict_market_shares(self, new_profiles):
        """Predict market shares for new product profiles."""
        if self.trace is None:
            raise ValueError("Run fit_model() first")

        new_profiles = new_profiles.copy()
        new_profiles['Price'] = new_profiles['Price'] / 1000
        new_profiles_encoded = pd.get_dummies(
            new_profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=int
        )

        if self.rename_dict:
            new_profiles_encoded = new_profiles_encoded.rename(columns=self.rename_dict)

        missing_cols = set(self.X_cols) - set(new_profiles_encoded.columns)
        for col in missing_cols:
            new_profiles_encoded[col] = 0
        new_profiles_encoded = new_profiles_encoded[self.X_cols]

        with self.model:
            pm.set_data({
                "X_data": new_profiles_encoded.values,
                "choice_idx": np.zeros(len(new_profiles), dtype=int)
            })
            ppc = pm.sample_posterior_predictive(self.trace, var_names=["choices"])

        shares = ppc['choices'].mean(axis=0)
        shares_hdi = az.hdi(ppc['choices'], hdi_prob=0.95)

        return pd.DataFrame({
            'profile': new_profiles.index,
            'market_share': shares,
            'share_lower': shares_hdi[:, 0],
            'share_upper': shares_hdi[:, 1]
        })

    def simulate_scenarios(self, scenarios):
        """Simulate market response to different scenarios."""
        results = []
        original_shares = self.predict_market_shares(self.profiles)

        for scenario in scenarios:
            modified_profiles = self.profiles.copy()
            for idx in scenario.get('target_profiles', self.profiles.index):
                for feature, change in scenario['changes'].items():
                    if feature == 'Price':
                        modified_profiles.loc[idx, feature] *= change
                    else:
                        modified_profiles.loc[idx, feature] = change

            modified_shares = self.predict_market_shares(modified_profiles)
            comparison = original_shares.merge(
                modified_shares, on='profile', suffixes=('_original', '_modified')
            )
            comparison['scenario'] = scenario['name']
            results.append(comparison)

        return pd.concat(results)

if __name__ == "__main__":
    # Define renaming dictionary
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

    # Load data
    profiles = pd.read_excel('profiles.xlsx', index_col=0)
    choices = pd.read_excel("CBC_Data_Final_09Jun25295.xlsx")
    groups = pd.read_excel("A2_9Jun25295.xlsx")

    # Initialize analyzer
    analyzer = EnhancedBayesianChoiceAnalyzer(profiles, choices, groups, rename_dict)
    # Prepare data
    choice_data = analyzer.prepare_data()
    
    # Fit model with ADVI
    trace = analyzer.fit_model(n_iterations=5000, method='advi')
    
    # Analyze results
    analyzer.analyze_results()
    
    # Display WTP results
    wtp_results = analyzer._analyze_willingness_to_pay()
    print("\nWillingness-to-Pay Results:")
    print(wtp_results)