import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import os
from itertools import combinations
import textwrap
import xarray as xr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBayesianChoiceAnalyzer:
    def __init__(self, profiles=None, choices=None, groups=None, rename_dict=None):
        """
        Initialize the Bayesian choice analyzer with data validation and feature renaming.

        Args:
            profiles (pd.DataFrame, optional): Product profiles with features.
            choices (pd.DataFrame, optional): Respondent choices with respondent_id, choice_set, chosen_profile, profiles_presented.
            groups (pd.DataFrame, optional): Respondent group assignments.
            rename_dict (dict, optional): Dictionary to rename feature columns for clarity.
        """
        self.profiles = profiles.copy() if profiles is not None else None
        self.choices = choices.copy() if choices is not None else None
        self.groups = groups.copy() if groups is not None else None
        self.rename_dict = rename_dict or {}
        self.model = None
        self.trace = None
        self.choice_data = None
        self.X_cols = None
        self.n_alternatives = 3
        self.n_unique_choice_sets = 24
        self.n_respondents = 295
        self.tasks_per_respondent = 4
        self.n_choice_sets = self.n_respondents * self.tasks_per_respondent  # 1180

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
        print("Unique AF values:", sorted(profiles['AF'].unique()))
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

        # Prepare observed choices (alternative indices for chosen profiles)
        choices_observed = []
        for _, group in self.choice_data.groupby(['respondent_id', 'choice_set']):
            chosen_idx = group[group['chosen'] == 1].index
            if len(chosen_idx) != 1:
                raise ValueError(f"Choice set {group['choice_set'].iloc[0]} has {len(chosen_idx)} chosen alternatives")
            # Map profile index to alternative index (0, 1, 2 within choice set)
            alt_idx = group.index.get_loc(chosen_idx[0]) % self.n_alternatives
            choices_observed.append(alt_idx)
        choices_observed = np.array(choices_observed)
        if len(choices_observed) != self.n_choice_sets:
            raise ValueError(f"Expected {self.n_choice_sets} choices, got {len(choices_observed)}")

        coords = {
            "features": self.X_cols,
            "obs": np.arange(len(self.choice_data)),
            "alternatives": np.arange(self.n_alternatives),
            "obs_id": np.arange(self.n_choice_sets)
        }

        with pm.Model(coords=coords) as self.model:
            betas = pm.Normal("betas", 0, 5, dims="features")  # Tighter prior
            X_data = pm.Data("X_data", self.choice_data[self.X_cols].values, dims=("obs", "features"))
            choices_data = pm.Data("choices", choices_observed, dims="obs_id")
            
            utilities = pm.math.dot(X_data, betas)
            utilities_reshaped = utilities.reshape((self.n_choice_sets, self.n_alternatives))
            choice_prob = pm.math.softmax(utilities_reshaped, axis=1)

            pm.Categorical("choices_observed", p=choice_prob, observed=choices_data, dims="obs_id")

            # WTP calculations with enhanced stability
            price_beta = pm.math.switch(pm.math.abs(betas[0]) > 1e-3, betas[0], 1e-3)  # Stricter threshold
            for i, feat in enumerate(self.X_cols[1:], start=1):
                wtp = -betas[i] / price_beta
                # Clamp WTP to prevent extreme values
                wtp_clamped = pm.math.switch(pm.math.abs(wtp) < 1e6, wtp, 0.0)
                pm.Deterministic(f"wtp_{feat}", wtp_clamped)
                logger.debug(f"Added WTP deterministic for feature: {feat}")

            pm.Deterministic(
                "price_elasticity",
                betas[0] * X_data[:, 0] * (1 - choice_prob.flatten()[:len(self.choice_data)]),
                dims="obs"
            )

        logger.info(f"Model built with WTP variables for features: {self.X_cols[1:]}")
        return self.model

    def fit_model(self, n_iterations=100000, method='advi', mcmc_draws=2000, mcmc_tune=1000, mcmc_chains=4):
        """
        Fit the model using ADVI or MCMC with enhanced diagnostics.

        Args:
            n_iterations (int): Number of iterations for ADVI.
            method (str): Optimization method ('advi' or 'mcmc').
            mcmc_draws (int): Number of MCMC draws per chain.
            mcmc_tune (int): Number of MCMC tuning steps.
            mcmc_chains (int): Number of MCMC chains.

        Returns:
            az.InferenceData: Model trace.
        """
        if self.model is None:
            self.build_model()

        # Log PyMC and ArviZ versions
        logger.info(f"PyMC version: {pm.__version__}, ArviZ version: {az.__version__}")

        logger.info(f"Fitting model with {method.upper()} (iterations: {n_iterations})")
        with self.model:
            if method.lower() == 'advi':
                mean_field = pm.fit(n=n_iterations, method='advi', random_seed=42, progressbar=True)
                elbo_history = mean_field.hist
                self.trace = mean_field.sample(1000)
                self.trace = az.convert_to_inference_data(self.trace)
                
                # Log ELBO and plot convergence
                logger.info(f"Final ELBO: {elbo_history[-1]:.2f}")
                plt.figure(figsize=(10, 6))
                plt.plot(elbo_history)
                plt.title("ADVI ELBO Convergence")
                plt.xlabel("Iteration")
                plt.ylabel("ELBO")
                plt.savefig('elbo_convergence.png', dpi=300)
                plt.close()
                
                logger.warning("ADVI used: r_hat diagnostics unavailable. Check ELBO and trace plots.")
            else:
                self.trace = pm.sample(
                    draws=mcmc_draws,
                    tune=mcmc_tune,
                    chains=mcmc_chains,
                    target_accept=0.9,
                    return_inferencedata=True,
                    random_seed=42
                )

            # Verify WTP variables in trace
            wtp_vars = [var for var in self.trace.posterior.data_vars if var.startswith("wtp_")]
            logger.info(f"WTP variables in trace: {wtp_vars}")
            if not wtp_vars:
                logger.warning("No WTP variables found in trace. Attempting to recompute.")
                with self.model:
                    ppc = pm.sample_posterior_predictive(self.trace, var_names=wtp_vars, extend_inferencedata=True)
                    self.trace = self.trace.merge(ppc)
                    wtp_vars = [var for var in self.trace.posterior.data_vars if var.startswith("wtp_")]

            # Generate summary without stat_funcs
            var_names = ["betas"] + wtp_vars
            try:
                summary = az.summary(self.trace, var_names=var_names, hdi_prob=0.95)
                logger.info(f"Model summary:\n{summary}")
            except Exception as e:
                logger.error(f"Summary failed with var_names={var_names}: {str(e)}")
                # Fallback to only betas
                summary = az.summary(self.trace, var_names=["betas"], hdi_prob=0.95)
                logger.info(f"Fallback model summary (betas only):\n{summary}")

            # Plot trace
            az.plot_trace(self.trace, var_names=var_names, figsize=(12, len(self.X_cols) * 2))
            plt.savefig('convergence_diagnostics.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Posterior predictive check
            ppc = pm.sample_posterior_predictive(self.trace, var_names=["choices_observed"])
            logger.info(f"PPC choices_observed shape: {ppc.posterior_predictive['choices_observed'].shape}")

            # Prepare observed choices for PPC (alternative indices)
            observed_choices = []
            for _, group in self.choice_data.groupby(['respondent_id', 'choice_set']):
                chosen_idx = group[group['chosen'] == 1].index
                if len(chosen_idx) != 1:
                    logger.warning(f"Choice set {group['choice_set'].iloc[0]} has {len(chosen_idx)} chosen alternatives")
                    continue
                alt_idx = group.index.get_loc(chosen_idx[0]) % self.n_alternatives
                observed_choices.append(alt_idx)
            observed_choices = np.array(observed_choices)
            logger.info(f"Observed choices shape: {observed_choices.shape}")

            # Plot PPC without data_pairs, using observed data directly
            try:
                az.plot_ppc(ppc, num_pp_samples=100, observed=observed_choices, var_names=["choices_observed"])
                plt.savefig('posterior_predictive_check.png', dpi=300)
                plt.close()
            except Exception as e:
                logger.error(f"PPC plotting failed: {str(e)}")
                plt.close()

            # Check price coefficient stability
            betas = az.extract(self.trace, var_names="betas")
            price_beta = betas.sel(features='Price').values
            logger.info(f"Price coefficient: mean={price_beta.mean():.4f}, std={price_beta.std():.4f}")
            if abs(price_beta.mean()) < 1e-5 or price_beta.std() / abs(price_beta.mean()) > 1:
                logger.warning("Unstable price coefficient detected. WTP estimates may be unreliable.")

        return self.trace

    def save_model(self, model_file='bayesian_choice_model.nc', metadata_file='bayesian_choice_metadata.pkl'):
        """
        Save the trained model and metadata to disk.

        Args:
            model_file (str): Path to save the posterior trace (NetCDF format).
            metadata_file (str): Path to save metadata (pickle format).
        """
        if self.trace is None:
            raise ValueError("No trained model to save. Run fit_model() first.")
        if self.model is None:
            raise ValueError("No model to save. Run build_model() first.")

        logger.info(f"Saving model to {model_file} and metadata to {metadata_file}")

        # Save the posterior trace
        self.trace.to_netcdf(model_file)

        # Save metadata (X_cols, rename_dict, n_alternatives, n_choice_sets)
        metadata = {
            'X_cols': self.X_cols,
            'rename_dict': self.rename_dict,
            'n_alternatives': self.n_alternatives,
            'n_choice_sets': self.n_choice_sets
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info("Model and metadata saved successfully")

    @classmethod
    def load_model(cls, model_file='bayesian_choice_model.nc', metadata_file='bayesian_choice_metadata.pkl', profiles=None):
        """
        Load a saved model and initialize the analyzer for simulation.

        Args:
            model_file (str): Path to the saved posterior trace (NetCDF format).
            metadata_file (str): Path to the saved metadata (pickle format).
            profiles (pd.DataFrame, optional): Product profiles for simulation. Required for market share predictions.

        Returns:
            EnhancedBayesianChoiceAnalyzer: Initialized analyzer with loaded model.
        """
        logger.info(f"Loading model from {model_file} and metadata from {metadata_file}")

        # Load the trace
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file {model_file} not found")
        trace = az.from_netcdf(model_file)

        # Load metadata
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file {metadata_file} not found")
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        # Initialize analyzer
        analyzer = cls(profiles=profiles, rename_dict=metadata['rename_dict'])
        analyzer.trace = trace
        analyzer.X_cols = metadata['X_cols']
        analyzer.n_alternatives = metadata['n_alternatives']
        analyzer.n_choice_sets = metadata['n_choice_sets']

        # Rebuild the model to ensure compatibility
        analyzer.model = analyzer._rebuild_model()
        logger.info("Model and metadata loaded successfully")

        return analyzer

    def _rebuild_model(self):
        """
        Rebuild the PyMC model for use with loaded trace.

        Returns:
            pm.Model: Reconstructed PyMC model.
        """
        logger.info("Rebuilding PyMC model for loaded trace")
        coords = {
            "features": self.X_cols,
            "obs": np.arange(self.n_choice_sets * self.n_alternatives),
            "alternatives": np.arange(self.n_alternatives),
            "obs_id": np.arange(self.n_choice_sets)
        }

        with pm.Model(coords=coords) as model:
            betas = pm.Normal("betas", 0, 10, dims="features")
            X_data = pm.Data("X_data", np.zeros((self.n_choice_sets * self.n_alternatives, len(self.X_cols))),
                            dims=("obs", "features"))
            choices_data = pm.Data("choices", np.zeros(self.n_choice_sets, dtype=int), dims="obs_id")
            
            utilities = pm.math.dot(X_data, betas)
            utilities_reshaped = utilities.reshape((self.n_choice_sets, self.n_alternatives))
            choice_prob = pm.math.softmax(utilities_reshaped, axis=1)
            pm.Categorical("choices_observed", p=choice_prob, observed=choices_data, dims="obs_id")

            # WTP calculations
            for i, feat in enumerate(self.X_cols[1:], start=1):
                pm.Deterministic(f"wtp_{feat}", -betas[i] / (betas[0] + 1e-10))

            pm.Deterministic(
                "price_elasticity",
                betas[0] * X_data[:, 0] * (1 - choice_prob.flatten()[:self.n_choice_sets * self.n_alternatives]),
                dims="obs"
            )

        return model

    
    def analyze_results(self):
        """
        Analyze model results, including utilities, feature importance, WTP, elasticity, and segments.

        Args:
            original_price_range (float): Range of original prices (max - min). Default is 200.
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
        plt.savefig('utilities_with_uncertainty_bdcm.png', dpi=300, bbox_inches='tight')
        plt.close()

        if save_betas_to_excel:
            logger.info(f"Extracting and saving beta coefficients to {beta_excel_file}")
            betas = az.extract(self.trace, var_names="betas")
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

            beta_df = pd.DataFrame(beta_summary)
            beta_df.to_excel(beta_excel_file, index=False)
            logger.info(f"Beta coefficients saved to {beta_excel_file}")

            return beta_df
    def _calculate_feature_importance(self):
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
        plt.savefig('feature_importance_bdcm.png', dpi=300)
        plt.close()

        return self.feature_importance

    def _calculate_feature_importance_x(self, original_price_range=200):
        """
        Calculate and plot feature importance based on utility ranges.

        Args:
            original_price_range (float): Range of original prices (max - min). Default is 200.
        """
        logger.info("Calculating feature importance")
        betas = az.extract(self.trace, var_names="betas")
        importance = {}

        for feat in self.X_cols:
            if feat == 'Price':
                importance[feat] = np.abs(betas.sel(features=feat)) * (original_price_range / 1000)
                logger.debug(f"Price importance: beta mean = {betas.sel(features=feat).mean().item()}, "
                             f"range = {original_price_range / 1000}, "
                             f"importance mean = {importance[feat].mean().item()}")
            else:
                levels = [col for col in self.X_cols if feat in col]
                if levels:
                    level_names = [self.rename_dict.get(lvl, lvl) for lvl in levels]
                    max_util = betas.sel(features=level_names).max(dim='features')
                    min_util = betas.sel(features=level_names).min(dim='features')
                    importance[feat] = max_util - min_util
                    logger.debug(f"{feat} importance: max_util = {max_util.mean().item()}, "
                                 f"min_util = {min_util.mean().item()}, "
                                 f"importance mean = {importance[feat].mean().item()}")

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

        logger.info(f"Feature importance calculated: {self.feature_importance}")
        return self.feature_importance
    def _analyze_willingness_to_pay(self, price_col='Price', hdi_prob=0.95):
        """Analyze WTP with uncertainty intervals and improved visualization."""
        logger.info("Analyzing willingness-to-pay with uncertainty intervals")
        wtp_results = {}
        price_col = next((col for col in self.X_cols if 'Price' in col), price_col)

        # Check price coefficient stability
        betas = az.extract(self.trace, var_names="betas")
        price_beta = betas.sel(features=price_col).values
        price_beta_mean = price_beta.mean()
        price_beta_std = price_beta.std()
        logger.info(f"Price coefficient: mean={price_beta_mean:.4f}, std={price_beta_std:.4f}")
        if abs(price_beta_mean) < 1e-5 or price_beta_std / abs(price_beta_mean) > 1:
            logger.warning("Price coefficient is very small or unstable (mean=%f, std=%f). WTP estimates may be unreliable.",
                        price_beta_mean, price_beta_std)

        # Calculate WTP for each feature
        for feat in self.X_cols:
            if feat == price_col:
                continue
            wtp_var = f"wtp_{feat}"
            if wtp_var not in self.trace.posterior:
                logger.warning(f"WTP variable {wtp_var} not found in trace")
                continue
            try:
                wtp = az.extract(self.trace, var_names=wtp_var)
                wtp_values = wtp.values * 1000  # Convert to original price units
                hdi = az.hdi(wtp_values, hdi_prob=hdi_prob)
                wtp_results[feat] = {
                    'mean': wtp_values.mean(),
                    'hdi_95': hdi,
                    'median': np.median(wtp_values),
                    'std': wtp_values.std()
                }
            except Exception as e:
                logger.error(f"Error processing WTP for {feat}: {str(e)}")
                continue

        if not wtp_results:
            logger.warning("No valid WTP results to display")
            return None

        # Prepare display names with wrapping
        display_names = {feat: textwrap.fill(self.rename_dict.get(feat, feat).replace('_', ' '), 30) 
                        for feat in wtp_results.keys()}

        # Create plot with dynamic axis limits
        plt.figure(figsize=(12, max(8, len(wtp_results) * 0.8)))
        y_pos = np.arange(len(wtp_results))
        
        # Calculate dynamic x-axis limits based on HDI ranges
        all_hdi = [vals['hdi_95'] for vals in wtp_results.values()]
        x_min = min(hdi[0] for hdi in all_hdi) * 1.1  # Add 10% padding
        x_max = max(hdi[1] for hdi in all_hdi) * 1.1
        if abs(x_max - x_min) < 1000:  # Ensure minimum range for small HDIs
            x_center = (x_max + x_min) / 2
            x_min, x_max = x_center - 500, x_center + 500

        for i, (feat, vals) in enumerate(wtp_results.items()):
            lower_error = vals['mean'] - vals['hdi_95'][0]
            upper_error = vals['hdi_95'][1] - vals['mean']
            plt.errorbar(
                x=vals['mean'], y=y_pos[i], 
                xerr=[[lower_error], [upper_error]],
                fmt='o', markersize=8, capsize=5, color='blue', 
                label=display_names[feat] if i == 0 else None  # Legend on first item only
            )
            plt.scatter(x=vals['median'], y=y_pos[i], marker='x', color='red', s=100, 
                        label='Median' if i == 0 else None)

        plt.yticks(y_pos, [display_names[feat] for feat in wtp_results.keys()])
        plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
        price_ref = self.profiles['Price'].median() * 1000
        plt.axvline(price_ref, color='green', linestyle=':', alpha=0.5, 
                    label=f'Median Price ({price_ref:,.0f})')
        plt.axvline(-price_ref, color='green', linestyle=':', alpha=0.5)
        plt.xlim(x_min, x_max)  # Set dynamic x-axis limits
        plt.title("Willingness-to-Pay Analysis\n(Mean ± 95% HDI, X=Median)")
        plt.xlabel("WTP (Currency Units)")
        plt.grid(True, axis='x', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('wtp_analysis_enhanced_bdcm.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save WTP results to DataFrame
        wtp_df = pd.DataFrame.from_dict(wtp_results, orient='index')
        wtp_df.index.name = 'Feature'
        wtp_df.to_excel('wtp_results_bdcm.xlsx')
        logger.info("WTP results saved to wtp_results_bdcm.xlsx")
        return wtp_df

    def _analyze_willingness_to_pay_x(self, price_col='Price', hdi_prob=0.95):
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
        price_ref = self.profiles['Price'].median() * 1000
        plt.axvline(price_ref, color='green', linestyle=':', alpha=0.5, 
                    label=f'Median Price ({price_ref:,.0f})')
        plt.axvline(-price_ref, color='green', linestyle=':', alpha=0.5)
        plt.title("Willingness-to-Pay Analysis\n(Mean ± 95% HDI, X=Median)")
        plt.xlabel("WTP (Currency Units)")
        plt.grid(True, axis='x', alpha=0.3)
        plt.legend()
        plt.savefig('wtp_analysis_enhanced_bdcm.png', dpi=300, bbox_inches='tight')
        plt.close()

        wtp_df = pd.DataFrame.from_dict(wtp_results, orient='index')
        wtp_df.index.name = 'Feature'
        wtp_df.to_excel('wtp_results_bdcm.xlsx')
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
        plt.savefig('elasticity_by_price_level_bdcm.png', dpi=300)
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
            plt.savefig(f'comparison_{feat}_bdcm.png', dpi=300, bbox_inches='tight')
            plt.close()

        return comparisons

    def predict_market_shares(self, new_profiles):
        """
        Predict market shares for new product profiles.

        Args:
            new_profiles (pd.DataFrame): Profiles with features (Price, SP, AF).

        Returns:
            pd.DataFrame: Market shares with 95% HDI.
        """
        if self.trace is None:
            raise ValueError("Run fit_model() or load_model() first")

        # Log xarray version
        logger.info(f"xarray version: {xr.__version__}")

        logger.info(f"Predicting market shares for {len(new_profiles)} profiles")
        new_profiles = new_profiles.copy()
        new_profiles['Price'] = new_profiles['Price'] / 1000  # Normalize price
        new_profiles_encoded = pd.get_dummies(
            new_profiles,
            columns=['SP', 'AF'],
            drop_first=True,
            dtype=int
        )

        # Apply renaming dictionary
        if self.rename_dict:
            new_profiles_encoded = new_profiles_encoded.rename(columns=self.rename_dict)

        # Ensure all required columns are present
        missing_cols = set(self.X_cols) - set(new_profiles_encoded.columns)
        for col in missing_cols:
            new_profiles_encoded[col] = 0
        new_profiles_encoded = new_profiles_encoded[self.X_cols]

        # Define new coordinates for a single choice set
        n_new_obs = len(new_profiles_encoded)
        new_coords = {
            "obs": np.arange(n_new_obs),
            "alternatives": np.arange(n_new_obs),
            "obs_id": np.array([0]),  # Single choice set
            "features": self.X_cols
        }

        # Log input shapes
        logger.info(f"New profiles encoded shape: {new_profiles_encoded.shape}")
        logger.info(f"Expected utilities reshape: (1, {n_new_obs})")

        with self.model:
            # Update data
            pm.set_data(
                {
                    "X_data": new_profiles_encoded.values,
                    "choices": np.zeros(1, dtype=int)  # Dummy observed choices
                },
                coords=new_coords
            )

            # Temporarily redefine utilities reshape for prediction
            utilities = pm.math.dot(self.model["X_data"], self.model["betas"])
            utilities_reshaped = utilities.reshape((1, n_new_obs))
            choice_prob = pm.math.softmax(utilities_reshaped, axis=1)
            pm.Categorical("choices_observed_pred", p=choice_prob, observed=np.zeros(1, dtype=int))

            # Sample posterior predictive
            try:
                ppc = pm.sample_posterior_predictive(self.trace, var_names=["choices_observed_pred"], random_seed=42)
                logger.info(f"choices_observed_pred shape: {ppc.posterior_predictive['choices_observed_pred'].shape}")
            except Exception as e:
                logger.error(f"Posterior predictive sampling failed: {str(e)}")
                raise

        # Compute market shares
        choices_pred = ppc.posterior_predictive['choices_observed_pred']
        shares = np.zeros(n_new_obs)
        for i in range(n_new_obs):
            shares[i] = np.mean(choices_pred == i)
        logger.info(f"Computed shares shape: {shares.shape}")

        # Compute HDI
        shares_hdi = az.hdi(choices_pred, hdi_prob=0.95)
        
        # Handle Dataset vs DataArray
        if isinstance(shares_hdi, xr.Dataset):
            logger.info(f"shares_hdi is Dataset with variables: {list(shares_hdi.data_vars)}")
            shares_hdi = shares_hdi['choices_observed_pred']
        logger.info(f"shares_hdi is DataArray with dims: {shares_hdi.dims}, coords: {shares_hdi.coords}, shape: {shares_hdi.shape}")

        # Extract HDI bounds dynamically
        try:
            share_lower = shares_hdi.sel(hdi='lower').values.flatten()
            share_upper = shares_hdi.sel(hdi='upper').values.flatten()
        except KeyError:
            try:
                share_lower = shares_hdi.sel(hdi='hdi_3%').values.flatten()
                share_upper = shares_hdi.sel(hdi='hdi_97%').values.flatten()
            except KeyError:
                logger.warning("HDI coordinates not found, attempting positional indexing")
                if 'hdi' in shares_hdi.dims:
                    share_lower = shares_hdi.isel(hdi=0).values.flatten()
                    share_upper = shares_hdi.isel(hdi=1).values.flatten()
                else:
                    logger.warning("No hdi dimension found, computing HDI manually")
                    shares_hdi_np = np.zeros((n_new_obs, 2))
                    for i in range(n_new_obs):
                        choices_i = (choices_pred == i).mean(axis=(0, 1)).values
                        shares_hdi_np[i] = np.percentile(choices_i, [2.5, 97.5])
                    share_lower = shares_hdi_np[:, 0]
                    share_upper = shares_hdi_np[:, 1]

        # Validate output shapes
        logger.info(f"Shares shape: {shares.shape}, Lower HDI shape: {share_lower.shape}, Upper HDI shape: {share_upper.shape}")
        if shares.shape != share_lower.shape or shares.shape != share_upper.shape:
            logger.error(f"Shape mismatch: shares {shares.shape}, lower {share_lower.shape}, upper {share_upper.shape}")
            raise ValueError("Inconsistent shapes in market share output")

        return pd.DataFrame({
            'profile': new_profiles.index,
            'market_share': shares,
            'share_lower': share_lower,
            'share_upper': share_upper
        })
    def simulate_price_scenarios(self, custom_prices=None, output_file='price_scenario_shares.xlsx'):
        """
        Simulate market shares under different price scenarios: original, +10%, -10%, and custom prices.

        Args:
            custom_prices (list, optional): List of custom price levels to simulate. If None, uses original prices.
            output_file (str): Path to save the simulation results (Excel format).

        Returns:
            pd.DataFrame: Combined results of market shares for all scenarios.
        """
        if self.trace is None or self.model is None:
            raise ValueError("No trained model available. Run fit_model() or load_model() first.")
        if self.profiles is None:
            raise ValueError("Profiles data required for simulation. Provide profiles during initialization.")

        logger.info(f"Simulating market shares for {len(self.profiles)} profiles across price scenarios")
        scenarios = [
            {'name': 'Original Price', 'price_factor': 1.0},
            {'name': '10% Price Increase', 'price_factor': 1.1},
            {'name': '10% Price Decrease', 'price_factor': 0.9}
        ]

        if custom_prices is not None:
            scenarios.append({'name': 'Custom Prices', 'custom_prices': custom_prices})

        results = []
        original_prices = self.profiles['Price'].copy() * 1000  # Denormalize to raw prices
        unique_prices = sorted(original_prices.unique())

        for scenario in scenarios:
            new_profiles = self.profiles.copy()
            if scenario['name'] == 'Custom Prices':
                if len(scenario['custom_prices']) != len(unique_prices):
                    raise ValueError(f"Custom prices {scenario['custom_prices']} must match number of unique price levels "
                                    f"({len(unique_prices)}: {unique_prices})")
                price_mapping = dict(zip(unique_prices, sorted(scenario['custom_prices'])))
                new_profiles['Price'] = new_profiles['Price'] * 1000  # Denormalize
                new_profiles['Price'] = new_profiles['Price'].map(price_mapping) / 1000  # Apply custom prices and normalize
            else:
                new_profiles['Price'] = new_profiles['Price'] * scenario['price_factor']  # Apply price factor

            # Predict market shares
            shares = self.predict_market_shares(new_profiles)
            shares['Scenario'] = scenario['name']
            results.append(shares)

        # Combine results
        result_df = pd.concat(results, ignore_index=True)
        result_df['market_share'] = result_df['market_share'] * 100  # Convert to percentage
        result_df['share_lower'] = result_df['share_lower'] * 100
        result_df['share_upper'] = result_df['share_upper'] * 100

        # Add price column
        scenario_prices = []
        for scenario in scenarios:
            temp_profiles = self.profiles.copy()
            if scenario['name'] == 'Custom Prices' and custom_prices is not None:
                price_mapping = dict(zip(unique_prices, sorted(custom_prices)))
                temp_profiles['Price'] = temp_profiles['Price'] * 1000
                temp_profiles['Price'] = temp_profiles['Price'].map(price_mapping)
            else:
                temp_profiles['Price'] = temp_profiles['Price'] * 1000 * scenario['price_factor']
            temp_profiles['Scenario'] = scenario['name']
            scenario_prices.append(temp_profiles[['Price']].reset_index())
        price_df = pd.concat(scenario_prices, ignore_index=True)

        result_df = result_df.merge(price_df, on=['profile', 'Scenario'], how='left')

        # Save to Excel
        result_df.to_excel(output_file, index=False)
        logger.info(f"Price scenario simulation results saved to {output_file}")

        return result_df

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
    trace = analyzer.fit_model(n_iterations=100000, method='advi')
    
    # Save the model
    analyzer.save_model()
    
    # Analyze results
    analyzer.analyze_results()
    
    # Display WTP results
    wtp_results = analyzer._analyze_willingness_to_pay()
    print("\nWillingness-to-Pay Results:")
    print(wtp_results)

    # Example: Run price scenario simulation with custom prices
    custom_prices = [11800, 120000, 122000]  # Example custom price levels
    scenario_results = analyzer.simulate_price_scenarios(custom_prices=custom_prices)
    print("\nPrice Scenario Simulation Results:")
    print(scenario_results)

    # Example: Load saved model and run simulation
    loaded_analyzer = EnhancedBayesianChoiceAnalyzer.load_model(profiles=profiles)
    loaded_scenario_results = loaded_analyzer.simulate_price_scenarios(custom_prices=[1350, 1500, 1650])
    print("\nPrice Scenario Simulation Results (Loaded Model):")
    print(loaded_scenario_results)