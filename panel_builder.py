import pandas as pd
from dcm10 import DiscreteChoiceAnalyzer
import logging
import os

logging.basicConfig(filename='test_log.log', level=logging.DEBUG, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")
logger = logging.getLogger("kdcm_logger")

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    profiles = pd.read_excel(os.path.join(base_dir, 'profiles.xlsx'), index_col=0)
    choices = pd.read_excel(os.path.join(base_dir, 'data_final.xlsx'))
    groups = pd.read_excel(os.path.join(base_dir, 'groups_final.xlsx'))
    choices['respondent_id'] = pd.to_numeric(choices['respondent_id'], errors='coerce')
    choices['choice_set'] = pd.to_numeric(choices['choice_set'], errors='coerce')
    choices['chosen_profile'] = pd.to_numeric(choices['chosen_profile'], errors='coerce')
    logger.info("Data files loaded")

    analyzer = DiscreteChoiceAnalyzer(profiles, choices, groups)
    choice_data = analyzer.prepare_data()
    logger.info(f"Choice data rows: {len(choice_data)}")
    logger.info(f"Choice data columns: {choice_data.columns.tolist()}")

    utilities = analyzer.fit_model(group="Panel builder", use_interactions=False)
    logger.info(f"Utilities for Panel builder:\n{utilities}")
    importance = analyzer.calculate_feature_importance()
    logger.info(f"Feature importance for Panel builder:\n{importance}")

    df = pd.read_excel(os.path.join(base_dir, 'profiles.xlsx'))
    d_dict = {'Scenario1': df['Price'].to_dict()}
    shares = analyzer.evaluate_price_scenario_lp(d_dict, plot=False)
    logger.info(f"Shares for Panel builder:\n{shares.head()}")
except Exception as e:
    logger.error(f"Test failed: {str(e)}", exc_info=True)
    raise