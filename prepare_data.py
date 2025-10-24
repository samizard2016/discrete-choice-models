import pandas as pd
from dcm10 import DiscreteChoiceAnalyzer
import logging

logging.basicConfig(filename='test_log.log', level=logging.DEBUG, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")
logger = logging.getLogger("kdcm_logger")

try:
    profiles = pd.read_excel('profiles.xlsx', index_col=0)
    choices = pd.read_excel("data_final.xlsx")
    groups = pd.read_excel("groups_final.xlsx")
    choices['respondent_id'] = pd.to_numeric(choices['respondent_id'], errors='coerce')
    choices['choice_set'] = pd.to_numeric(choices['choice_set'], errors='coerce')
    choices['chosen_profile'] = pd.to_numeric(choices['chosen_profile'], errors='coerce')
    logger.info("Data files loaded")

    analyzer = DiscreteChoiceAnalyzer(profiles, choices, groups)
    choice_data = analyzer.prepare_data()
    logger.info(f"Choice data rows: {len(choice_data)}")
except Exception as e:
    logger.error(f"Test failed: {str(e)}", exc_info=True)
    raise