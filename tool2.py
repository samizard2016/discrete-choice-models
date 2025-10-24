import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def filter_respondents_by_row_count(df, min_rows=4):
    """
    Remove records for respondent_id with fewer than min_rows rows in the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with columns: respondent_id, choice_set, chosen_profile, profiles_presented
    min_rows : int, optional
        Minimum number of rows required per respondent_id (default: 4)
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with only respondent_id having at least min_rows rows
    """
    # Validate input DataFrame
    required_columns = ['respondent_id', 'choice_set', 'chosen_profile', 'profiles_presented']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"DataFrame must contain all required columns: {required_columns}")

    logger.info(f"Initial DataFrame shape: {df.shape}")
    
    # Count rows per respondent_id
    respondent_counts = df['respondent_id'].value_counts()
    logger.info(f"Number of unique respondent_ids: {len(respondent_counts)}")
    
    # Identify respondent_ids with at least min_rows
    valid_respondents = respondent_counts[respondent_counts >= min_rows].index
    logger.info(f"Number of respondent_ids with at least {min_rows} rows: {len(valid_respondents)}")
    
    # Filter DataFrame to keep only valid respondent_ids
    filtered_df = df[df['respondent_id'].isin(valid_respondents)].copy()
    logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")
    
    # Log removed respondents
    removed_respondents = respondent_counts[respondent_counts < min_rows]
    if not removed_respondents.empty:
        logger.info(f"Removed {len(removed_respondents)} respondent_ids with fewer than {min_rows} rows: {removed_respondents.to_dict()}")
    
    return filtered_df

if __name__ == "__main__":
    # Example usage
    try:
        # Sample DataFrame (replace with your data loading, e.g., pd.read_excel('data_final.xlsx'))
        # data = {
        #     'respondent_id': [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        #     'choice_set': [1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 1, 2, 3],
        #     'chosen_profile': [0, 1, 2, 0, 1, 0, 2, 1, 0, 2, 1, 0, 2],
        #     'profiles_presented': ['[0,1,2]', '[0,1,2]', '[0,1,2]', '[0,1,2]', '[0,1,2]', 
        #                            '[0,1,2]', '[0,1,2]', '[0,1,2]', '[0,1,2]', '[0,1,2]', 
        #                            '[0,1,2]', '[0,1,2]', '[0,1,2]']
        # }
        df = pd.read_excel("data_final bdcm.xlsx")
        
        # logger.info("Sample DataFrame before filtering:")
        # logger.info(f"\n{df}")
        
        # Filter DataFrame
        filtered_df = filter_respondents_by_row_count(df, min_rows=4)
        
        logger.info("Filtered DataFrame:")
        logger.info(f"\n{filtered_df}")
        
        # Optionally save the filtered DataFrame
        filtered_df.to_excel('final_data filtered.xlsx', index=False)
        logger.info("Filtered DataFrame saved to 'final_data filtered.xlsx'")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")