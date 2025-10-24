import pandas as pd
import os

def preprocess_data(choice_files, design_file, output_dir="C:\works 2025 July plus\discrete choice models\Cargill\pylogit"):
    """Preprocess market-wise choice data files and design workbook, retaining raw columns."""
    # Define market-to-sheet mapping
    market_to_sheet = {
        'Maharashtra': 'choice_design_MH_',
        'Gujarat': 'choice_design_GJ',
        'Punjab': 'choice_design_PB_',
        'Tamilnadu': 'choice_design_South_',
        'Karnataka': 'choice_design_South_'
    }
    
    # Output columns
    columns = [
        'id', 'Respondent_ID', 'ChoiceID', 'choice_set', 'chosen_profile', 'Market',
        'Brand', 'Price', 'CP', 'FC', 'AH', 'VAS', 'Credit', 'Chosen', 'State',
        'herd_size', 'cattle_type'
    ]
    output_data = []
    
    # Load design data
    try:
        design_data = pd.read_excel(design_file, sheet_name=None)
    except Exception as e:
        print(f"Error loading design file {design_file}: {e}")
        return pd.DataFrame()
    
    design_dfs = []
    for market, sheet in market_to_sheet.items():
        if sheet in design_data:
            df = design_data[sheet].copy()
            df['Market'] = market if market not in ['Tamilnadu', 'Karnataka'] else 'South'
            design_dfs.append(df)
        else:
            print(f"Warning: Sheet {sheet} not found in design file for {market}")
    if not design_dfs:
        print("Error: No valid design sheets found")
        return pd.DataFrame()
    design_data = pd.concat(design_dfs, ignore_index=True)
    
    # Process each choice data file
    for market, choice_file in choice_files.items():
        print(f"Processing {market} from {choice_file}...")
        try:
            choice_df = pd.read_excel(choice_file)
        except Exception as e:
            print(f"Error loading choice file {choice_file}: {e}")
            continue
        
        # Map Tamilnadu/Karnataka to South
        choice_df['Market'] = market if market not in ['Tamilnadu', 'Karnataka'] else 'South'
        
        # Create ChoiceID
        choice_df['ChoiceID'] = choice_df['Respondent_ID'].astype(str) + '_' + choice_df['choice_set'].astype(str)
        
        # Verify respondent counts and tasks
        expected_tasks = len(choice_df) / choice_df['Respondent_ID'].nunique()
        if expected_tasks != 12:
            print(f"Warning: Expected 12 tasks per respondent for {market}, found {expected_tasks}")
        
        for i, row in choice_df.iterrows():
            respondent_id = row['Respondent_ID']
            choice_set = row['choice_set']
            choice_id = row['ChoiceID']
            chosen_profile = row['chosen_profile']
            market = row['Market']
            state = row.get('State', market)
            herd_size = str(row.get('herd_size', 'Unknown'))
            cattle_type = str(row.get('cattle_type', 'Unknown'))
            
            # Get profiles_presented
            profiles_presented = eval(row.get('profiles_presented', '[]'))
            if not profiles_presented:
                market_design = design_data[design_data['Market'] == market]
                task_profiles = market_design[market_design['Task'] == choice_set][['Profile', 'Alternative']].sort_values('Alternative')
                profiles_presented = task_profiles['Profile'].tolist()
                if len(profiles_presented) != 3:
                    print(f"Warning: Expected 3 profiles for {market}, choice_set {choice_set}, found {len(profiles_presented)}")
            
            # Merge with design data
            market_design = design_data[design_data['Market'] == market]
            for profile in profiles_presented:
                design_row = market_design[market_design['Profile'] == profile]
                if not design_row.empty:
                    try:
                        output_row = {
                            'id': row.get('id', i+1),
                            'Respondent_ID': respondent_id,
                            'ChoiceID': choice_id,
                            'choice_set': choice_set,
                            'chosen_profile': chosen_profile,
                            'Market': market,
                            'Brand': str(design_row['Brand'].iloc[0]),
                            'Price': float(design_row['Price'].iloc[0]),
                            'CP': float(str(design_row['CP'].iloc[0]).replace(',', '.')) if pd.notnull(design_row['CP'].iloc[0]) else 0.0,
                            'FC': str(design_row['FC'].iloc[0]),
                            'AH': str(design_row['AH'].iloc[0]),
                            'VAS': str(design_row['VAS'].iloc[0]),
                            'Credit': str(design_row['Credit'].iloc[0]),
                            'Chosen': 1 if profile == chosen_profile else 0,
                            'State': str(state),
                            'herd_size': herd_size,
                            'cattle_type': cattle_type
                        }
                        output_data.append(output_row)
                    except Exception as e:
                        print(f"Error processing profile {profile} for {market}, choice_set {choice_set}: {e}")
                else:
                    print(f"Warning: Profile {profile} not found in design data for market {market}")
            
            # Handle None option
            if chosen_profile == 99:
                output_row = {
                    'id': row.get('id', i+1),
                    'Respondent_ID': respondent_id,
                    'ChoiceID': choice_id,
                    'choice_set': choice_set,
                    'chosen_profile': chosen_profile,
                    'Market': market,
                    'Brand': 'None',
                    'Price': 0.0,
                    'CP': 0.0,
                    'FC': 'None',
                    'AH': 'None',
                    'VAS': 'None',
                    'Credit': 'None',
                    'Chosen': 1,
                    'State': str(state),
                    'herd_size': herd_size,
                    'cattle_type': cattle_type
                }
                output_data.append(output_row)
            
            if i % 1000 == 0:
                print(f"Processed {i} choice sets for {market}, {len(output_data)} rows generated")
    
    # Create combined DataFrame
    df = pd.DataFrame(output_data, columns=columns)
    
    # Standardize Price and CP
    if not df['Price'].empty:
        df['Price'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()
    if not df['CP'].empty:
        df['CP'] = (df['CP'] - df['CP'].mean()) / df['CP'].std()
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_choice_data_v2.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved combined data to {output_path}")
    print(f"Total rows generated: {len(df)}")
    
    # Log data types
    print("Data types in output DataFrame:")
    print(df.dtypes)
    
    return df

# Example usage
if __name__ == "__main__":
    choice_files = {
        'Maharashtra': 'C:/works 2025 July plus/discrete choice models/Cargill/data/AnimalFeeder_CBC_Maharashtra.xlsx',
        'Gujarat': 'C:/works 2025 July plus/discrete choice models/Cargill/data/AnimalFeeder_CBC_Gujarat.xlsx',
        'Punjab': 'C:/works 2025 July plus/discrete choice models/Cargill/data/AnimalFeeder_CBC_Punjab.xlsx',
        'Tamilnadu': 'C:/works 2025 July plus/discrete choice models/Cargill/data/AnimalFeeder_CBC_Tamilnadu.xlsx',
        'Karnataka': 'C:/works 2025 July plus/discrete choice models/Cargill/data/AnimalFeeder_CBC_Karnataka.xlsx'
    }
    design_file = 'C:/works 2025 July plus/discrete choice models/Cargill/data/choice_design_Cargill.xlsx'
    df = preprocess_data(choice_files, design_file)