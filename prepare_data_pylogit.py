import pandas as pd
import os

def preprocess_data(choice_files, design_file, output_dir="cattle_feed_output"):
    """Preprocess market-wise choice data files and design workbook, adding ChoiceID and interactions."""
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
    design_data = pd.read_excel(design_file, sheet_name=None)
    design_dfs = []
    for market, sheet in market_to_sheet.items():
        if sheet in design_data:
            df = design_data[sheet].copy()
            df['Market'] = market if market not in ['Tamilnadu', 'Karnataka'] else 'South'
            design_dfs.append(df)
    design_data = pd.concat(design_dfs, ignore_index=True)
    
    # Process each choice data file
    for market, choice_file in choice_files.items():
        print(f"Processing {market} from {choice_file}...")
        choice_df = pd.read_excel(choice_file)
        
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
            herd_size = row.get('herd_size', 'Unknown')
            cattle_type = row.get('cattle_type', 'Unknown')
            
            # Get profiles_presented
            profiles_presented = eval(row.get('profiles_presented', '[]'))
            if not profiles_presented:
                # Infer from design data if missing
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
                    output_row = {
                        'id': row.get('id', i+1),
                        'Respondent_ID': respondent_id,
                        'ChoiceID': choice_id,
                        'choice_set': choice_set,
                        'chosen_profile': chosen_profile,
                        'Market': market,
                        'Brand': design_row['Brand'].iloc[0],
                        'Price': design_row['Price'].iloc[0],
                        'CP': design_row['CP'].iloc[0],
                        'FC': design_row['FC'].iloc[0],
                        'AH': design_row['AH'].iloc[0],
                        'VAS': design_row['VAS'].iloc[0],
                        'Credit': design_row['Credit'].iloc[0],
                        'Chosen': 1 if profile == chosen_profile else 0,
                        'State': state,
                        'herd_size': herd_size,
                        'cattle_type': cattle_type
                    }
                    output_data.append(output_row)
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
                    'Price': 0,
                    'CP': 0,
                    'FC': 'None',
                    'AH': 'None',
                    'VAS': 'None',
                    'Credit': 'None',
                    'Chosen': 1,
                    'State': state,
                    'herd_size': herd_size,
                    'cattle_type': cattle_type
                }
                output_data.append(output_row)
            
            if i % 1000 == 0:
                print(f"Processed {i} choice sets for {market}, {len(output_data)} rows generated")
    
    # Create combined DataFrame
    df = pd.DataFrame(output_data, columns=columns)
    
    # Standardize Price
    df['Price'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()
    
    # Add brand-feature interactions
    categorical_features = ['CP', 'FC', 'AH', 'VAS', 'Credit']
    for market in df['Market'].unique():
        market_df = df[df['Market'] == market]
        brands = market_df['Brand'].unique()
        for brand in brands:
            if brand != 'None':
                for col in categorical_features:
                    df[f'{col}_{brand}'] = (df['Brand'] == brand) * df[col].astype(str)
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_choice_data_pylog.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved combined data to {output_path}")
    print(f"Total rows generated: {len(df)}")
    
    return df

# Example usage
if __name__ == "__main__":
    choice_files = {
        'Maharashtra': '.\\data\\AnimalFeeder_CBC_Maharashtra.xlsx',
        'Gujarat': '.\\data\\AnimalFeeder_CBC_Gujarat.xlsx',
        'Punjab': '.\\data\\AnimalFeeder_CBC_Punjab.xlsx',
        'Tamilnadu': '.\\data\\AnimalFeeder_CBC_Tamilnadu.xlsx',
        'Karnataka': '.\\data\\AnimalFeeder_CBC_Karnataka.xlsx'
    }
    design_file = '.\\data\\choice_design_Cargill.xlsx'
    df = preprocess_data(choice_files, design_file)