import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm

# Placeholder: Load actual data (replace with actual file path)
# df = pd.read_csv('cattle_feed_data_final.csv')
# Simulated data
np.random.seed(42)
n_obs = 1000
n_alts = 3
markets = ['Maharashtra', 'Gujarat', 'Punjab', 'South']
brand_shares = {
    'Maharashtra': {'Cargill': 0.0877, 'Baramati': 0.2289, 'Other': 0.6834},
    'Gujarat': {'Cargill': 0.5, 'Amul': 0.2025, 'Other': 0.2975},
    'Punjab': {'Cargill': 0.5, 'Tiwana': 0.2946, 'Other': 0.2054},
    'South': {'Cargill': 0.5, 'KMF Nandini': 0.1545, 'Other': 0.3455}
}
initial_params = {
    'Maharashtra': [-0.030027, 0.1, -0.862332, 0.1, 0.1],  # From cargill_findings_v16.xlsx
    'Gujarat': [-0.056879, 0.1, -0.256111, 0.1, 0.1],
    'Punjab': [0.054103, 0.1, 2.200154, 0.1, 0.1],
    'South': [0.013685, 0.1, 0.082258, 0.1, 0.1]
}
data = []
for market in markets:
    brands = list(brand_shares[market].keys())
    probs = list(brand_shares[market].values())
    for obs in range(n_obs):
        chosen_brand = np.random.choice(brands, p=probs)
        for alt in range(n_alts):
            brand = brands[alt % len(brands)]
            row = {
                'obs_id': f"{market}_{obs}",
                'alt_id': alt,
                'Chosen': 1 if brand == chosen_brand else 0,
                'Price': np.random.normal(0, 1.5) + (0.3 if brand == 'Cargill' else -0.15),
                'CP': np.random.normal(0, 1) + (0.2 if brand == 'Cargill' else -0.1),
                'Brand_Cargill': 1 if brand == 'Cargill' else 0,
                'FC_1': np.random.choice([0, 1], p=[0.1 if brand == 'Cargill' else 0.9, 0.9 if brand == 'Cargill' else 0.1]),
                'AH_1': np.random.choice([0, 1], p=[0.1 if brand == 'Cargill' else 0.9, 0.9 if brand == 'Cargill' else 0.1]),
                'Market': market
            }
            data.append(row)
df = pd.DataFrame(data)

# Data cleaning
duplicates = df.groupby(['obs_id', 'Brand_Cargill'])['Chosen'].count().reset_index()
duplicates = duplicates[duplicates['Chosen'] > 1]
if not duplicates.empty:
    df = df.drop_duplicates(subset=['obs_id', 'Brand_Cargill'], keep='first')
    print(f"Removed {len(duplicates)} duplicate choice sets")

df = df.dropna(subset=['obs_id', 'Brand_Cargill', 'Price', 'CP', 'FC_1', 'AH_1', 'Chosen'])
print(f"Removed {len(df) - len(df.dropna())} rows with NaN")

choice_sums = df.groupby('obs_id')['Chosen'].sum()
invalid_choices = choice_sums[choice_sums != 1].index
if not invalid_choices.empty:
    df = df[~df['obs_id'].isin(invalid_choices)]
    print(f"Dropped {len(invalid_choices)} choice sets with invalid Chosen sums")

if df['Price'].nunique() < 10:
    df['Price'] += np.random.normal(0, 0.02, len(df))
    print("Added synthetic variation to Price")
if df['CP'].nunique() < 10:
    df['CP'] += np.random.normal(0, 0.02, len(df))
    print("Added synthetic variation to CP")
if df['FC_1'].nunique() < 2:
    df.loc[df['Brand_Cargill'] == 1, 'FC_1'] = np.random.choice([0, 1], size=sum(df['Brand_Cargill'] == 1), p=[0.1, 0.9])
    df.loc[df['Brand_Cargill'] == 0, 'FC_1'] = np.random.choice([0, 1], size=sum(df['Brand_Cargill'] == 0), p=[0.9, 0.1])
    print("Added synthetic variation to FC_1")
if df['AH_1'].nunique() < 2:
    df.loc[df['Brand_Cargill'] == 1, 'AH_1'] = np.random.choice([0, 1], size=sum(df['Brand_Cargill'] == 1), p=[0.1, 0.9])
    df.loc[df['Brand_Cargill'] == 0, 'AH_1'] = np.random.choice([0, 1], size=sum(df['Brand_Cargill'] == 0), p=[0.9, 0.1])
    print("Added synthetic variation to AH_1")

# Check multicollinearity
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [np.inf if sm.OLS(X[col], sm.add_constant(X.drop(columns=col))).fit().rsquared >= 0.999 else 1 / (1 - sm.OLS(X[col], sm.add_constant(X.drop(columns=col))).fit().rsquared) for col in X.columns]
    return vif_data

X = df[['Price', 'CP', 'Brand_Cargill', 'FC_1', 'AH_1']]
vif = calculate_vif(X)
print("VIF:\n", vif)
if (vif['VIF'] > 100).any():
    high_vif = vif[vif['VIF'] > 100]['Feature'].tolist()
    X = X.drop(columns=high_vif)
    df = df.drop(columns=high_vif)
    print(f"Dropped high-VIF features: {high_vif}")

# Custom MNL log-likelihood
def mnl_log_likelihood(params, X, y, choice_ids):
    beta_price, beta_cp, beta_cargill, beta_fc, beta_ah = params
    utilities = X @ np.array([beta_price, beta_cp, beta_cargill, beta_fc, beta_ah])
    exp_utilities = np.exp(utilities)
    choice_sums = exp_utilities.groupby(choice_ids).sum()
    probs = exp_utilities / choice_sums.reindex(utilities.index)
    log_likelihood = np.sum(y * np.log(probs + 1e-10))
    return -log_likelihood

# Fit custom MNL
results = {}
for market in markets:
    market_data = df[df['Market'] == market]
    if market_data.empty:
        print(f"No data for {market}")
        continue
    X = market_data[['Price', 'CP', 'Brand_Cargill', 'FC_1', 'AH_1']]
    y = market_data['Chosen']
    choice_ids = market_data['obs_id']
    
    initial_params_market = initial_params[market]
    def objective(params, X, y, choice_ids):
        return mnl_log_likelihood(params, X, y, choice_ids)
    
    result = minimize(
        objective,
        initial_params_market,
        args=(X, y, choice_ids),
        method='SLSQP',
        options={'disp': True, 'maxiter': 1000}
    )
    results[market] = {
        'coefficients': result.x,
        'success': result.success,
        'message': result.message
    }
    print(f"\n{market} Coefficients: Price={result.x[0]:.6f}, CP={result.x[1]:.6f}, Brand_Cargill={result.x[2]:.6f}, FC_1={result.x[3]:.6f}, AH_1={result.x[4]:.6f}")

# Predict shares with FC and AH scenarios
def predict_shares(market, beta_price, beta_cargill, beta_fc, beta_ah, price_changes=np.arange(-10, 11, 1)):
    price_mean, price_std = 0, 1.5
    price_cargill_original = 1.0
    price_other_original = 1.0
    results = []
    competitor = list(brand_shares[market].keys())[1]
    for change in price_changes:
        price_cargill_new = price_cargill_original * (1 + change / 100)
        price_cargill_scaled = (price_cargill_new - price_mean) / price_std
        price_other_scaled = (price_other_original - price_mean) / price_std
        # Base case: FC_1=0.9, AH_1=0.9 for Cargill; 0.1 for others
        u_cargill = beta_price * price_cargill_scaled + beta_cargill * 1 + beta_fc * 0.9 + beta_ah * 0.9
        u_competitor = beta_price * price_other_scaled + beta_fc * 0.1 + beta_ah * 0.1
        u_other = beta_price * price_other_scaled + beta_fc * 0.1 + beta_ah * 0.1
        exp_utilities = np.array([np.exp(u_cargill), np.exp(u_competitor), np.exp(u_other)])
        p_cargill = exp_utilities[0] / np.sum(exp_utilities)
        p_other = 1 - p_cargill
        total_revenue = price_cargill_new * p_cargill + price_other_original * p_other
        value_share = (price_cargill_new * p_cargill) / total_revenue if total_revenue > 0 else 0
        print(f"{market} Price Change {change}%: u_cargill={u_cargill:.6f}, u_competitor={u_competitor:.6f}, u_other={u_other:.6f}, p_cargill={p_cargill:.6f}")
        results.append({
            'Price_Change_Percent': change,
            'Brand_Share': p_cargill,
            'Value_Share': value_share
        })
    # Scenario: No FC/AH advantage (FC_1=AH_1=0.5 for all)
    results_no_fc_ah = []
    for change in price_changes:
        price_cargill_new = price_cargill_original * (1 + change / 100)
        price_cargill_scaled = (price_cargill_new - price_mean) / price_std
        price_other_scaled = (price_other_original - price_mean) / price_std
        u_cargill = beta_price * price_cargill_scaled + beta_cargill * 1 + beta_fc * 0.5 + beta_ah * 0.5
        u_competitor = beta_price * price_other_scaled + beta_fc * 0.5 + beta_ah * 0.5
        u_other = beta_price * price_other_scaled + beta_fc * 0.5 + beta_ah * 0.5
        exp_utilities = np.array([np.exp(u_cargill), np.exp(u_competitor), np.exp(u_other)])
        p_cargill = exp_utilities[0] / np.sum(exp_utilities)
        p_other = 1 - p_cargill
        total_revenue = price_cargill_new * p_cargill + price_other_original * p_other
        value_share = (price_cargill_new * p_cargill) / total_revenue if total_revenue > 0 else 0
        results_no_fc_ah.append({
            'Price_Change_Percent': change,
            'Brand_Share_No_FC_AH': p_cargill,
            'Value_Share_No_FC_AH': value_share
        })
    df_shares = pd.DataFrame(results).merge(pd.DataFrame(results_no_fc_ah), on='Price_Change_Percent')
    return df_shares

# Generate predictions
with pd.ExcelWriter('cargill_predicted_shares_with_fc_ah_v5.xlsx') as writer:
    for market in markets:
        if market not in results:
            continue
        coefs = results[market]['coefficients']
        df_shares = predict_shares(market, coefs[0], coefs[2], coefs[3], coefs[4])
        print(f"\nPredicted Shares for {market}:\n", df_shares.round(6))
        df_shares.to_excel(writer, sheet_name=f"{market}_predicted_shares", index=False)

print("\nResults saved to 'cargill_predicted_shares_with_fc_ah_v5.xlsx'")

def predict_new_scenarios():
    # Scaling parameters - based on approximate real data
    price_mean = 1407.47
    price_std = 203.71
    cp_mean = 20
    cp_std = 2  # Assuming standard deviation of 2% for protein content

    # Market prices
    market_prices = {
        'Maharashtra': {
            'Cargill': 1500,
            'Competitor': 1600,  # Baramati
            'Other': (1318 + 1260 + 1600 + 1700) / 4
        },
        'Gujarat': {
            'Cargill': 1550,
            'Competitor': 1050,  # Amul
            'Other': (1050 + 1223 + 1450 + 1544 + 1650) / 5
        },
        'Punjab': {
            'Cargill': 1500,
            'Competitor': 1217,  # Tiwana
            'Other': (1517 + 1525 + 1650) / 3
        },
        'South': {
            'Cargill': (1550 + 1610) / 2,
            'Competitor': (1000 + 1100) / 2,  # KMF Nandini
            'Other': (1350 + 1250 + 1280 + 1650) / 4
        }
    }

    with pd.ExcelWriter('cargill_new_predictions.xlsx') as writer:
        # 1. Predicting Brand Share in each state with current prices
        current_shares = []
        for market in markets:
            if market not in results:
                continue
            coefs = results[market]['coefficients']
            beta_price, beta_cp, beta_cargill, beta_fc, beta_ah = coefs
            prices = market_prices[market]
            p_c = prices['Cargill']
            p_comp = prices['Competitor']
            p_o = prices['Other']
            scaled_p_c = (p_c - price_mean) / price_std
            scaled_p_comp = (p_comp - price_mean) / price_std
            scaled_p_o = (p_o - price_mean) / price_std
            scaled_cp = (20 - cp_mean) / cp_std  # base
            scaled_cp_c = scaled_cp
            scaled_cp_comp = scaled_cp
            scaled_cp_o = scaled_cp
            fc_c = 0.9
            fc_comp = 0.1
            fc_o = 0.1
            ah_c = 0.9
            ah_comp = 0.1
            ah_o = 0.1
            u_c = beta_price * scaled_p_c + beta_cp * scaled_cp_c + beta_cargill * 1 + beta_fc * fc_c + beta_ah * ah_c
            u_comp = beta_price * scaled_p_comp + beta_cp * scaled_cp_comp + beta_fc * fc_comp + beta_ah * ah_comp
            u_o = beta_price * scaled_p_o + beta_cp * scaled_cp_o + beta_fc * fc_o + beta_ah * ah_o
            exp_u = np.exp([u_c, u_comp, u_o])
            shares = exp_u / exp_u.sum()
            current_shares.append({
                'Market': market,
                'Cargill Share': shares[0],
                'Competitor Share': shares[1],
                'Other Share': shares[2]
            })
        df_current = pd.DataFrame(current_shares)
        df_current.to_excel(writer, sheet_name='Current_Shares', index=False)

        # 2. Varying Cargill prices, competition fixed
        cargill_prices = [1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750]
        for market in markets:
            if market not in results:
                continue
            beta_price, beta_cp, beta_cargill, beta_fc, beta_ah = results[market]['coefficients']
            p_comp = market_prices[market]['Competitor']
            p_o = market_prices[market]['Other']
            scaled_p_comp = (p_comp - price_mean) / price_std
            scaled_p_o = (p_o - price_mean) / price_std
            scaled_cp = (20 - cp_mean) / cp_std
            fc_c = 0.9
            fc_comp = 0.1
            fc_o = 0.1
            ah_c = 0.9
            ah_comp = 0.1
            ah_o = 0.1
            u_comp = beta_price * scaled_p_comp + beta_cp * scaled_cp + beta_fc * fc_comp + beta_ah * ah_comp
            u_o = beta_price * scaled_p_o + beta_cp * scaled_cp + beta_fc * fc_o + beta_ah * ah_o
            results_var = []
            for p_c in cargill_prices:
                scaled_p_c = (p_c - price_mean) / price_std
                u_c = beta_price * scaled_p_c + beta_cp * scaled_cp + beta_cargill + beta_fc * fc_c + beta_ah * ah_c
                exp_u = np.exp([u_c, u_comp, u_o])
                shares = exp_u / exp_u.sum()
                results_var.append({
                    'Cargill Price': p_c,
                    'Cargill Share': shares[0],
                    'Competitor Share': shares[1],
                    'Other Share': shares[2]
                })
            df_var = pd.DataFrame(results_var)
            df_var.to_excel(writer, sheet_name=f'Price_Var_{market}', index=False)

        # 3. Protein content changes for Cargill
        cp_levels = [20, 22, 24]
        cargill_prices_cp = [1400, 1500, 1600]
        for market in markets:
            if market not in results:
                continue
            beta_price, beta_cp, beta_cargill, beta_fc, beta_ah = results[market]['coefficients']
            p_comp = market_prices[market]['Competitor']
            p_o = market_prices[market]['Other']
            scaled_p_comp = (p_comp - price_mean) / price_std
            scaled_p_o = (p_o - price_mean) / price_std
            scaled_cp_comp = (20 - cp_mean) / cp_std
            scaled_cp_o = scaled_cp_comp
            fc_c = 0.9
            fc_comp = 0.1
            fc_o = 0.1
            ah_c = 0.9
            ah_comp = 0.1
            ah_o = 0.1
            u_comp = beta_price * scaled_p_comp + beta_cp * scaled_cp_comp + beta_fc * fc_comp + beta_ah * ah_comp
            u_o = beta_price * scaled_p_o + beta_cp * scaled_cp_o + beta_fc * fc_o + beta_ah * ah_o
            results_cp = []
            for cp in cp_levels:
                scaled_cp_c = (cp - cp_mean) / cp_std
                for p_c in cargill_prices_cp:
                    scaled_p_c = (p_c - price_mean) / price_std
                    u_c = beta_price * scaled_p_c + beta_cp * scaled_cp_c + beta_cargill + beta_fc * fc_c + beta_ah * ah_c
                    exp_u = np.exp([u_c, u_comp, u_o])
                    shares = exp_u / exp_u.sum()
                    results_cp.append({
                        'CP Level': cp,
                        'Cargill Price': p_c,
                        'Cargill Share': shares[0],
                        'Competitor Share': shares[1],
                        'Other Share': shares[2]
                    })
            df_cp = pd.DataFrame(results_cp)
            df_cp.to_excel(writer, sheet_name=f'CP_Changes_{market}', index=False)

        # 4. Animal Health changes for Cargill
        ah_levels = {'below 3': 0, 'maintained at 3': 0.5, 'over 3': 1}
        for market in markets:
            if market not in results:
                continue
            beta_price, beta_cp, beta_cargill, beta_fc, beta_ah = results[market]['coefficients']
            p_comp = market_prices[market]['Competitor']
            p_o = market_prices[market]['Other']
            scaled_p_comp = (p_comp - price_mean) / price_std
            scaled_p_o = (p_o - price_mean) / price_std
            scaled_cp = (20 - cp_mean) / cp_std
            fc_c = 0.9
            fc_comp = 0.1
            fc_o = 0.1
            ah_comp = 0.1
            ah_o = 0.1
            u_comp = beta_price * scaled_p_comp + beta_cp * scaled_cp + beta_fc * fc_comp + beta_ah * ah_comp
            u_o = beta_price * scaled_p_o + beta_cp * scaled_cp + beta_fc * fc_o + beta_ah * ah_o
            results_ah = []
            for level, ah_val in ah_levels.items():
                ah_c = ah_val
                for p_c in cargill_prices_cp:
                    scaled_p_c = (p_c - price_mean) / price_std
                    u_c = beta_price * scaled_p_c + beta_cp * scaled_cp + beta_cargill + beta_fc * fc_c + beta_ah * ah_c
                    exp_u = np.exp([u_c, u_comp, u_o])
                    shares = exp_u / exp_u.sum()
                    results_ah.append({
                        'AH Level': level,
                        'Cargill Price': p_c,
                        'Cargill Share': shares[0],
                        'Competitor Share': shares[1],
                        'Other Share': shares[2]
                    })
            df_ah = pd.DataFrame(results_ah)
            df_ah.to_excel(writer, sheet_name=f'AH_Changes_{market}', index=False)

    print("New predictions saved to 'cargill_new_predictions.xlsx'")

# Call the new function
predict_new_scenarios()