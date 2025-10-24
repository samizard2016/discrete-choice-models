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
                'FC_1': np.random.choice([0, 1], p=[0.2 if brand == 'Cargill' else 0.8, 0.8 if brand == 'Cargill' else 0.2]),
                'AH_1': np.random.choice([0, 1], p=[0.2 if brand == 'Cargill' else 0.8, 0.8 if brand == 'Cargill' else 0.2]),
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
    
    initial_params = np.array([-0.03, 0.1, -0.8, 0.05, 0.05])  # From cargill_findings_v16.xlsx
    def objective(params, X, y, choice_ids):
        return mnl_log_likelihood(params, X, y, choice_ids)  # No regularization
    
    result = minimize(
        objective,
        initial_params,
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

# Predict shares
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
        u_cargill = beta_price * price_cargill_scaled + beta_cargill * 1 + beta_fc * 0.8 + beta_ah * 0.8
        u_competitor = beta_price * price_other_scaled + beta_fc * 0.2 + beta_ah * 0.2
        u_other = beta_price * price_other_scaled + beta_fc * 0.2 + beta_ah * 0.2
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
    return pd.DataFrame(results)

# Generate predictions
with pd.ExcelWriter('cargill_predicted_shares_with_fc_ah_v4.xlsx') as writer:
    for market in markets:
        if market not in results:
            continue
        coefs = results[market]['coefficients']
        df_shares = predict_shares(market, coefs[0], coefs[2], coefs[3], coefs[4])
        print(f"\nPredicted Shares for {market}:\n", df_shares.round(6))
        df_shares.to_excel(writer, sheet_name=f"{market}_predicted_shares", index=False)

print("\nResults saved to 'cargill_predicted_shares_with_fc_ah_v4.xlsx'")