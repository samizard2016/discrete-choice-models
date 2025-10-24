import numpy as np
import pandas as pd

# Coefficients from cargill_findings_v16.xlsx
coefficients = {
    'Maharashtra': {'Price': -0.030027, 'CP': 0.247563, 'Brand_Cargill': -0.862332},
    'Gujarat': {'Price': -0.056879, 'CP': -0.291053, 'Brand_Cargill': -0.025527},
    'Punjab': {'Price': 0.061117, 'CP': 0.176946, 'Brand_Cargill': 2.119954},
    'South': {'Price': 0.036905, 'CP': 0.042858, 'Brand_Cargill': -0.420083}
}

# Price change percentages
price_changes = np.arange(-10, 11, 1)

# Baseline prices (arbitrary units, mean=1.0, std=0.5 for scaling)
price_mean = 1.0
price_std = 0.5
price_cargill_original = 1.0
price_other_original = 1.0

# Function to compute MNL probabilities
def compute_probabilities(price_cargill_scaled, price_other_scaled, beta_price, beta_cargill):
    u_cargill = beta_price * price_cargill_scaled + beta_cargill * 1
    u_other = beta_price * price_other_scaled + beta_cargill * 0
    exp_u_cargill = np.exp(u_cargill)
    exp_u_other = np.exp(u_other)
    p_cargill = exp_u_cargill / (exp_u_cargill + exp_u_other)
    p_other = exp_u_other / (exp_u_cargill + exp_u_other)
    return p_cargill, p_other

# Function to compute shares for a market
def compute_shares_for_market(market, beta_price, beta_cargill):
    results = []
    for change in price_changes:
        price_cargill_new = price_cargill_original * (1 + change / 100)
        price_cargill_scaled = (price_cargill_new - price_mean) / price_std
        price_other_scaled = (price_other_original - price_mean) / price_std
        p_cargill, p_other = compute_probabilities(price_cargill_scaled, price_other_scaled, beta_price, beta_cargill)
        total_revenue = price_cargill_new * p_cargill + price_other_original * p_other
        value_share = (price_cargill_new * p_cargill) / total_revenue if total_revenue > 0 else 0
        results.append({
            'Price_Change_Percent': change,
            'Brand_Share': p_cargill,
            'Value_Share': value_share
        })
    return pd.DataFrame(results)

# Compute shares for all markets
all_results = {}
for market, coefs in coefficients.items():
    df = compute_shares_for_market(market, coefs['Price'], coefs['Brand_Cargill'])
    all_results[market] = df
    print(f"\nPredicted Shares for {market}:")
    print(df.round(6))

# Save results to Excel
with pd.ExcelWriter('cargill_predicted_shares.xlsx') as writer:
    for market, df in all_results.items():
        df.to_excel(writer, sheet_name=f"{market}_predicted_shares", index=False)

print("\nResults saved to 'cargill_predicted_shares.xlsx'")