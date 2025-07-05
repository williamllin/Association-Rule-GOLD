# Association-Rule-GOLD
import pandas as pd
import yfinance as yf
from mlxtend.frequent_patterns import apriori, association_rules

# Define tickers and download historical data
tickers = {
    'PALL': 'PALL', 'SLV': 'SLV', 'PLG': 'PLG', 'ALI=F': 'ALI=F',
    'USO': 'USO', 'CL=F': 'CO', 'BZ=F': 'BCO',
    'EURUSD=X': 'EUR/USD', 'DX-Y.NYB': 'USD', '^DJI': 'DJ', '^GSPC': 'S&P 500',
    'ILTB': 'USD Bond', 'AGZ': 'AGZ', 'TIP': 'TIP', 'DJP': 'DJP',
    'IAU': 'iShares', 'SGOL': 'SGOL', 'PHYS': 'Sprott'
}
data_frames = []
for ticker, prefix in tickers.items():
    data = yf.download([ticker], period='10y')
    data.columns = [f'{prefix}_{col}' for col in data.columns]
    data_frames.append(data)
combined_df = pd.concat(data_frames, axis=1)
combined_df = combined_df.iloc[:-1]

# Extract columns for price features
adj_close_columns = [col for col in combined_df.columns if col.endswith('_Adj Close')]
low_columns = [col for col in combined_df.columns if col.endswith('_Low')]
Open_columns = [col for col in combined_df.columns if col.endswith('_Open')]
adj_close_df = combined_df[adj_close_columns]
low_df = combined_df[low_columns]
Open_df = combined_df[Open_columns]

# Construct feature matrix X
low_df_shifted = low_df.shift(1).reindex(Open_df.index)
result_values = Open_df.values - low_df_shifted.values
new_columns = [col.replace('_Open', '_increased') for col in Open_df.columns]
increased_df = pd.DataFrame(result_values, columns=new_columns, index=Open_df.index)
dropped_X_df = increased_df.dropna()
X_df = dropped_X_df.drop(columns=['iShares_increased', 'SGOL_increased', 'Sprott_increased'])

# Construct target matrix Y
adj_close_df_shifted = adj_close_df.shift(1).reindex(Open_df.index)
diff_df = adj_close_df.diff()
dropped_Y_df = diff_df.dropna()
Y_df = pd.DataFrame([
    dropped_Y_df['iShares_Adj Close'],
    dropped_Y_df['SGOL_Adj Close'],
    dropped_Y_df['Sprott_Adj Close']
]).T
Y_df.columns = ["iShares_increased", "SGOL_increased", "Sprott_increased"]

# Combine and binarize for association rule mining
list_df = pd.concat([X_df, Y_df], axis=1)
def classify_value_y(y):
    if y > 0:
        return 1
    else:
        return 0
classified_df = list_df.applymap(classify_value_y)

# Perform association rule mining
min_support = 0.4
min_confidence = 0.4
min_lift = 1.2
frequent_itemset = apriori(classified_df, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemset, metric="lift", min_threshold=min_lift)
valuable_rules = rules[rules['confidence'] >= min_confidence]
def filter_rules(rules):
    rules = rules[~rules['antecedents'].astype(str).str.contains('iShares_increased|SGOL_increased|Sprott_increased')]
    rules = rules[rules['consequents'].astype(str).str.contains('iShares_increased|SGOL_increased|Sprott_increased')]
    return rules
filtered_valuable_rules = filter_rules(valuable_rules)
print(filtered_valuable_rules.sort_values(by='lift', ascending=False).head(10))
