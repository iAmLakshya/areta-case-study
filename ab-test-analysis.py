import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the data
data = pd.read_csv('data_artea_ab_test_demog.csv')

# Display basic information
print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Check data types and missing values
print("\nData types:")
print(data.dtypes)
print("\nMissing values:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# 1. RANDOMIZATION CHECK
print("\n" + "="*50)
print("1. RANDOMIZATION CHECK")
print("="*50)

# Distribution of test vs control group
test_dist = data['test_coupon'].value_counts(normalize=True) * 100
print("\nDistribution of test vs control group:")
print(test_dist)

# Check balance of covariates across test and control groups
covariates = ['minority', 'non_male', 'channel_acq', 'num_past_purch', 
              'spent_last_purchase', 'weeks_since_visit', 'browsing_minutes', 'shopping_cart']

print("\nBalance check for covariates:")
print("-" * 80)
print(f"{'Variable':<20} {'Test Mean':>10} {'Control Mean':>12} {'Diff':>8} {'t-stat':>8} {'p-value':>8}")
print("-" * 80)

for var in covariates:
    test_mean = data[data['test_coupon'] == 1][var].mean()
    control_mean = data[data['test_coupon'] == 0][var].mean()
    diff = test_mean - control_mean
    
    # Perform t-test
    t_stat, p_val = stats.ttest_ind(
        data[data['test_coupon'] == 1][var],
        data[data['test_coupon'] == 0][var],
        equal_var=True
    )
    
    print(f"{var:<20} {test_mean:>10.4f} {control_mean:>12.4f} {diff:>8.4f} {t_stat:>8.4f} {p_val:>8.4f}")

# Count significant differences at 5% level
significant_diffs = sum([
    stats.ttest_ind(
        data[data['test_coupon'] == 1][var],
        data[data['test_coupon'] == 0][var]
    )[1] < 0.05 for var in covariates
])

print(f"\nNumber of significant differences at 5% level: {significant_diffs} out of {len(covariates)}")
if significant_diffs <= len(covariates) * 0.1:  # Less than 10% of variables show significant differences
    print("Randomization appears successful as we observe balance across most covariates.")
else:
    print("Randomization may have issues as several covariates show significant differences.")

# 2. COUPON EFFECT ANALYSIS
print("\n" + "="*50)
print("2. COUPON EFFECT ANALYSIS")
print("="*50)

# Analyze outcome variables
outcome_vars = ['trans_after', 'revenue_after']

print("\nOverall treatment effects:")
print("-" * 80)
print(f"{'Outcome':<15} {'Test Mean':>10} {'Control Mean':>12} {'Diff':>8} {'t-stat':>8} {'p-value':>8}")
print("-" * 80)

for var in outcome_vars:
    test_mean = data[data['test_coupon'] == 1][var].mean()
    control_mean = data[data['test_coupon'] == 0][var].mean()
    diff = test_mean - control_mean
    
    # Perform t-test
    t_stat, p_val = stats.ttest_ind(
        data[data['test_coupon'] == 1][var],
        data[data['test_coupon'] == 0][var],
        equal_var=True
    )
    
    print(f"{var:<15} {test_mean:>10.4f} {control_mean:>12.4f} {diff:>8.4f} {t_stat:>8.4f} {p_val:>8.4f}")

# 3. LINEAR REGRESSION ANALYSIS
print("\n" + "="*50)
print("3. LINEAR REGRESSION ANALYSIS")
print("="*50)

# Simple regression for transactions
print("\nSimple regression for transactions:")
model_trans = smf.ols('trans_after ~ test_coupon', data=data).fit()
print(model_trans.summary().tables[1])

# Simple regression for revenue
print("\nSimple regression for revenue:")
model_rev = smf.ols('revenue_after ~ test_coupon', data=data).fit()
print(model_rev.summary().tables[1])

# Multiple regression for transactions with controls
print("\nMultiple regression for transactions with controls:")
model_trans_ctrl = smf.ols('trans_after ~ test_coupon + minority + non_male + num_past_purch + ' +
                          'spent_last_purchase + weeks_since_visit + browsing_minutes + shopping_cart', 
                          data=data).fit()
print(model_trans_ctrl.summary().tables[1])

# Multiple regression for revenue with controls
print("\nMultiple regression for revenue with controls:")
model_rev_ctrl = smf.ols('revenue_after ~ test_coupon + minority + non_male + num_past_purch + ' +
                        'spent_last_purchase + weeks_since_visit + browsing_minutes + shopping_cart', 
                        data=data).fit()
print(model_rev_ctrl.summary().tables[1])

# 4. HETEROGENEOUS TREATMENT EFFECTS
print("\n" + "="*50)
print("4. HETEROGENEOUS TREATMENT EFFECTS")
print("="*50)

# Define segments for analysis
segments = [
    ('minority', 0, 'Non-minority'),
    ('minority', 1, 'Minority'),
    ('non_male', 0, 'Male'),
    ('non_male', 1, 'Non-male'),
    ('shopping_cart', 0, 'No cart items'),
    ('shopping_cart', 1, 'Has cart items')
]

# Analyze coupon effect by segment
for var, val, label in segments:
    segment_data = data[data[var] == val]
    
    print(f"\nSegment: {label} (n={len(segment_data)})")
    print("-" * 80)
    print(f"{'Outcome':<15} {'Test Mean':>10} {'Control Mean':>12} {'Diff':>8} {'t-stat':>8} {'p-value':>8}")
    print("-" * 80)
    
    for outcome in outcome_vars:
        test_mean = segment_data[segment_data['test_coupon'] == 1][outcome].mean()
        control_mean = segment_data[segment_data['test_coupon'] == 0][outcome].mean()
        diff = test_mean - control_mean
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(
            segment_data[segment_data['test_coupon'] == 1][outcome],
            segment_data[segment_data['test_coupon'] == 0][outcome],
            equal_var=True
        )
        
        print(f"{outcome:<15} {test_mean:>10.4f} {control_mean:>12.4f} {diff:>8.4f} {t_stat:>8.4f} {p_val:>8.4f}")

# Analysis by acquisition channel
print("\nAnalysis by acquisition channel:")
for channel in sorted(data['channel_acq'].unique()):
    channel_data = data[data['channel_acq'] == channel]
    channel_name = {1: 'Google', 2: 'Facebook', 3: 'Instagram', 4: 'Referral', 5: 'Other'}.get(channel, f'Channel {channel}')
    
    print(f"\nChannel: {channel_name} (n={len(channel_data)})")
    print("-" * 80)
    print(f"{'Outcome':<15} {'Test Mean':>10} {'Control Mean':>12} {'Diff':>8} {'t-stat':>8} {'p-value':>8}")
    print("-" * 80)
    
    for outcome in outcome_vars:
        test_mean = channel_data[channel_data['test_coupon'] == 1][outcome].mean()
        control_mean = channel_data[channel_data['test_coupon'] == 0][outcome].mean()
        diff = test_mean - control_mean
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(
            channel_data[channel_data['test_coupon'] == 1][outcome],
            channel_data[channel_data['test_coupon'] == 0][outcome],
            equal_var=True
        )
        
        print(f"{outcome:<15} {test_mean:>10.4f} {control_mean:>12.4f} {diff:>8.4f} {t_stat:>8.4f} {p_val:>8.4f}")

# Analysis by shopping frequency
# Create shopping frequency segments
data['freq_segment'] = pd.cut(
    data['num_past_purch'], 
    bins=[-1, 1, 3, float('inf')], 
    labels=['Low (0-1)', 'Medium (2-3)', 'High (4+)']
)

for segment in data['freq_segment'].unique():
    segment_data = data[data['freq_segment'] == segment]
    
    print(f"\nShopping Frequency: {segment} (n={len(segment_data)})")
    print("-" * 80)
    print(f"{'Outcome':<15} {'Test Mean':>10} {'Control Mean':>12} {'Diff':>8} {'t-stat':>8} {'p-value':>8}")
    print("-" * 80)
    
    for outcome in outcome_vars:
        test_mean = segment_data[segment_data['test_coupon'] == 1][outcome].mean()
        control_mean = segment_data[segment_data['test_coupon'] == 0][outcome].mean()
        diff = test_mean - control_mean
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(
            segment_data[segment_data['test_coupon'] == 1][outcome],
            segment_data[segment_data['test_coupon'] == 0][outcome],
            equal_var=True
        )
        
        print(f"{outcome:<15} {test_mean:>10.4f} {control_mean:>12.4f} {diff:>8.4f} {t_stat:>8.4f} {p_val:>8.4f}")

# 5. INTERACTION MODELS
print("\n" + "="*50)
print("5. INTERACTION MODELS")
print("="*50)

# Interaction model for key segments
interaction_vars = ['minority', 'non_male', 'shopping_cart']

for var in interaction_vars:
    # Transactions model with interaction
    model_formula = f'trans_after ~ test_coupon + {var} + test_coupon:{var}'
    model_int = smf.ols(model_formula, data=data).fit()
    
    print(f"\nInteraction model: trans_after ~ test_coupon * {var}")
    print(model_int.summary().tables[1])
    
    # Revenue model with interaction
    model_formula = f'revenue_after ~ test_coupon + {var} + test_coupon:{var}'
    model_int = smf.ols(model_formula, data=data).fit()
    
    print(f"\nInteraction model: revenue_after ~ test_coupon * {var}")
    print(model_int.summary().tables[1])
