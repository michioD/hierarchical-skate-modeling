import pandas as pd
import numpy as np
from scipy import stats

def get_actual_total_score(row):
    # SLS 2022 scoring: Top 1 run + Top 2 tricks
    runs = [row['run 1'], row['run 2']]
    tricks = [row['trick 1'], row['trick 2'], row['trick 3'], row['trick 4'], row['trick 5'], row['trick 6']]
    
    # Filter out NaNs if any (trick 5 and 6 are sometimes missing)
    runs = [r for r in runs if not np.isnan(r)]
    tricks = [t for t in tricks if not np.isnan(t)]
    
    best_run = max(runs) if runs else 0
    tricks.sort(reverse=True)
    best_tricks_sum = sum(tricks[:2]) if len(tricks) >= 2 else sum(tricks)
    
    return best_run + best_tricks_sum

def theta_mom(data):
    non_zero = data > 0
    return np.mean(non_zero) if len(non_zero) > 0 else 0.001

def alpha_beta_mom(data):
    pos_data = [x for x in data if x > 0]
    if len(pos_data) <= 1:
        # Fallback if we don't have enough data
        return 10.0, 5.0
    
    m = np.mean(pos_data)
    v = np.var(pos_data, ddof=1)
    
    # Check for invalid moments
    if v == 0 or (m*(1-m)/v - 1) <= 0:
         return 10.0, 5.0

    alpha = m * (m*(1-m)/v - 1)
    beta = (1-m) * (m*(1-m)/v - 1)
    return alpha, beta

def get_expected_score(theta, alpha, beta, run_alpha, run_beta):
    # We can use Monte Carlo simulation to get the expected total score for a skater
    N_SIMS = 1000
    total_scores = []
    
    for _ in range(N_SIMS):
        # 4 Tricks
        V = np.random.choice([0, 1], 4, p=[1-theta, theta])
        Z = np.random.beta(alpha, beta, 4)
        trick_scores = V * Z
        
        # 2 Runs
        run_scores = np.random.beta(run_alpha, run_beta, 2)
        
        best_run = max(run_scores)
        trick_scores.sort()
        best_tricks = sum(trick_scores[-2:])
        
        total_scores.append(best_run + best_tricks)
        
    return np.mean(total_scores)

def main():
    df = pd.read_csv("data/SLS22.csv")
    
    # Normalize scores (0-1 range)
    for col in ['run 1', 'run 2', 'trick 1', 'trick 2', 'trick 3', 'trick 4', 'trick 5', 'trick 6']:
        df[col] = df[col] / 10.0

    df['actual_total'] = df.apply(get_actual_total_score, axis=1)
    
    locations = df['location'].unique()
    
    overall_errors = []
    
    print("Running Leave-One-Competition-Out Cross-Validation...\n")
    
    for loc in locations:
        print(f"--- Holding out: {loc.upper()} ---")
        train_df = df[df['location'] != loc]
        test_df = df[df['location'] == loc]
        
        errors = []
        
        for _, test_row in test_df.iterrows():
            skater = test_row['id']
            actual_score = test_row['actual_total']
            
            # Get train data for this skater
            skater_train = train_df[train_df['id'] == skater]
            
            if len(skater_train) == 0:
                continue # Skater not in train set
                
            # Flatten tricks and runs
            tricks_list = skater_train[['trick 1', 'trick 2', 'trick 3', 'trick 4']].values.flatten()
            runs_list = skater_train[['run 1', 'run 2']].values.flatten()
            
            # Train parameters
            theta = theta_mom(tricks_list)
            alpha, beta = alpha_beta_mom(tricks_list)
            
            # Train run parameters (Assume runs don't have 0s like tricks, or treat similarly)
            run_alpha, run_beta = alpha_beta_mom(runs_list)
            
            # Predict
            expected_score = get_expected_score(theta, alpha, beta, run_alpha, run_beta)
            
            error = abs(expected_score - actual_score)
            errors.append(error)
            
            # print(f"{skater:<15} Actual: {actual_score:.3f} | Predicted: {expected_score:.3f} | MAE: {error:.3f}")
            
        loc_mae = np.mean(errors)
        overall_errors.extend(errors)
        print(f"Mean Absolute Error for {loc.upper()}: {loc_mae:.3f} (out of ~3.00 max)\n")

    print(f"=====================================")
    print(f"OVERALL MEAN ABSOLUTE ERROR: {np.mean(overall_errors):.3f}")
    print(f"=====================================")

if __name__ == '__main__':
    main()
