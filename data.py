import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of rows
n_rows = 10000

# 1. Generate base variables (no initial correlation)
active_sessions = np.random.randint(100, 5000, n_rows)
crc_errors = np.random.exponential(scale=0.01, size=n_rows)  # mostly near 0
crc_errors = np.clip(crc_errors, 0, 0.1)  # limit to maximum 0.1

buffer_utilization = np.random.uniform(0, 100, n_rows)
previous_unplanned_restarts = np.random.poisson(lam=0.5, size=n_rows)  # mostly 0, some 1, few >=2
previous_unplanned_restarts = np.clip(previous_unplanned_restarts, 0, 5)

dropped_packets = np.random.negative_binomial(2, 0.05, size=n_rows)  # long tail
dropped_packets = np.clip(dropped_packets, 0, 1000)

# 2. Calculate probability of target = 1 (failure) based on variables
# Logical formula: higher CRC errors, buffer full, previous restarts and drops increase probability
prob_raw = (0.3 * (crc_errors / 0.1) + 
            0.25 * (buffer_utilization / 100) + 
            0.2 * (previous_unplanned_restarts / 5) + 
            0.25 * (np.log1p(dropped_packets) / np.log1p(1000)))

# Normalize to [0,1] and add noise
prob = np.clip(prob_raw + np.random.normal(0, 0.1, n_rows), 0, 0.95)

# Adjust so that approximately 8% of cases have target=1 (infrequent event)
prob = prob * 0.5  # reduce base probability
prob = np.clip(prob, 0, 0.7)

# 3. Generate target via Bernoulli
target = np.random.binomial(1, prob)

# Ensure at least 5% positive events
if target.sum() < 0.05 * n_rows:
    missing = int(0.05 * n_rows) - target.sum()
    zero_indices = np.where(target == 0)[0]
    np.random.shuffle(zero_indices)
    target[zero_indices[:missing]] = 1

# 4. Slightly adjust variables to make sense with target=1
# (improves separability, useful for supervised learning)
for i in np.where(target == 1)[0]:
    crc_errors[i] = min(0.1, crc_errors[i] + np.random.uniform(0.01, 0.05))
    buffer_utilization[i] = min(100, buffer_utilization[i] + np.random.uniform(5, 15))
    dropped_packets[i] = min(1000, dropped_packets[i] + np.random.randint(10, 100))

# 5. Create DataFrame with English column names
df = pd.DataFrame({
    'active_sessions': active_sessions.astype(int),
    'crc_errors_per_second': np.round(crc_errors, 6),
    'buffer_memory_utilization_percent': np.round(buffer_utilization, 1),
    'unplanned_restarts_last_24h': previous_unplanned_restarts.astype(int),
    'dropped_packets_due_to_buffer_full_last_hour': dropped_packets.astype(int),
    'target': target
})

# 6. Save to CSV
df.to_csv('datos_routers.csv', index=False)


print(f"File 'router_data.csv' generated with {len(df)} rows.")
print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")