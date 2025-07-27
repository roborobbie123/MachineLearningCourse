import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import math
# total number of users
N = 10000
# number of ads
d = 10
ads_selected = []
# array of 10 0s
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

# UCB algorithm
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
           upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

# Visualizing the results
# Plot 1: Histogram of selections
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, d + 1), numbers_of_selections, color='blue', alpha=0.7)
plt.title('Number of Times Each Ad Was Selected')
plt.xlabel('Ad')
plt.ylabel('Selections')

# Plot 2: Rewards per ad
plt.subplot(1, 2, 2)
plt.bar(range(1, d + 1), sums_of_rewards, color='green', alpha=0.7)
plt.title('Total Rewards (Clicks) per Ad')
plt.xlabel('Ad')
plt.ylabel('Rewards')

plt.tight_layout()
plt.show()
        
