import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datset = pd.read_csv('Ads_CTR_Optimisation.csv')

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

