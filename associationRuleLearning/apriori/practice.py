import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# There is no header row, so we need to count the first row
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
# Looping through dataset and appending each order as an array to the transactions array
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training the apriori model on the dataset
from apyori import apriori
# min_support = product shows up in 21 transactions a week, use online calc
# min_confidence = play around with value starting from 0.8
# min_lift = 3
# min_length = want to be exactly 2 elements for this problem
# max_length = want to be exactly 2 elements for this problem
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualizing the results
results = list(rules)

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
resultsinDataFrame

# Results sorted by lift
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))
