# Distribution of 'Purchase' variable
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.sandbox.regression.sympy_diff import df

plt.figure(figsize=(10, 6))
sns.histplot(df['Purchase'], bins=30, kde=True)
plt.title('Distribution of Purchase Amounts')
plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')
plt.show()

# Explore average purchase amount by age group
plt.figure(figsize=(12, 6))
sns.barplot(x='Age', y='Purchase', data=df, estimator=np.mean)
plt.title('Average Purchase by Age Group')
plt.show()
