# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df= pd.read_csv(path)
p_a=df[df['fico']>700].shape[0]/df.shape[0]
p_b = df[df['purpose']=='debt_consolidation'].shape[0]/df.shape[0]
df1 = df[df['purpose']=='debt_consolidation'].copy()
p_a_b = df1[df1['fico']>700].shape[0]/df.shape[0]/p_a
result = p_a_b==p_a
print(result)
# code ends here


# --------------
# code starts here
prob_lp = df[df['paid.back.loan']=='Yes'].shape[0]/df.shape[0]
prob_cs = df[df['credit.policy']=='Yes'].shape[0]/df.shape[0]
new_df = df[df['paid.back.loan']=='Yes'].copy()
prob_pd_cs = new_df[new_df['credit.policy']=='Yes'].shape[0]/df.shape[0]/prob_lp
bayes = prob_pd_cs*prob_lp/prob_cs
print(bayes)

# code ends here


# --------------
# code starts here
df['purpose'].value_counts().plot(kind='bar')
plt.show()
df1 = df[df['paid.back.loan']=='No'].copy()
df1['purpose'].value_counts().plot(kind='bar')
plt.show()
# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
inst_mean = df['installment'].mean()
df['installment'].hist()
df['log.annual.inc'].hist()


# code ends here


