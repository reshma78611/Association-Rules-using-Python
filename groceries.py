# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:30:58 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


groceries=[]
#as data is transaction data we will be reading it directly
with open('groceries.csv') as f:
    groceries=f.read()

#splitting data in to separate transactions using separater as \n
groceries=groceries.split('\n')

groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(','))
    
all_groceries_list=[]
all_groceries_list=[i for item in groceries_list for  i in item]

from collections import Counter
item_frequencies=Counter(all_groceries_list)
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
plt.bar(height = frequencies[0:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[0:11]);plt.xlabel("items")
plt.ylabel("Count")

# Creating Data Frame for the transactions data 
# Purpose of converting all list into Series object Coz to treat each list element as entire element not to separate 
groceries_series=pd.DataFrame(pd.Series(groceries_list))
# removing the last empty transaction
groceries_series=groceries_series.iloc[:9835,:]
groceries_series.columns=['transactions']
# creating a dummy columns for the each item in each transactions ... Using column names as item name
x=groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

#implementing Aprioro algorithm
from mlxtend.frequent_patterns import apriori,association_rules
frequent_itemsets=apriori(x,min_support=0.05,max_len=3,use_colnames=True)
frequent_itemsets.shape
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
rules.shape
rules.head(1)
rules.sort_values('lift',ascending=False,inplace=True)

########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)


