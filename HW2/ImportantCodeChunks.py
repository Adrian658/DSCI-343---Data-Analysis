import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import scipy.stats as scipystats
import sklearn.feature_selection as skfeatureselect
import math
import time

"""
This is the code used for Monte-Carlo simulations
"""
def exact_mc_perm_test(xs, ys, nmc, statistic='normal'):
    n = len(xs)
    k = 0
    #Difference between means of both vectors
    diff = np.abs(np.mean(xs) - np.mean(ys))
    m_score = metrics.mutual_info_score(xs, ys)
    j_score = metrics.jaccard_score(xs, ys)
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        if statistic == 'normal':
            #increment if difference between means is greater than original difference
            k += diff <= np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
        elif statistic == 'mutual_info':
            #If mutual score of permuted version is lower, increment count
            k += m_score > metrics.mutual_info_score(zs[:n], zs[n:])
        elif statistic == 'jaccard_index':
            #If jscore of permuted version is lower, increment count
            k += j_score > metrics.jaccard_score(zs[:n], zs[n:])
    return (k+1)/ (nmc+1)

"""
These are the methods used to get the pvalues given two vectors
"""
def get_mutual_info_pval(v1, v2, N=10000):
    return exact_mc_perm_test(v1, v2, N, 'mutual_info')

def get_jaccard_pval(v1, v2, N=10000):
    return exact_mc_perm_test(v1, v2, N, 'jaccard_index')

def get_chi_pval(v1, v2, N=10000):
    frequencies = {
        "00": 0,
        "01": 0,
        "10": 0,
        "11": 0
    }
    
    for x in range(len(v1)):
        frequencies[str(v1[x]) + str(v2[x])] += 1
        
    frequencies = list(frequencies.values())
        
    return scipy.stats.chi2_contingency(np.array([frequencies, expected_frequencies]))


"""
This is the code used for calculating all statistics for all pairs in 1b
"""
df = pd.DataFrame(columns=['MutualInfo', 'MutualInfoPval', 'JaccardIndex', 'JaccardPval', 'ChiSquared', 'ChiSquaredPval'])
start = time.time()
for idx, iterable_gene1 in enumerate(p1b.columns):
    for iterable_gene2 in p1b.columns[idx+1:]:
        gene1 = p1b[iterable_gene1]
        gene2 = p1b[iterable_gene2]
        #Do calculations with the two gene vectors
        mutual_info = metrics.normalized_mutual_info_score(gene1, gene2)
        mutual_info_pval = get_mutual_info_pval(gene1, gene2, 500)
        jaccard_index = metrics.jaccard_score(gene1, gene2)
        jaccard_pval = get_jaccard_pval(gene1, gene2, 500)
        chi_squared = get_chi_pval(gene1, gene2)
        chi_squared_stat = chi_squared[0]
        chi_squared_pval = chi_squared[1]
        #Add info to a dictionary
        df.loc[iterable_gene1 + ',' + iterable_gene2] = [mutual_info, mutual_info_pval, jaccard_index, jaccard_pval, chi_squared_stat, chi_squared_pval]
print(time.time() - start)


"""
This is the code used for generating the BH p-value thresholds
"""
a = 0.05
series = df['ChiSquaredPvalNorm'].sort_values()
adj_ps = []
for x,y in enumerate(series):
    adj_p = (x/len(series)) * a
    adj_ps.append(adj_p)
new = pd.DataFrame(columns=['OriginalPval', 'BHThreshold'])
new['OriginalPval'] = series
new['BHThreshold'] = adj_ps