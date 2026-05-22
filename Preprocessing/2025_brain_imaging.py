from sklearn.cross_decomposition import CCA
import pandas as pd
import joblib
import numpy as np
import os
import sklearn
from sklearn.preprocessing import StandardScaler
from nilearn.signal import clean

print(joblib.__version__)
print(sklearn.__version__)
pop_cca = joblib.load('org_CCA_k25_lonely_full')

# PREVENT-AD data 
# HC data 
BL00_HC = pd.read_csv('HC_segmentation/csv/BL00_HC_left_right.csv', index_col=0)
FU12_HC = pd.read_csv('HC_segmentation/csv/FU12_HC_left_right.csv', index_col=0)
FU24_HC = pd.read_csv('HC_segmentation/csv/FU24_HC_left_right.csv', index_col=0)
FU36_HC = pd.read_csv('HC_segmentation/csv/FU36_HC_left_right.csv', index_col=0)
FU48_HC = pd.read_csv('HC_segmentation/csv/FU48_HC_left_right.csv', index_col=0)

# DN data
BL00_DN = pd.read_csv('BL00_DN.csv', index_col=0)
FU12_DN = pd.read_csv('FU12_DN.csv', index_col=0)
FU24_DN = pd.read_csv('FU24_DN.csv', index_col=0)
FU36_DN = pd.read_csv('FU36_DN.csv', index_col=0)
FU48_DN = pd.read_csv('FU48_DN.csv', index_col=0)

col_HC = list(BL00_HC.columns)[0:-4]
col_HC.remove('Whole_hippocampal_body_left')
col_HC.remove('Whole_hippocampal_head_left')
col_HC.remove('Whole_hippocampus_left')
col_HC.remove('PSCID_left')
col_DN = BL00_DN.columns

def deconf(beh):

    age = StandardScaler().fit_transform(beh['Age_baseline_months'].values[:, np.newaxis])  # Age at recruitment
    age2 = age ** 2
    sex = np.array(pd.get_dummies(beh['Sex']).values, dtype=np.int)  # Sex (self-reported)
    sex_x_age = sex * age
    sex_x_age2 = sex * age2
    
    print('Deconfounding for sex and age!')
    conf_mat = np.hstack([
        age, age2, sex, sex_x_age, sex_x_age2,
        ])
    
    return conf_mat

##############################################################################################################################################

# Deconfounding
# removing two participants without HC volumes based on index
BL00_DN = BL00_DN.iloc[np.where(BL00_DN.index.isin(BL00_HC[col_HC].index))]
# removing participants without DN volumes based on index
BL00_HC = BL00_HC.iloc[np.where(BL00_HC.index.isin(BL00_DN.index))]
# we want the sex of each participant based on PSCID
infos = pd.read_csv('deconf_infos.csv',index_col=0)
print(infos.isna().sum())

col_HC = list(BL00_HC.columns)[0:-4]
col_HC.remove('Whole_hippocampal_body_left')
col_HC.remove('Whole_hippocampal_head_left')
col_HC.remove('Whole_hippocampus_left')
col_HC.remove('PSCID_left')
print(BL00_HC[col_HC].shape)
print(BL00_HC[col_HC].isna().sum())

# Deconfounding for sex and age 
# HIPPOCAMPUS 
col_HC.append('PSCID_left')
hc_infos = BL00_HC[col_HC].merge(infos, how='left',left_on='PSCID_left', right_on='PSCID')
print(hc_infos.shape)

conf_mat_hc = deconf(hc_infos)
print(conf_mat_hc.shape)

X_scaler = StandardScaler()
col_HC.remove('PSCID_left')
X = X_scaler.fit_transform(BL00_HC[col_HC])
print(X.shape)

hc = clean(X, confounds=conf_mat_hc, detrend=False, standardize=False)
print(hc.shape)

pd.DataFrame(hc, columns=col_HC).set_index(BL00_HC.index).to_csv('BL00_HC_deconf.csv')
BL00_DN['PSCID'] = BL00_DN.index

# Deconfounding for sex and age 
# DEFAULT NETWORK
dn_infos = BL00_DN.merge(infos, how='left',left_on='PSCID', right_on='PSCID')
print(dn_infos.shape)

conf_mat_dn = deconf(dn_infos)
print(conf_mat_dn.shape)

X_scaler = StandardScaler()
BL00_DN = BL00_DN.drop(columns='PSCID')
X = X_scaler.fit_transform(BL00_DN)
print(X.shape)

dn = clean(X, confounds=conf_mat_dn, detrend=False, standardize=False)
print(dn.shape)

# get x_scores and y_scores from PreventAD
pop_cca = joblib.load('org_CCA_k25_lonely_full')
hc, dn = pop_cca.transform(hc,dn)

cca_BL00 = pd.DataFrame(
#stacking data in a single numpy array    
    np.vstack((
        BL00_HC.index.T, hc.T, dn.T))).T
np.where(cca_BL00==0)

cca_BL00.to_csv('BL00_CCA_modes.csv')

##############################################################################################################################################

# FU12
# removing two participants without HC volumes based on index
FU12_DN = FU12_DN.iloc[np.where(FU12_DN.index.isin(FU12_HC[col_HC].index))]
#removing two participants without HC volumes based on index
FU12_HC = FU12_HC.iloc[np.where(FU12_HC.index.isin(FU12_DN.index))]

# Deconfounding for sex and age 
# HIPPOCAMPUS 
col_HC.append('PSCID_left')
hc_infos = FU12_HC[col_HC].merge(infos, how='left',left_on='PSCID_left', right_on='PSCID')
print(hc_infos.shape)

conf_mat_hc = deconf(hc_infos)
print(conf_mat_hc.shape)

X_scaler = StandardScaler()
col_HC.remove('PSCID_left')
X = X_scaler.fit_transform(FU12_HC[col_HC])
print(X.shape)

hc = clean(X, confounds=conf_mat_hc, detrend=False, standardize=False)
print(hc.shape)

pd.DataFrame(hc, columns=col_HC).set_index(FU12_HC.index).to_csv('FU12_HC_deconf.csv')

# Deconfounding for sex and age 
# DEFAULT NETWORK
FU12_DN['PSCID'] = FU12_DN.index
dn_infos = FU12_DN.merge(infos, how='left',left_on='PSCID', right_on='PSCID')
print(dn_infos.shape)

conf_mat_dn = deconf(dn_infos)
print(conf_mat_dn.shape)

X_scaler = StandardScaler()
FU12_DN = FU12_DN.drop(columns='PSCID')
X = X_scaler.fit_transform(FU12_DN)
print(X.shape)

dn = clean(X, confounds=conf_mat_dn, detrend=False, standardize=False)
print(dn.shape)

pd.DataFrame(dn, columns=col_DN).set_index(FU12_DN.index).to_csv('FU12_DN_deconf.csv')
# get x_scores and y_scores from PreventAD
pop_cca = joblib.load('org_CCA_k25_lonely_full')
hc, dn = pop_cca.transform(hc,dn)

cca_FU12 = pd.DataFrame(
# stacking data in a single numpy array    
    np.vstack((
        FU12_HC.index.T, hc.T, dn.T))).T

cca_FU12.to_csv('FU12_CCA_modes.csv')

##############################################################################################################################################

# FU24
# removing two participants without HC volumes based on index
FU24_DN = FU24_DN.iloc[np.where(FU24_DN.index.isin(FU24_HC[col_HC].index))]
FU24_HC = FU24_HC.iloc[np.where(FU24_HC.index.isin(FU24_DN.index))]

# Deconfounding for sex and age 
# HIPPOCAMPUS 
col_HC.append('PSCID_left')
hc_infos = FU24_HC[col_HC].merge(infos, how='left',left_on='PSCID_left', right_on='PSCID')
print(hc_infos.shape)

conf_mat_hc = deconf(hc_infos)
print(conf_mat_hc.shape)

X_scaler = StandardScaler()
col_HC.remove('PSCID_left')
X = X_scaler.fit_transform(FU24_HC[col_HC])
print(X.shape)

hc = clean(X, confounds=conf_mat_hc, detrend=False, standardize=False)
print(hc.shape)

pd.DataFrame(hc, columns=col_HC).set_index(FU24_HC.index).to_csv('FU24_HC_deconf.csv')

# Deconfounding for sex and age 
# DEFAULT NETWORK
FU24_DN['PSCID'] = FU24_DN.index
dn_infos = FU24_DN.merge(infos, how='left',left_on='PSCID', right_on='PSCID')
print(dn_infos.shape)

conf_mat_dn = deconf(dn_infos)
print(conf_mat_dn.shape)

X_scaler = StandardScaler()
FU24_DN = FU24_DN.drop(columns='PSCID')
X = X_scaler.fit_transform(FU24_DN)
print(X.shape)

dn = clean(X, confounds=conf_mat_dn, detrend=False, standardize=False)
print(dn.shape)

pd.DataFrame(dn, columns=col_DN).set_index(FU24_DN.index).to_csv('FU24_DN_deconf.csv')
# get x_scores and y_scores from PreventAD
hc, dn = pop_cca.transform(hc,dn)

cca_FU24 = pd.DataFrame(
# stacking data in a single numpy array    
    np.vstack((
        FU24_HC.index.T, hc.T, dn.T))).T

cca_FU24.to_csv('FU24_CCA_modes.csv')

##############################################################################################################################################

# FU36
FU36_HC['PSCID'] = FU36_HC.index
# removing two participants without HC volumes based on index
FU36_DN = FU36_DN.iloc[np.where(FU36_DN.index.isin(FU36_HC[col_HC].index))]
FU36_HC = FU36_HC.iloc[np.where(FU36_HC.index.isin(FU36_DN[col_DN].index))]

# Deconfounding for sex and age 
# HIPPOCAMPUS 
col_HC.append('PSCID')
hc_infos = FU36_HC[col_HC].merge(infos, how='left',left_on='PSCID', right_on='PSCID')
print(hc_infos.shape)

conf_mat_hc = deconf(hc_infos)
print(conf_mat_hc.shape)

X_scaler = StandardScaler()
col_HC.remove('PSCID')
X = X_scaler.fit_transform(FU36_HC[col_HC])
print(X.shape)

hc = clean(X, confounds=conf_mat_hc, detrend=False, standardize=False)
print(hc.shape)

pd.DataFrame(hc, columns=col_HC).set_index(FU36_HC.index).to_csv('FU36_HC_deconf.csv')

# Deconfounding for sex and age 
# DEFAULT NETWORK
FU36_DN['PSCID'] = FU36_DN.index
dn_infos = FU36_DN.merge(infos, how='left',left_on='PSCID', right_on='PSCID')
print(dn_infos.shape)

conf_mat_dn = deconf(dn_infos)
print(conf_mat_dn.shape)

X_scaler = StandardScaler()
FU36_DN = FU36_DN.drop(columns='PSCID')
X = X_scaler.fit_transform(FU36_DN)
print(X.shape)

dn = clean(X, confounds=conf_mat_dn, detrend=False, standardize=False)
print(dn.shape)

pd.DataFrame(dn, columns=col_DN).set_index(FU36_DN.index).to_csv('FU36_DN_deconf.csv')
# get x_scores and y_scores from PreventAD
hc, dn = pop_cca.transform(hc,dn)

cca_FU36 = pd.DataFrame(
# stacking data in a single numpy array    
    np.vstack((
        FU36_HC.index.T, hc.T, dn.T))).T

cca_FU36.to_csv('FU36_CCA_modes.csv')

##############################################################################################################################################

# FU48
# removing two participants without HC volumes based on index
FU48_DN = FU48_DN.iloc[np.where(FU48_DN.index.isin(FU48_HC[col_HC].index))]
FU48_HC = FU48_HC.iloc[np.where(FU48_HC.index.isin(FU48_DN[col_DN].index))]

# Deconfounding for sex and age 
# HIPPOCAMPUS
FU48_HC['PSCID'] = FU48_HC.index
col_HC.append('PSCID')
hc_infos = FU48_HC[col_HC].merge(infos, how='left',left_on='PSCID', right_on='PSCID')
print(hc_infos.shape)

conf_mat_hc = deconf(hc_infos)
print(conf_mat_hc.shape)

X_scaler = StandardScaler()
col_HC.remove('PSCID')
X = X_scaler.fit_transform(FU48_HC[col_HC])
print(X.shape)

hc = clean(X, confounds=conf_mat_hc, detrend=False, standardize=False)
print(hc.shape)

pd.DataFrame(hc, columns=col_HC).set_index(FU48_HC.index).to_csv('FU48_HC_deconf.csv')

#Deconfounding for sex and age 
#DEFAULT NETWORK
FU48_DN['PSCID'] = FU48_DN.index
dn_infos = FU48_DN.merge(infos, how='left',left_on='PSCID', right_on='PSCID')
print(dn_infos.shape)

conf_mat_dn = deconf(dn_infos)
print(conf_mat_dn.shape)

X_scaler = StandardScaler()
FU48_DN = FU48_DN.drop(columns='PSCID')
X = X_scaler.fit_transform(FU48_DN)
print(X.shape)

dn = clean(X, confounds=conf_mat_dn, detrend=False, standardize=False)
print(dn.shape)

pd.DataFrame(dn, columns=col_DN).set_index(FU48_DN.index).to_csv('FU48_DN_deconf.csv')

# get x_scores and y_scores from PreventAD
hc, dn = pop_cca.transform(hc,dn)

cca_FU48 = pd.DataFrame(
#stacking data in a single numpy array    
    np.vstack((
        FU48_HC.index.T, hc.T, dn.T))).T

cca_FU48.to_csv('FU48_CCA_modes.csv')