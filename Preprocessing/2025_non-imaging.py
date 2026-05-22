import os
from tqdm import tqdm
import math
import random

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chisquare

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Retreiving fixed data (demographics, genetics, etc..) recorded at initial visit
EN00 = pd.read_csv(os.path.abspath('/Users/chloesavignac/_bzdok_lab_notebooks/Prevent-AD/internal/non-imaging-data/EL00_cleaned.csv'), index_col=0)
EN00 = EN00.set_index('PSCID_dem',drop=False).rename(columns={'PSCID_dem':'PSCID'}) # setting indixes to participant ids
EN00['Age_baseline_months'] = EN00.Candidate_Age_MoCA # keeping only one column for age
EN00[['PSCID','Age_baseline_months','Sex']].to_csv('deconf_infos.csv') # storing infos for deconfounding MRI data

# looping through medication
SU_MED = set()
for med in EN00.SU_medication.unique():
    if isinstance(med, str): 
        #print(med.split(';'))
        for e in med.split(';'):
            SU_MED.add(e)     

PRN_MED = set()
for med in EN00.PRN_medication.unique():
    if isinstance(med, str): 
        #print(med.split(';'))
        for e in med.split(';'):
            PRN_MED.add(e)

# looking for statins intake
statins = set()
for med in PRN_MED:
    if 'statin' in med:
        statins.add(med)    
print(statins)

statins = set()
for med in SU_MED:
    if 'statin' in med:
        statins.add(med)
print(statins)

# recording statins intake
statins = []
for med in EN00.SU_medication:
    if 'statin' in med:
        #print('yes')
        statins.append(1)
    else:
        #print('no')
        statins.append(0)

EN00['statins']=statins

# cleaning columns
EN00_cleaned = EN00.drop(columns=['PSCID',
                                 'INTREPAD_Tx_assignment',
                                 'Work',
                                 'Method_ApoE',
                                 'Method_BchE',
                                 'Method_HMGCR',
                                 'Method_TLR4',
                                 'Method_PPP2r1A',
                                 'Method_CDK5RAP2',
                                 'Study_visit_label',
                                 'Visit_label',
                                 'Date_taken',
                                 'SU_medication',
                                 'PRN_medication',
                                 'Date_taken_lab',
                                 'Date_taken_BP',
                                 'Date_taken_med_hist',
                                 'Date_taken_MoCA',
                                  '7_memory_NoPoint'
                                 ])
EN00_cleaned = EN00_cleaned.dropna(subset=['APOE'
                                          ])
EN00_cleaned_merged = EN00_cleaned
EN00_cleaned_merged['CandID'] = EN00_cleaned['CandID']
EN00_cleaned_merged['APOE'] = EN00_cleaned['APOE']
EN00_cleaned_merged = EN00_cleaned_merged.set_index('CandID').sort_index()
print(EN00_cleaned_merged.shape)

EN00_cleaned_merged.merge(EN00[['CandID','PSCID']],on='CandID').to_csv('EN00_01.27.23.csv')
EN00_cleaned_merged.to_csv('EN00_03.28.23.csv')

##############################################################################################################################################

# Longitudinal Data
#retreiving longitudinal data
BL00 = pd.read_csv(os.path.abspath('/Users/chloesavignac/_bzdok_lab_notebooks/Prevent-AD/internal/non-imaging-data/BL00_outter_merge.csv'), index_col=0)
FU12 = pd.read_csv(os.path.abspath('/Users/chloesavignac/_bzdok_lab_notebooks/Prevent-AD/internal/non-imaging-data/FU12_outter_merge.csv'), index_col=0)
FU24 = pd.read_csv(os.path.abspath('/Users/chloesavignac/_bzdok_lab_notebooks/Prevent-AD/internal/non-imaging-data/FU24_outter_merge.csv'), index_col=0)
FU36 = pd.read_csv(os.path.abspath('/Users/chloesavignac/_bzdok_lab_notebooks/Prevent-AD/internal/non-imaging-data/FU36_outter_merge.csv'), index_col=0)
FU48 = pd.read_csv(os.path.abspath('/Users/chloesavignac/_bzdok_lab_notebooks/Prevent-AD/internal/non-imaging-data/FU48_outter_merge.csv'), index_col=0)

#Canonical Variates
#see notebook called "CCA_modes"
BL00_CCA = pd.read_csv('BL00_CCA_modes.csv',index_col=0).rename(columns={'0':'PSCID'})
FU12_CCA = pd.read_csv('FU12_CCA_modes.csv',index_col=0).rename(columns={'0':'PSCID'})
FU24_CCA = pd.read_csv('FU24_CCA_modes.csv',index_col=0).rename(columns={'0':'PSCID'})
FU36_CCA = pd.read_csv('FU36_CCA_modes.csv',index_col=0).rename(columns={'0':'PSCID'})
FU48_CCA = pd.read_csv('FU48_CCA_modes.csv',index_col=0).rename(columns={'0':'PSCID'})

#using participant IDs for merging datasets
BL00['PSCID'] = BL00.PSCID_RBANS
FU12['PSCID'] = FU12.PSCID_RBANS
FU24['PSCID'] = FU24.PSCID_RBANS
FU36['PSCID'] = FU36.PSCID_RBANS
FU48['PSCID'] = FU48.PSCID_RBANS

BL00_merged = BL00.merge(BL00_CCA, on='PSCID', how = 'outer')
FU12_merged = FU12.merge(FU12_CCA, on='PSCID', how = 'outer')
FU24_merged = FU24.merge(FU24_CCA, on='PSCID', how = 'outer')
FU36_merged = FU36.merge(FU36_CCA, on='PSCID', how = 'outer')
FU48_merged = FU48.merge(FU48_CCA, on='PSCID', how = 'outer')

BL00_merged['PSCID'] = BL00_merged.PSCID_RBANS
FU12_merged['PSCID'] = FU12_merged.PSCID_RBANS
FU24_merged['PSCID'] = FU24_merged.PSCID_RBANS
FU36_merged['PSCID'] = FU36_merged.PSCID_RBANS
FU48_merged['PSCID'] = FU48_merged.PSCID_RBANS

BL00_merged = BL00_merged.set_index('CandID', drop = False).sort_index()
FU12_merged = FU12_merged.set_index('CandID', drop = False).sort_index()
FU24_merged = FU24_merged.set_index('CandID', drop = False).sort_index()
FU36_merged = FU36_merged.set_index('CandID', drop = False).sort_index()
FU48_merged = FU48_merged.set_index('CandID', drop = False).sort_index()

# concatenating all time points together
dfs = [BL00_merged,FU12_merged,FU24_merged,FU36_merged,FU48_merged]
all_time_points = pd.concat(dfs, keys=["BL00", "FU12","FU24","FU36",'FU48'])
all_time_points.select_dtypes(include=['object']).columns
all_time_points = all_time_points.rename(columns={'diagnosis':'anosmia_diagnosis'})
all_time_points['anosmia_diagnosis'].isna().sum()
all_time_points['RBANS_version'].isna().sum()
all_time_points.to_csv('all_time_points_01.27.23.csv')

##############################################################################################################################################

# Data Cleaning
all_time_points = pd.read_csv('all_time_points_01.27.23.csv')
all_time_points = all_time_points.rename(columns={'Unnamed: 0':'Visit'})

# setting multilevel indices 
inds = pd.MultiIndex.from_frame(all_time_points.iloc[:,0:2])
all_time_points = pd.DataFrame(np.array(all_time_points.iloc[:,2:]), index=inds, columns = all_time_points.iloc[:,2:].columns)

all_time_points = pd.read_csv('all_time_points_01.27.23.csv')
all_time_points = all_time_points.rename(columns={'Unnamed: 0':'Visit'})

# setting multilevel indices 
inds = pd.MultiIndex.from_frame(all_time_points.iloc[:,0:2])
all_time_points = pd.DataFrame(np.array(all_time_points.iloc[:,2:]), index=inds, columns = all_time_points.iloc[:,2:].columns)

#selecting all columns with age 
age_cols = ['Candidate_Age','Candidate_Age_Aud_pro','Candidate_Age_BP_Pulse_Weight','Candidate_Age_CSF_Proteins','Candidate_Age_lab','Candidate_Age_Med_use','Candidate_Age_RBANS','Candidate_Age_Smell']
## Use Candidate_Age_RBANS to establish time difference between visits (relative to initial visit)
differences = []
ids = []

# loop through all participants
for participant in all_time_points.index.get_level_values('CandID').unique():
    # print(participant)
    # select all of their visits
    visits = all_time_points.xs(participant, level=1)
    # print(visits)
    t1 = visits.iloc[0,:]
    # for T1
    diff = 0    
    differences.append(diff)
    ids.append(participant)  
    # verify if at least two visits    
    if len(visits.index)>1:
        for i in range(1,len(visits.index)):
            # for subsequent time points 
            t2 = visits.iloc[i,:]
            # computing time difference since T1 based on age at assessement in months 
            diff = t2.Candidate_Age_RBANS - t1.Candidate_Age_RBANS 
            differences.append(diff)
            ids.append(participant)
diff_df = pd.DataFrame(np.array(differences), columns = ['Time_diff_months'], index = ids)

# adding a time difference column to our dataset
all_time_points['time_diff'] = differences
# dropping columns
all_time_points = all_time_points.drop(columns=[
                              'PSCID_AD8',
                              'CandID.1',
                              'PSCID_APS',
                              'PSCID',
                              'PSCID_BP_Pulse_Weight',
                              'PSCID_CSF_Proteins',
                              'PSCID_lab',
                              'PSCID_Med_use',
                              'PSCID_Smell',
                              'PSCID_RBANS',
                              'Study_visit_label_AD8',
                              'Visit_label_AD8',
                              'Date_taken',
                              'Study_visit_label_APS',
                              'Visit_label_APS',
                              'Study_visit_label',
                              'Visit_label',
                              'Date_taken_Aud_pro',
                              'Study_visit_label_BP_Pulse_Weight',
                              'Visit_label_BP_Pulse_Weight',
                              'Date_taken_BP_Pulse_Weight',
                              'Study_visit_label_lab',
                              'Visit_label_lab',
                              'Date_taken_lab',
                              'Study_visit_label_Med_use',
                              'Visit_label_Med_use',
                              'Date_taken_Med_use',
                              'Study_visit_label_RBANS',
                              'Visit_label_RBANS',
                              'Date_taken_RBANS',
                              'Study_visit_label_CSF_Proteins',
                              'Visit_label_CSF_Proteins',
                              'Date_taken_CSF_Proteins',
                              'Candidate_Age',
                              'Candidate_Age_Aud_pro',
                              'Candidate_Age_BP_Pulse_Weight',
                              'Candidate_Age_CSF_Proteins',
                              'Candidate_Age_lab',
                              'Candidate_Age_Med_use',
                              'Candidate_Age_Smell',
                              'Study_visit_label_Smell',
                              'Visit_label_Smell', 
                              'Date_taken_Smell',
                              'comments_uncategorized',
                              'SU_medication',
                              'PRN_medication',
                             ])

all_time_points = all_time_points.rename(columns={'Candidate_Age_RBANS':'Candidate_Age'})
list(all_time_points.columns)
print(all_time_points.isna().sum())
print(all_time_points.index.get_level_values('CandID').isna().sum())
print(all_time_points.shape)

# Getting fixed demographic measures from baseline assessement
fixed = pd.read_csv('EN00_03.28.23.csv')
list(fixed.columns)

# appending fixed variables to our dataset
df = all_time_points.merge(fixed, on='CandID', how = 'outer', suffixes=[None,'_baseline'])
print(df.shape)

# dropping visit without brain imaging recording
df = df.dropna(subset=['50'])
print(df.shape)

# dropping visit without apoe screening
df = df.dropna(subset=['APOE'])
print(df.shape)

list(df.columns)
candids = list(df.CandID)

# column is there twice
df = df.drop(columns=['probable_MCI_visit'])
df = df.rename(columns={'probable_MCI_visit_baseline':'probable_MCI_visit'})

# Missing Data Imputation
def numerize(col):
    cat = list(pd.Categorical(df[col],ordered=True).categories)
    code_num = list(pd.Series(pd.Categorical(df[col],
                              ordered=True).codes).replace(({-1: np.nan})).value_counts().sort_index().index)
    df[col] = list(pd.Series(pd.Categorical(df[col],ordered=True).codes).replace(({-1: np.nan})))
    print(cat)
    return cat,code_num

codes = dict()
codes_num = dict()
for col in ['BchE_K_variant','BDNF','HMGCR_Intron_M','TLR4_rs_4986790',
            'PPP2r1A_rs_10406151','CDK5RAP2_rs10984186',
            'Work_by_category','Handedness_interpretation','Mother_tongue',
            'Test_language','Ethnicity','Education_level','Retirement_status',
            'APOE','RBANS_version','anosmia_diagnosis','Sex',
            'probable_MCI','probable_MCI_visit']:
    
    code,code_num = numerize(col)

    codes[col] = code
    codes_num[col] = code_num

print(codes)
print(codes_num)

np.random.seed(0) # fixed for reproducibility
def my_impute(df):
    new_df = pd.DataFrame()
    for col in tqdm(df.columns[1:]):
        print(col)
        arr = pd.to_numeric(df[col])
        arr = np.array(arr)
        b_nan = np.isnan(arr)
        b_negative = arr < 0
        b_bad = b_nan | b_negative
        arr[b_bad] = np.random.choice(arr[~b_bad], np.sum(b_bad))
        new_df[col] = arr
    return new_df

imputed_df = my_impute(df)

imputed_df['CandID'] = list(df.CandID)
codes
codes_num
for key in codes:
    print(key)
    str_values = codes[key]
    num_values = codes_num[key]
    for i in num_values:
        print(i)
        print(str_values[int(i)])
        imputed_df[key] = imputed_df[key].replace({int(i):str_values[int(i)]})

print(imputed_df['Education_level'].value_counts())
print(imputed_df.isna().sum().sum())

print(imputed_df['Ethnicity'].value_counts())
print(imputed_df.isna().sum())

imputed_df.to_csv('prevent_AD_data_jan_2023.csv')

##############################################################################################################################################

# Dummy Coding
pd.read_csv('prevent_AD_data_jan_2023.csv',index_col=0)
df = pd.read_csv('prevent_AD_data_jan_2023.csv',index_col=0)
print(df.head())
print(df.APOE)

# keeping only one column for age
df['Age_baseline_months'] = df.Candidate_Age_MoCA
list(df.select_dtypes(include=['object']).columns)
df[['CandID','Age_baseline_months','Sex']].to_csv('deconf_infos.csv')

# separate numerical and non-numerical data
df_cleaned_num = df.select_dtypes(exclude=['object'])
df_cleaned_cat = df.select_dtypes(include=['object'])

# dummy-code all categorical data
df_cleaned_cat = pd.get_dummies(df_cleaned_cat, dummy_na=False)
print(df_cleaned_cat.shape)

# merge back categorical and numerical variables
df_cleaned_merged = df_cleaned_num.join(df_cleaned_cat)
print(df_cleaned_merged.shape)

df_cleaned_merged['APOE']=df.APOE
df_cleaned_merged.to_csv('prevent_AD_data_dummy_jan_2023.csv')

# make one column for APOE
allele1 = []
allele2 = []

apoe_unique = df[['CandID','APOE']].drop_duplicates()
print(apoe_unique.shape)

for geno in apoe_unique.APOE:
    allele1.append(geno.split(' ')[0])
    allele2.append(geno.split(' ')[1])

print(apoe_unique.APOE.value_counts()/368)

##############################################################################################################################################

# Family lineage encoding (keeping individuals with ONLY maternal and ONLY paternal)
df = pd.read_csv('prevent_AD_data_dummy_jan_2023.csv', index_col=0)
df = pd.read_csv('prevent_AD_data_dummy_jan_2023.csv', index_col=0)
print(df.shape)

# Maternal AD history
only_mother = []
for i in range(0,len(df)):
    if (df['father_dx_ad_dementia'][i]==0)&(df['mother_dx_ad_dementia'][i]==1):
        only_mother.append(1)
    else: 
        only_mother.append(0)
df['only_mother'] = only_mother

# Paternal AD history
only_father = []
for i in range(0,len(df)):
    if (df['father_dx_ad_dementia'][i]==1)&(df['mother_dx_ad_dementia'][i]==0):
        only_father.append(1)
    else: 
        only_father.append(0)
df['only_father'] = only_father

# Both maternal and paternal (to be dropped)
both = []
for i in range(0,len(df)):
    if (df['father_dx_ad_dementia'][i]==1)&(df['mother_dx_ad_dementia'][i]==1):
        both.append(1)
    else: 
        both.append(0)
df['both'] = both

df.to_csv('prevent_AD_data_dummy_jan_2023_fam_hist.csv')

df = df.drop(df.iloc[np.where(df.both==1)].index).reset_index(drop=True) # dropping individuals with both maternal and paternal lineage
print(df.shape)

# drop individuals with only sibling history
df = df.drop(index=np.where((df['father_dx_ad_dementia']==0)&(df['mother_dx_ad_dementia']==0)&(df['sibling_dx_ad_dementia']==1))[0]).reset_index(drop=True)
print(df.shape)

id_f = np.where(df.Sex_Female==1) # get women id (sex is self-reported)
id_m = np.where(df.Sex_Male==1) # get men id
id_all = df.index

# get apoe genotypes 
apoe = df.APOE

cca_cols = []
for i in range(1,51):
    cca_cols.append(f"{i}")

# keep one-hot encoded APOE genotypes
# APOE e4e4 excluded (not enough individuals)
apoe_gen = [ 'APOE_3 2',
 'APOE_3 3',
 'APOE_4 3']

# print(df.iloc[np.where((df.Sex_Male==1)&(df.APOE=='4 4'))].CandID.unique())
# print(df.iloc[np.where((df.Sex_Male==0)&(df.APOE=='4 4'))].CandID.unique())
# print(df.iloc[np.where((df.Sex_Male==1)&(df.APOE=='4 2'))].CandID.unique())
# print(df.iloc[np.where((df.Sex_Female==1)&(df.APOE=='4 2'))].CandID.unique())

# Genotypes
gen_all = pd.DataFrame(df.APOE.value_counts()/len(df))
gen_all = gen_all.sort_index()
print(gen_all.head())

# males
gen_m = pd.DataFrame(df.iloc[id_m].APOE.value_counts()/len(id_m[0]))
gen_m = gen_m.sort_index()
print(gen_m.head())

# females
gen_f = pd.DataFrame(df.iloc[id_f].APOE.value_counts()/len(id_f[0]))
gen_f = gen_f.sort_index()
print(gen_f.head())

##############################################################################################################################################

# Compute summary statistics
print(chisquare(gen_m.APOE, f_exp=gen_all.APOE))
print(chisquare(gen_f.APOE, f_exp=gen_all.APOE))
print(chisquare(gen_m.APOE, f_exp=gen_f.APOE))

df.iloc[np.where((df.APOE=='4 4') & (df.Sex_Female==1))].CandID.unique()
df.iloc[np.where((df.APOE=='4 4') & (df.Sex_Female==0))].CandID.unique()
e4e4 = list(np.where((df.APOE=='4 4'))[0])

test_df = df.drop(index=e4e4) # drop all e4e4 because only 1 male with this genotype
test_df = test_df.reset_index(drop=True)

e2e4 = list(np.where((test_df.APOE=='4 2'))[0]) # drop all e2e4 because only 1 male with this genotype
test_df = test_df.drop(index=e2e4)

test_df = test_df.reset_index(drop=True)
print(test_df.APOE.value_counts())
print(test_df.shape)

##############################################################################################################################################

# Balancing % of men and women with maternal vs. paternal lineage across APOE genotype groups
random.seed(10) # setting random seed for reproducibility

allele1 = []
allele2 = []

apoe_unique = test_df[['CandID','APOE']].drop_duplicates()
print(apoe_unique.shape)

for geno in apoe_unique.APOE:
    allele1.append(geno.split(' ')[0])
    allele2.append(geno.split(' ')[1])

# print confusion matrix
cm = confusion_matrix(allele1, allele2, sample_weight=None, normalize=None)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ['APOE-E2','APOE-E3','APOE-E4'])
disp.plot(values_format = '.7g', cmap='Blues')
plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')

# Droping 120 females e3e3 at random (larger group)
f_e3e3 = list(np.where((test_df.APOE=='3 3') & (test_df.Sex_Female==1))[0])
test_df = test_df.drop(index=random.sample(f_e3e3, 95)).reset_index(drop=True)

# dropping 20 females e3e4 at random
f_e4e3 = list(np.where((test_df.APOE=='4 3') & (test_df.Sex_Female==1))[0])
test_df = test_df.drop(index=random.sample(f_e4e3, 55)).reset_index(drop=True)

# family history: drop 100 maternal cases at random (larger group)
f_mother = list(np.where((test_df.Sex_Female==1) & (test_df.only_mother==1))[0])
test_df = test_df.drop(index=random.sample(f_mother, 100)).reset_index(drop=True)

id_f = np.where(test_df.Sex_Female==1)
id_m = np.where(test_df.Sex_Male==1)
id_all = test_df.index

# PRINTING BALANCED GROUPS
gen_all = pd.DataFrame(test_df.APOE.value_counts()/len(test_df))
gen_all = gen_all.sort_index()
print('all')
print(gen_all.head())

# males
gen_m = pd.DataFrame(test_df.iloc[id_m].APOE.value_counts()/len(test_df.iloc[id_m]))
gen_m = gen_m.sort_index()
print('males')
print(gen_m.head())

# females
gen_f = pd.DataFrame(test_df.iloc[id_f].APOE.value_counts()/len(test_df.iloc[id_f]))
gen_f = gen_f.sort_index()
print('females')
print(gen_f.head())

gen_all = pd.DataFrame(test_df.only_mother.value_counts()/len(test_df))
gen_all = gen_all.sort_index()
print('all')
print(gen_all.head())

# males
gen_m = pd.DataFrame(test_df.iloc[id_m].only_mother.value_counts()/len(test_df.iloc[id_m]))
gen_m = gen_m.sort_index()
print('males')
print(gen_m.head())

# females
gen_f = pd.DataFrame(test_df.iloc[id_f].only_mother.value_counts()/len(test_df.iloc[id_f]))
gen_f = gen_f.sort_index()
print('females')
print(gen_f.head())

# print confusion matrix of balanced dataset
test_df.iloc[id_f]
test_df.iloc[id_f][['CandID','APOE']].drop_duplicates()

allele1 = []
allele2 = []

apoe_unique = test_df.iloc[np.where(test_df.only_mother==0)][['CandID','APOE']].drop_duplicates()
print(apoe_unique.shape)

for geno in apoe_unique.APOE:
    allele1.append(geno.split(' ')[0])
    allele2.append(geno.split(' ')[1])

cm = confusion_matrix(allele1, allele2, sample_weight=None, normalize=None)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ['APOE-E2','APOE-E3','APOE-E4'])
disp.plot(values_format = '.7g', cmap='Blues')
plt.savefig('balanced_confusion_matrix.png', dpi=200, bbox_inches='tight')

balanced_df = test_df
balanced_df.to_csv('apoe_fh_balanced_df_03.28.23.csv')