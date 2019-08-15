import pandas as pd
import numpy as np
import seaborn as sns
import os 
import sklearn

import matplotlib.pyplot as plt
# C:\Users\585162\AppData\Local\Continuum\anaconda2
path_in = r'C:\\Users\\585162\\Documents\\GWCM_NLP-master\\GWCM_NLP-master\data\industrial_products_and_services'

file_path = os.path.join(path_in, 'I_K_Jansan_MRO_08_1819_cleaned_stop_tableau.csv')

df = pd.read_csv(file_path)

df=df.assign(PSC = df.PSC.astype('str').str.strip(),
df['MFR_PART_NO_BY_VENDOR'] = df.MFR_PART_NO_BY_VENDOR.astype('str').str.strip())

# calculate the count of MFR associated with a specific PSC
aggDF = df[df.PSC!='0'].groupby(['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned','PSC']).size()
aggDF=aggDF.pipe(pd.DataFrame).rename(columns = {0:'PSC_MFR_COUNT'}).reset_index()

# calculate total number of MFR (excluding untagged items)
mscDF = df[df.PSC!='0'].groupby(['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned']).size().pipe(pd.DataFrame).rename(columns = {0:'MSC_COUNT'})

# merge the two datasets so that we can create percentages (aka probabilities) that a specific MFR will be assigned to a PSC
aggDF=aggDF.merge(mscDF, how = 'left', on =['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned'] )
aggDF=aggDF.assign(PCT_OF_TOTAL = aggDF.PSC_MFR_COUNT/aggDF.MSC_COUNT)


sns.distplot(aggDF.PCT_OF_TOTAL)
plt.show()

aggDF[(aggDF.PCT_OF_TOTAL< 1) & (aggDF.MSC_COUNT>10)].sort_values('MSC_COUNT',ascending = False).head()
aggDF[aggDF.MFR_PART_NO_BY_VENDOR =='WGCWWGR']


def get_ref_table(test_table):
    aggDF = test_table[test_table.PSC!='0'].groupby(['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned','PSC']).size()
    aggDF=aggDF.pipe(pd.DataFrame).rename(columns = {0:'PSC_MFR_COUNT'}).reset_index()

    mscDF = test_table[test_table.PSC!='0'].groupby(['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned']).size().pipe(pd.DataFrame).rename(columns = {0:'MSC_COUNT'})
    # merge the two datasets so that we can create percentages (aka probabilities) that a specific MFR will be assigned to a PSC
    aggDF=aggDF.merge(mscDF, how = 'left', on =['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned'] )
    aggDF=aggDF.assign(PCT_OF_TOTAL = aggDF.PSC_MFR_COUNT/aggDF.MSC_COUNT)

    return aggDF

def get_P1_match(table_in,ref_table ):
    ref_table = ref_table[ref_table.PCT_OF_TOTAL==1][['PSC','MFR_PART_NO_BY_VENDOR']]

    table_in_subset=table_in.merge(ref_table, how = 'inner', on = 'MFR_PART_NO_BY_VENDOR')
    table_in = table_in.merge(table_in.subset, how = 'left',on = 'MFR_PART_NO_BY_VENDOR', indicator = True).query("_merge =='left_only' ")

    return table_in_subset, table_in

def get_p_PSC(table_in, ref_table, PSC):
    ref_table=ref_table[ref_table.PSC == PSC]

    table_in=table_in.merge(ref_table, how = 'left', on ='MFR_PART_NO_BY_VENDOR')

    return table_in


mostPopularMSC = aggDF.sort_values('MSC_COUNT',ascending = False).head(10).drop_duplicates()

dfSample = df.merge(mostPopularMSC, how = 'left',indicator = True, on = ['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned'])
dfSample.columns

distinctMFR = aggDF[['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned']].drop_duplicates().sort_values(['Mfr_Name_By_Vendor_Cleaned','MFR_PART_NO_BY_VENDOR'])
distinctMFR =distinctMFR.merge(distinctMFR, how = 'inner', on = 'Mfr_Name_By_Vendor_Cleaned').query("MFR_PART_NO_BY_VENDOR_x !=MFR_PART_NO_BY_VENDOR_y")

from nltk.metrics import edit_distance   

distinctMFR['edit_distance']=distinctMFR[['MFR_PART_NO_BY_VENDOR_x','MFR_PART_NO_BY_VENDOR_y']].apply(lambda g: edit_distance(s1= g.MFR_PART_NO_BY_VENDOR_x,s2 =g.MFR_PART_NO_BY_VENDOR_y), axis  =1)

distinctMFR.to_pickle(os.path.join(path_in, 'MFR_ANALYSIS.pkl'))

distinctMFR= pd.read_pickle(os.path.join(path_in, 'MFR_ANALYSIS.pkl'))

# distinctMFR.edit_distance.pipe(sns.distplot)
# plt.show()

# we should check to see how close the average prices are for 
# those with edit_distance less than a specific threshold
# similarly. check the max/min PSC counts to understand how closely related these things are/see if they are duplicates
# distinctMFR.query("edit_distance< 2").head()

aggPrice_count = df.groupby(['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned']).\
    agg({"TOTAL_PRICE":'mean'}).\
        rename(columns  = {'TOTAL_PRICE':'AVG_PRICE'})

aggPrice=aggPrice.reset_index()
distinctMFR=distinctMFR.merge(aggPrice_count, how = 'left', left_on = ['MFR_PART_NO_BY_VENDOR_x','Mfr_Name_By_Vendor_Cleaned'],
right_on = ['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned']).rename(columns = {'AVG_PRICE':"AVG_PRICE_x"}).\
        merge(aggPrice_count, how = 'left', left_on = ['MFR_PART_NO_BY_VENDOR_y','Mfr_Name_By_Vendor_Cleaned'],
        right_on = ['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned']).rename(columns = {'AVG_PRICE':"AVG_PRICE_y"})

orderedCols  = ['Unnamed: 0','Mfr_Name_By_Vendor_Cleaned','MFR_PART_NO_BY_VENDOR_x', 'MFR_PART_NO_BY_VENDOR_y','edit_distance','AVG_PRICE_x','AVG_PRICE_y']
distinctMFR=distinctMFR[orderedCols]
# distinctMFR.columns.value_counts()
# aggDF.columns.value_counts()
distinctMFR=distinctMFR.merge(aggDF[['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned','MSC_COUNT']].drop_duplicates(),
how = 'left',
left_on = ['MFR_PART_NO_BY_VENDOR_x','Mfr_Name_By_Vendor_Cleaned'],
right_on = ['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned']).\
    rename(columns = {'MSC_COUNT':'MSC_COUNT_x'})
# distinctMFR.columns.value_counts()
# Jdf= aggDF[['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned','MSC_COUNT']].drop_duplicates()

distinctMFR2 = distinctMFR.merge(Jdf, 
how = 'left', 
left_on = ['MFR_PART_NO_BY_VENDOR_y','Mfr_Name_By_Vendor_Cleaned'],
right_on = ['MFR_PART_NO_BY_VENDOR','Mfr_Name_By_Vendor_Cleaned'])


distinctMFR2=distinctMFR2.rename(columns = {'MSC_COUNT':'MSC_COUNT_y'})
# distinctMFR2.columns
# ['Unnamed: 0', 'Mfr_Name_By_Vendor_Cleaned','MFR_PART_NO_BY_VENDOR_x', 'MFR_PART_NO_BY_VENDOR_y','edit_distance','AVG_PRICE_x', 'AVG_PRICE_y','MFR_PART_NO_BY_VENDOR_x', 'MSC_COUNT_x', 'MFR_PART_NO_BY_VENDOR_y','MSC_COUNT']

distinctMFR2['PRICE_DELTA']=(distinctMFR2.AVG_PRICE_y-distinctMFR2.AVG_PRICE_x)

sns.distplot(distinctMFR2.PRICE_DELTA.dropna())
plt.show()

distinctMFR2.columns


inspectMFR = distinctMFR2[((distinctMFR2.edit_distance<2) & (distinctMFR2.AVG_PRICE_y-distinctMFR2.AVG_PRICE_x <distinctMFR2.AVG_PRICE_y/2)) ]\
    [(distinctMFR2.MSC_COUNT_x>10)|(distinctMFR2.MSC_COUNT_y>10)]

inspectMFR.head()

