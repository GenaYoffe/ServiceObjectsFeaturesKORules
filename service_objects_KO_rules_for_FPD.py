
#########################################  STEP 1  ################################################################### 

###  Building decision tree, coming up with nodes which are candidates for knockouts
###  This step will work for all introduced SO features. But step 3 (validation) will work only with
###  categorical and dummy vars. So if we want to validate the results, lets use only indicators or
###  categorical vars.


import pandas as pd
import numpy as np


###########################    Training the model   #######################################################


df=pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/3904b1fa-1764-5227-87aa-5c4bf1613e71')
df.drop_duplicates('current_loan_leadid', inplace = True)
df = df[(df['loan_approved_flag']==1) & ((df['fpd_flag']==0) | (df['fpd_flag']==1))]
#df=df.fillna(df.mean())
df['fpd_flag'].fillna(df['fpd_flag'].mean())
df['fpd_flag'] = df['fpd_flag'].astype('bool')

from sklearn import tree


# SO does not provide address and name data on more than 50% of customers
# at the same time when it is provided this information is useful
# We want to introduce variables which will indicate when this information is missing,
# since, in the matching address and name variables, which we introduce, the imputed for NA values cannot be 
# trusted much. Hopefully those indicators help resolving this problem. 
# There might be a better way of dealing with bigger amount of NA vlues
df['m_contact_address_na'] = pd.isna(df['mob_contactaddressout']) 
df['w_contact_address_na'] = pd.isna(df['work_contactaddressout'])
df['m_contact_name_na'] = pd.isna(df['m_last_name_match']) 
df['w_contact_name_na'] = pd.isna(df['w_last_name_match'])
#convert columns from boolean to numeric
df['m_contact_address_na'] *= 1
df['w_contact_address_na'] *= 1
df['m_contact_name_na'] *= 1
df['w_contact_name_na'] *= 1


#one_hot_data = pd.get_dummies(df[['m_is_mailable', 'w_is_mailable',
#                                 'm_is_wireless', 'w_is_wireless',
#                                 'm_is_ported', 'w_is_ported',                                   
#                                 'm_is_connected','w_is_connected',
#                                 'm_contact_phone_type', 'w_contact_phone_type', 
#                                 'm_is_toll_free_number', 'w_is_toll_free_number',
#                                 'm_is_google_voice_number', 'w_is_google_voice_number',
#                                 'm_is_possible_disconnected', 'w_is_possible_disconnected',
#                                 'm_is_portable_voip', 'w_is_portable_voip',
#                                 'm_is_possible_portable_voip', 'w_is_possible_portable_voip',
#                                 'm_is_contact_address_po_box', 'w_is_contact_address_po_box',
#                                 'm_contact_address_match_lev8', 'w_contact_address_match_lev8', 
#                                 'm_contact_address_na', 'w_contact_address_na',
#                                 'm_last_name_match', 'w_last_name_match',
#                                 'm_contact_name_na', 'w_contact_name_na',
#                                 'm_days_of_porting_since20180101', 'w_days_of_porting_since20180101',
#                                 'm_pn2', 'w_pn2',
#                                 'm_zip_sq_distance', 'w_zip_sq_distance'                                  
#                                 ]], drop_first=True)


categorical_vars_to_model = ['m_is_mailable', 'w_is_mailable', 
                             'm_is_wireless', 'w_is_wireless',
                             'm_is_ported', 'w_is_ported',
                             'm_is_toll_free_number', 'w_is_toll_free_number',
                             'm_is_connected', 'w_is_connected',
                             'm_is_unknown_contact', 'w_is_unknown_contact',
                             'm_contact_phone_type', 'w_contact_phone_type',
                             'm_contact_name_na', 'w_contact_name_na',
                             'm_last_name_match', 'w_last_name_match'
                             ]


one_hot_data = pd.get_dummies(df[categorical_vars_to_model], drop_first=True)


######  df=df.fillna(df.mean()) not always works for the entire dataframe (just leaves NA values in the dataset)
######  in the worst case let's do it column by column

one_hot_data['m_is_mailable'].fillna(one_hot_data['m_is_mailable'].mean(), inplace = True)
one_hot_data['w_is_mailable'].fillna(one_hot_data['w_is_mailable'].mean(), inplace = True)
one_hot_data['m_is_wireless'].fillna(one_hot_data['m_is_wireless'].mean(), inplace = True)
one_hot_data['w_is_wireless'].fillna(one_hot_data['w_is_wireless'].mean(), inplace = True)

one_hot_data['m_last_name_match'].fillna(one_hot_data['m_last_name_match'].mean(), inplace = True)
one_hot_data['w_last_name_match'].fillna(one_hot_data['m_last_name_match'].mean(), inplace = True)

#one_hot_data.fillna(one_hot_data.mean(), inplace=True)

estimator = tree.DecisionTreeClassifier()
estimator = estimator.fit(one_hot_data, df['fpd_flag'])


########## We need to traverse the fitted tree, so to compute FPD ratios at each node ###########

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
node_ratio = np.zeros(shape=n_nodes, dtype=np.float)

stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1
    if estimator.tree_.value[node_id][0][1] != 0:
        node_ratio[node_id] = estimator.tree_.value[node_id][0][0]/estimator.tree_.value[node_id][0][1]
    else:
        node_ratio[node_id] = 1000
        
    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

        
'''
print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
'''

########################## Coming up with potential candidates for knockouts #####################################        
        
print("Average FPD ratio is around 4:1") 

print("Nodes with FPD ratio lower than 3:1, having 100 or more loan applications in them - \n potential candidates for knockouts")        
print('node #\t', 'fpd ratio \t', 'number of loans')        
for i in range(n_nodes):
    if node_ratio[i]<=3 and estimator.tree_.n_node_samples[i]>=100:
        print(i, '\t', node_ratio[i], '\t', estimator.tree_.n_node_samples[i])
        #print(estimator.tree_.value[i][0])

        
print('\n \n \n')

print("Nodes with FPD ratio higher than 5:1, having 100 or more loan applications in them - \n potential candidates with more favorable purchase procedure, \n i.e. skipping purchasing some reports")
print('node #\t', 'fpd ratio \t', 'number of loans')    
for i in range(n_nodes):
    if node_ratio[i]>5  and estimator.tree_.n_node_samples[i]>=100:
        print(i,'\t', node_ratio[i],'\t', estimator.tree_.n_node_samples[i])
        #print(estimator.tree_.value[i][0])        



################## STEP 2: RETRIEVING THE PATH TO THE NODE IN THE TREE, WHICH REPRESENTS POTENTIAL KNOCKOUT RULE ##############



# The nodes of the tree are identified by an internal integer id. In the previos step we got a list of
#  id candidates, which 

ko_path = 150 # this is actually a candidate for a favorite bying, not for KO

def retrieve_path(clf, node):
    parent=node
    path=[parent]
    while parent>0:
        if parent in clf.tree_.children_right:
            parent = list(clf.tree_.children_right).index(parent)
        if parent in clf.tree_.children_left:
            parent = list(clf.tree_.children_left).index(parent)
        path.append(parent)
    return path[::-1]

retrieve_path(estimator, ko_path)

# we need to know not only the sequence of ancestors, but also direction used in splitting
# at each step leading to the node in question
def retrieve_directional_path(clf, node):
    parent=node
    parent_direct = (node, -1)
    path=[parent_direct]
    while parent>0:
        if parent in clf.tree_.children_right:
            parent = list(clf.tree_.children_right).index(parent)
            parent_direct = (parent, False)
        if parent in clf.tree_.children_left:
            parent = list(clf.tree_.children_left).index(parent)
            parent_direct = (parent, True )
        path.append(parent_direct)
    return path[::-1]
        
directioanal_path = retrieve_directional_path(estimator, ko_path)
print('directioanal_path: ', directioanal_path)

#mapped_feature = features_map[estimator.tree_.feature[path_6363[5][0]]]
#print( estimator.tree_.feature[path_6363[5][0]], mapped_feature)

### building KO rule from the directioanal path 
def get_ko_rule(clf, path):
    length = len(path)
    features =[]
    thresholds = []
    decision_list =[]
    for i in range(0,length):
        #features.append(features_map[clf.tree_.feature[path[i]]])
        #thresholds.append(clf.tree_.threshold[path[i]])
        #print('\n \n \n ', path)
        mapped_feature = features_map[clf.tree_.feature[path[i][0]]]
        decision_list.append( ( mapped_feature, path[i][1], clf.tree_.threshold[path[i][0]] ) )                                
        #decision_list.append( ( features_map[clf.tree_.feature[path[i]]] , clf.tree_.threshold[path[i]]  )   )
    #return features, thresholds
    return decision_list

features_map = list(one_hot_data.columns.values)
print("The candidate for KO rule is: ",  get_ko_rule(estimator, directioanal_path) )

decision_list = get_ko_rule(estimator, directioanal_path)


# presenting KO rule in the format used in validation. 
# Note: Only up to 10 indicators can appear in the implemented at step 3 validation !!!!!

def ko_rule_validation_format(decision_list):
    decision_list_validation_format = []
    for i in range(0, len(decision_list) - 1 ):
        rule = decision_list[i][0]
        value = 1 - int(decision_list[i][1])
        decision_list_validation_format += [ (rule, value) ] 
    for i in range(len(decision_list) -1 , 10 ):
        decision_list_validation_format += [ (rule, value) ]         
    return decision_list_validation_format


# EXAMPLE OF NOTATION IN THE DECISION TREE:  Rule ('w_is_toll_free_number', True, 0.5) means that 
# the value of the indicator w_is_toll_free_number is less than 0.5, means it is equal to zero, i.e. that
# the given phone is not toll free

# Note: the last node in the path is not used in the decision




##########################      STEP 3;    Validation step    ##########################################################

# It is done in 3 portions strictly for technical reasons (data coming as output of Periscope charts, and 
# Periscope chart is not able to put into CSV more than 10,000 rows with the given amount of columns

# The proportons of data used for model training adnd validation might be reconsidered
# I would keep the proportions about the same though (only quarter of data is used for training )

df_valid = pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/02f7e29a-03e0-89cb-7d4c-e99f9c2d4336')
df_valid.drop_duplicates('current_loan_leadid', inplace = True)
df_valid = df_valid[(df_valid['loan_approved_flag']==1) & ((df_valid['fpd_flag']==0) | (df_valid['fpd_flag']==1))]
df_valid_fpd = df_valid['fpd_flag']

# We build the additional indicators used in the model

df_valid['m_contact_address_na'] = pd.isna(df_valid['mob_contactaddressout']) 
df_valid['w_contact_address_na'] = pd.isna(df_valid['work_contactaddressout'])
df_valid['m_contact_name_na'] = pd.isna(df_valid['m_last_name_match']) 
df_valid['w_contact_name_na'] = pd.isna(df_valid['w_last_name_match'])


#convert columns from boolean to numeric
df_valid['m_contact_address_na'] *= 1
df_valid['w_contact_address_na'] *= 1
df_valid['m_contact_name_na'] *= 1
df_valid['w_contact_name_na'] *= 1    

# We split categorical to dummies ( which is done in the model)

df_valid = pd.get_dummies(df_valid[categorical_vars_to_model], drop_first=True)
df_valid['fpd_flag'] = df_valid_fpd

# some values of splitted categorical variables can be missing in the validation dataset,
# the corresponding to those values dummies were not added to the validation dataset. We add them here
list_of_missing_columns = list(set(list(one_hot_data.columns.values)) - set(list(df_valid.columns.values)))
for i in range(0, len(list_of_missing_columns)):
    df_valid[list_of_missing_columns[i]]=0
    


# df_valid=df_valid.fillna(df_valid.mean())
# df_valid[ df_valid['m_is_mailable'==1] ]


decision_list_vf = ko_rule_validation_format(decision_list)
#print( '\n length:', len(df_valid  [ df_valid [decision_list_vf[0][0] ]==decision_list_vf[0][1] ]) )


number_of_fpds = len(df_valid[ (df_valid[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid['fpd_flag']==1) ])

print( 'The number of fpds: ', number_of_fpds)


number_of_good_loans = len(df_valid[ (df_valid[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid['fpd_flag']==0) ])


print( 'The number of good loans: ', number_of_good_loans)

print('validation proportion :', number_of_good_loans /number_of_fpds)


df_valid2 = pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/c734adbd-5c91-bf51-b1d1-2c94a4c48ad2')
df_valid2.drop_duplicates('current_loan_leadid', inplace = True)
df_valid2 = df_valid2[(df_valid2['loan_approved_flag']==1) & ((df_valid2['fpd_flag']==0) | (df_valid2['fpd_flag']==1))]
df_valid2_fpd = df_valid2['fpd_flag']

df_valid2['m_contact_address_na'] = pd.isna(df_valid2['mob_contactaddressout']) 
df_valid2['w_contact_address_na'] = pd.isna(df_valid2['work_contactaddressout'])
df_valid2['m_contact_name_na'] = pd.isna(df_valid2['m_last_name_match']) 
df_valid2['w_contact_name_na'] = pd.isna(df_valid2['w_last_name_match'])


#convert columns from boolean to numeric
df_valid2['m_contact_address_na'] *= 1
df_valid2['w_contact_address_na'] *= 1
df_valid2['m_contact_name_na'] *= 1
df_valid2['w_contact_name_na'] *= 1    


df_valid2 = pd.get_dummies(df_valid2[categorical_vars_to_model], drop_first=True)
df_valid2['fpd_flag'] = df_valid2_fpd
list_of_missing_columns2 = list(set(list(one_hot_data.columns.values)) - set(list(df_valid2.columns.values)))
for i in range(0, len(list_of_missing_columns2)):
    df_valid2[list_of_missing_columns2[i]]=0


number_of_fpds2 = len(df_valid2[ (df_valid2[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid2[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid2[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid2[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid2[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid2[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid2[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid2[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid2[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid2[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid2['fpd_flag']==1) ])

print( 'The number of fpds2: ', number_of_fpds2)


number_of_good_loans2 = len(df_valid2[ (df_valid2[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid2[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid2[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid2[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid2[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid2[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid2[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid2[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid2[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid2[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid2['fpd_flag']==0) ])


print( 'The number of good loans2: ', number_of_good_loans2)     

 

print('validation proportion 2:', number_of_good_loans2/number_of_fpds2)


df_valid3 = pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/c271c9d9-7216-6aba-b569-5e61c31a0062')
df_valid3.drop_duplicates('current_loan_leadid', inplace = True)
df_valid3 = df_valid3[(df_valid3['loan_approved_flag']==1) & ((df_valid3['fpd_flag']==0) | (df_valid3['fpd_flag']==1))]
df_valid3_fpd = df_valid3['fpd_flag']

df_valid3['m_contact_address_na'] = pd.isna(df_valid3['mob_contactaddressout']) 
df_valid3['w_contact_address_na'] = pd.isna(df_valid3['work_contactaddressout'])
df_valid3['m_contact_name_na'] = pd.isna(df_valid3['m_last_name_match']) 
df_valid3['w_contact_name_na'] = pd.isna(df_valid3['w_last_name_match'])


#convert columns from boolean to numeric
df_valid3['m_contact_address_na'] *= 1
df_valid3['w_contact_address_na'] *= 1
df_valid3['m_contact_name_na'] *= 1
df_valid3['w_contact_name_na'] *= 1    


df_valid3 = pd.get_dummies(df_valid3[categorical_vars_to_model], drop_first=True)
df_valid3['fpd_flag'] = df_valid3_fpd
list_of_missing_columns3 = list(set(list(one_hot_data.columns.values)) - set(list(df_valid3.columns.values)))
for i in range(0, len(list_of_missing_columns3)):
    df_valid3[list_of_missing_columns3[i]]=0



number_of_fpds3 = len(df_valid3[ (df_valid3[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid3[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid3[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid3[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid3[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid3[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid3[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid3[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid3[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid3[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid3['fpd_flag']==1) ])


print( 'The number of fpds3: ', number_of_fpds3)


number_of_good_loans3 = len(df_valid3[ (df_valid3[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid3[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid3[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid3[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid3[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid3[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid3[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid3[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid3[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid3[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid3['fpd_flag']==0) ])


print( 'The number of good loans3: ', number_of_good_loans3)     

 

print('validation proportion 3:', number_of_good_loans3/number_of_fpds3)














































import pandas as pd
import numpy as np

df=pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/3904b1fa-1764-5227-87aa-5c4bf1613e71')
df.head()
df.drop_duplicates('current_loan_leadid', inplace = True)
#df = df[  (df['conversion_flag']==2) | (df['conversion_flag']==1) | (df['conversion_flag']==0) ]
df = df[(df['loan_approved_flag']==1) & ((df['fpd_flag']==0) | (df['fpd_flag']==1))]
df=df.fillna(df.mean())
df['fpd_flag'] = df['fpd_flag'].astype('bool')
import graphviz
from sklearn import tree


df['m_contact_address_na'] = pd.isna(df['mob_contactaddressout']) 
df['w_contact_address_na'] = pd.isna(df['work_contactaddressout'])
df['m_contact_name_na'] = pd.isna(df['m_last_name_match']) 
df['w_contact_name_na'] = pd.isna(df['w_last_name_match'])


#convert columns from boolean to numeric
df['m_contact_address_na'] *= 1
df['w_contact_address_na'] *= 1
df['m_contact_name_na'] *= 1
df['w_contact_name_na'] *= 1


#one_hot_data = pd.get_dummies(df[['m_is_mailable','m_is_ported', 'w_is_mailable', 'w_is_ported', 'w_is_wireless','m_is_connected','w_is_connected', 'm_days_of_porting_since20180101', 'w_days_of_porting_since20180101','m_contact_quality_score', 'w_contact_quality_score','m_contact_phone_type', 'w_contact_phone_type']], drop_first=True)
#one_hot_data = pd.get_dummies(df[['m_is_mailable','m_is_ported', 'w_is_mailable', 'w_is_ported', 'w_is_wireless','m_is_connected','w_is_connected', 'm_contact_quality_score', 'w_contact_quality_score','m_contact_phone_type', 'w_contact_phone_type']], drop_first=True)

#one_hot_data = pd.get_dummies(df[['m_is_mailable', 'w_is_mailable',
#                                 'm_is_wireless', 'w_is_wireless',
#                                 'm_is_ported', 'w_is_ported',                                   
#                                 'm_is_connected','w_is_connected',
#                                 'm_contact_phone_type', 'w_contact_phone_type', 
#                                 'm_is_toll_free_number', 'w_is_toll_free_number',
#                                 'm_is_google_voice_number', 'w_is_google_voice_number',
#                                 'm_is_possible_disconnected', 'w_is_possible_disconnected',
#                                 'm_is_portable_voip', 'w_is_portable_voip',
#                                 'm_is_possible_portable_voip', 'w_is_possible_portable_voip',
#                                 'm_is_contact_address_po_box', 'w_is_contact_address_po_box',
#                                 'm_contact_address_match_lev8', 'w_contact_address_match_lev8', 
#                                 'm_contact_address_na', 'w_contact_address_na',
#                                 'm_last_name_match', 'w_last_name_match',
#                                 'm_contact_name_na', 'w_contact_name_na',
#                                 'm_days_of_porting_since20180101', 'w_days_of_porting_since20180101',
#                                 'm_pn2', 'w_pn2',
#                                 'm_zip_sq_distance', 'w_zip_sq_distance'                                  
#                                 ]], drop_first=True)


categorical_vars_to_model = ['m_is_mailable', 'w_is_mailable', 
                                  'm_is_wireless', 'w_is_wireless',
                                   'm_is_ported', 'w_is_ported',
                                   'm_is_toll_free_number', 'w_is_toll_free_number',
                                   'm_is_connected', 'w_is_connected',
                                   'm_is_unknown_contact', 'w_is_unknown_contact',
                                    'm_contact_phone_type', 'w_contact_phone_type',
                             ]

one_hot_data = pd.get_dummies(df[categorical_vars_to_model], drop_first=True)

#one_hot_data = pd.get_dummies(df[['m_is_mailable', 'w_is_mailable', 
#                                  'm_is_wireless', 'w_is_wireless',
#                                   'm_is_ported', 'w_is_ported',
#                                   'm_is_toll_free_number', 'w_is_toll_free_number',
#                                   'm_is_connected', 'w_is_connected',
#                                   'm_is_unknown_contact', 'w_is_unknown_contact',
#                                   'm_last_name_match', 'w_last_name_match',
#                                   'm_contact_name_na', 'w_contact_name_na',
#                                   'm_pn2', 'w_pn2'
#                                 ]]
#                                   , drop_first=True)

#one_hot_data['w_last_name_match'].fillna(one_hot_data['w_last_name_match'].mean())


one_hot_data['m_is_mailable'].fillna(one_hot_data['m_is_mailable'].mean())
one_hot_data['w_is_mailable'].fillna(one_hot_data['w_is_mailable'].mean())
one_hot_data['m_is_wireless'].fillna(one_hot_data['m_is_wireless'].mean())
one_hot_data['w_is_wireless'].fillna(one_hot_data['w_is_wireless'].mean())
#df['fpd_flag'].fillna(df['fpd_flag'].mean())


#one_hot_data.fillna(one_hot_data.mean(), inplace=True)
estimator = tree.DecisionTreeClassifier()
####clf = clf.fit(iris.data, iris.target)
estimator = estimator.fit(one_hot_data, df['fpd_flag'])
#dot_data = tree.export_graphviz(clf, out_file=None, 
#                     feature_names=list(one_hot_data.columns.values),
                     #feature_names=['m_is_mailable','m_is_ported', 'w_is_mailable', 'w_is_ported', 'w_is_wireless','m_is_connected','w_is_connected', 'm_days_of_porting_since20180101', 'w_days_of_porting_since20180101'],           
                     #feature_names=iris.feature_names,  
                     #class_names=iris.target_names,  
#                     filled=True, rounded=True,  
#                     special_characters=True) 
#graph = graphviz.Source(dot_data)
#graph

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
node_ratio = np.zeros(shape=n_nodes, dtype=np.float)

stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1
    if estimator.tree_.value[node_id][0][1] != 0:
        node_ratio[node_id] = estimator.tree_.value[node_id][0][0]/estimator.tree_.value[node_id][0][1]
    else:
        node_ratio[node_id] = 1000
        
    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))

print("Average FPD ratio is around 4:1") 

print("Nodes with FPD ratio lower than 3:1, having 100 or more loan applications in them - \n potential candidates for knockouts")        
print('node #\t', 'fpd ratio \t', 'number of loans')        
for i in range(n_nodes):
    if node_ratio[i]<=3 and estimator.tree_.n_node_samples[i]>=100:
        print(i, '\t', node_ratio[i], '\t', estimator.tree_.n_node_samples[i])
        #print(estimator.tree_.value[i][0])

        
print('\n \n \n')

print("Nodes with FPD ratio higher than 5:1, having 100 or more loan applications in them - \n potential candidates with more favorable purchase procedure, \n i.e. skipping purchasing some reports")
print('node #\t', 'fpd ratio \t', 'number of loans')    
for i in range(n_nodes):
    if node_ratio[i]>5  and estimator.tree_.n_node_samples[i]>=100:
        print(i,'\t', node_ratio[i],'\t', estimator.tree_.n_node_samples[i])
        #print(estimator.tree_.value[i][0])        


ko_path = 403

def retrieve_path(clf, node):
    parent=node
    path=[parent]
    while parent>0:
        if parent in clf.tree_.children_right:
            parent = list(clf.tree_.children_right).index(parent)
        if parent in clf.tree_.children_left:
            parent = list(clf.tree_.children_left).index(parent)
        path.append(parent)
    return path[::-1]

retrieve_path(estimator, ko_path)


def retrieve_directional_path(clf, node):
    parent=node
    parent_direct = (node, -1)
    path=[parent_direct]
    while parent>0:
        if parent in clf.tree_.children_right:
            parent = list(clf.tree_.children_right).index(parent)
            parent_direct = (parent, False)
        if parent in clf.tree_.children_left:
            parent = list(clf.tree_.children_left).index(parent)
            parent_direct = (parent, True )
        path.append(parent_direct)
    return path[::-1]
        
directioanal_path = retrieve_directional_path(estimator, ko_path)
print('directioanal_path: ', directioanal_path)

#mapped_feature = features_map[estimator.tree_.feature[path_6363[5][0]]]
#print( estimator.tree_.feature[path_6363[5][0]], mapped_feature)


def get_ko_rule(clf, path):
    length = len(path)
    features =[]
    thresholds = []
    decision_list =[]
    for i in range(0,length):
        #features.append(features_map[clf.tree_.feature[path[i]]])
        #thresholds.append(clf.tree_.threshold[path[i]])
        #print('\n \n \n ', path)
        mapped_feature = features_map[clf.tree_.feature[path[i][0]]]
        decision_list.append( ( mapped_feature, path[i][1], clf.tree_.threshold[path[i][0]] ) )                                
        #decision_list.append( ( features_map[clf.tree_.feature[path[i]]] , clf.tree_.threshold[path[i]]  )   )
    #return features, thresholds
    return decision_list

features_map = list(one_hot_data.columns.values)
get_ko_rule(estimator, directioanal_path)




decision_list = get_ko_rule(estimator, directioanal_path)

def ko_rule_validation_format(decision_list):
    decision_list_validation_format = []
    for i in range(0, len(decision_list) - 1 ):
        rule = decision_list[i][0]
        value = 1 - int(decision_list[i][1])
        decision_list_validation_format += [ (rule, value) ] 
    for i in range(len(decision_list) -1 , 10 ):
        decision_list_validation_format += [ (rule, value) ]         
    return decision_list_validation_format

print(decision_list)
print(ko_rule_validation_format(decision_list))


df_valid = pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/02f7e29a-03e0-89cb-7d4c-e99f9c2d4336')
df_valid.drop_duplicates('current_loan_leadid', inplace = True)
df_valid = df_valid[(df_valid['loan_approved_flag']==1) & ((df_valid['fpd_flag']==0) | (df_valid['fpd_flag']==1))]
df_valid_fpd = df_valid['fpd_flag']
df_valid = pd.get_dummies(df_valid[categorical_vars_to_model], drop_first=True)
df_valid['fpd_flag'] = df_valid_fpd
list_of_missing_columns = list(set(list(one_hot_data.columns.values)) - set(list(df_valid.columns.values)))
for i in range(0, len(list_of_missing_columns)):
    df_valid[list_of_missing_columns[i]]=0

# df_valid=df_valid.fillna(df_valid.mean())
# df_valid[ df_valid['m_is_mailable'==1] ]


decision_list_vf = ko_rule_validation_format(decision_list)
#print( '\n length:', len(df_valid  [ df_valid [decision_list_vf[0][0] ]==decision_list_vf[0][1] ]) )


number_of_fpds = len(df_valid[ (df_valid[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid['fpd_flag']==1) ])

print( 'The number of fpds: ', number_of_fpds)


number_of_good_loans = len(df_valid[ (df_valid[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid['fpd_flag']==0) ])


print( 'The number of good loans: ', number_of_good_loans)

print('validation proportion :', number_of_good_loans /number_of_fpds)

df_valid2 = pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/c734adbd-5c91-bf51-b1d1-2c94a4c48ad2')
df_valid2.drop_duplicates('current_loan_leadid', inplace = True)
df_valid2 = df_valid2[(df_valid2['loan_approved_flag']==1) & ((df_valid2['fpd_flag']==0) | (df_valid2['fpd_flag']==1))]
df_valid2_fpd = df_valid2['fpd_flag']
df_valid2 = pd.get_dummies(df_valid2[categorical_vars_to_model], drop_first=True)
df_valid2['fpd_flag'] = df_valid2_fpd
list_of_missing_columns2 = list(set(list(one_hot_data.columns.values)) - set(list(df_valid2.columns.values)))
for i in range(0, len(list_of_missing_columns2)):
    df_valid2[list_of_missing_columns2[i]]=0


number_of_fpds2 = len(df_valid2[ (df_valid2[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid2[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid2[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid2[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid2[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid2[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid2[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid2[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid2[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid2[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid2['fpd_flag']==1) ])

print( 'The number of fpds2: ', number_of_fpds2)


number_of_good_loans2 = len(df_valid2[ (df_valid2[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid2[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid2[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid2[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid2[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid2[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid2[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid2[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid2[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid2[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid2['fpd_flag']==0) ])


print( 'The number of good loans2: ', number_of_good_loans2)     

 

print('validation proportion 2:', number_of_good_loans2/number_of_fpds2)


df_valid3 = pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/c271c9d9-7216-6aba-b569-5e61c31a0062')
df_valid3.drop_duplicates('current_loan_leadid', inplace = True)
df_valid3 = df_valid3[(df_valid3['loan_approved_flag']==1) & ((df_valid3['fpd_flag']==0) | (df_valid3['fpd_flag']==1))]
df_valid3_fpd = df_valid3['fpd_flag']
df_valid3 = pd.get_dummies(df_valid3[categorical_vars_to_model], drop_first=True)
df_valid3['fpd_flag'] = df_valid3_fpd
list_of_missing_columns3 = list(set(list(one_hot_data.columns.values)) - set(list(df_valid3.columns.values)))
for i in range(0, len(list_of_missing_columns3)):
    df_valid3[list_of_missing_columns3[i]]=0



number_of_fpds3 = len(df_valid3[ (df_valid3[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid3[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid3[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid3[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid3[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid3[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid3[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid3[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid3[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid3[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid3['fpd_flag']==1) ])


print( 'The number of fpds3: ', number_of_fpds3)


number_of_good_loans3 = len(df_valid3[ (df_valid3[decision_list_vf[0][0]]==decision_list_vf[0][1]) &  (df_valid3[decision_list_vf[1][0]]==decision_list_vf[1][1]  )
                   & (df_valid3[decision_list_vf[2][0]]==decision_list_vf[2][1])  &  (df_valid3[decision_list_vf[3][0]]==decision_list_vf[3][1])
                   & (df_valid3[decision_list_vf[4][0]]==decision_list_vf[4][1]) & (df_valid3[decision_list_vf[5][0]]==decision_list_vf[5][1])
                   & (df_valid3[decision_list_vf[6][0]]==decision_list_vf[6][1]) &  (df_valid3[decision_list_vf[7][0]]==decision_list_vf[7][1])
                   & (df_valid3[decision_list_vf[8][0]]==decision_list_vf[8][1])  &  (df_valid3[decision_list_vf[9][0]]==decision_list_vf[9][1])                   
                   & (df_valid3['fpd_flag']==0) ])


print( 'The number of good loans3: ', number_of_good_loans3)     

 

print('validation proportion 3:', number_of_good_loans3/number_of_fpds3)





