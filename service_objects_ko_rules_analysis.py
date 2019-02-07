
################################### We need to fit decision tree  first #####################################

import pandas as pd
import numpy as np

df=pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/3904b1fa-1764-5227-87aa-5c4bf1613e71')
df.head()
df.drop_duplicates('current_loan_leadid', inplace = True)
df = df[  (df['conversion_flag']==2) | (df['conversion_flag']==1) | (df['conversion_flag']==0) ]
#df=df.fillna(df.mean())
import graphviz
from sklearn import tree
#one_hot_data = pd.get_dummies(df[['m_is_mailable','m_is_ported', 'w_is_mailable', 'w_is_ported', 'w_is_wireless','m_is_connected','w_is_connected', 'm_days_of_porting_since20180101', 'w_days_of_porting_since20180101','m_contact_quality_score', 'w_contact_quality_score','m_contact_phone_type', 'w_contact_phone_type']], drop_first=True)
#one_hot_data = pd.get_dummies(df[['m_is_mailable','m_is_ported', 'w_is_mailable', 'w_is_ported', 'w_is_wireless','m_is_connected','w_is_connected', 'm_contact_quality_score', 'w_contact_quality_score','m_contact_phone_type', 'w_contact_phone_type']], drop_first=True)
one_hot_data = pd.get_dummies(df[['m_is_mailable', 'w_is_mailable',
                                  'm_is_ported', 'w_is_ported', 
                                  'w_is_wireless', 'w_is_wireless',
                                  'm_is_connected','w_is_connected',
                                  'm_contact_phone_type', 'w_contact_phone_type', 
                                  'm_is_toll_free_number', 'w_is_toll_free_number',
                                  'm_is_google_voice_number', 'w_is_google_voice_number',
                                  'm_is_possible_disconnected', 'w_is_possible_disconnected',
                                  'm_is_portable_voip', 'w_is_portable_voip',
                                  'm_is_possible_portable_voip', 'w_is_possible_portable_voip',
                                  'm_is_contact_address_po_box', 'w_is_contact_address_po_box' 
                                 ]], drop_first=True)
#one_hot_data = pd.get_dummies(df[['m_days_of_porting_since20180101', 'w_days_of_porting_since20180101']], drop_first=True)
one_hot_data.fillna(one_hot_data.mean())
estimator = tree.DecisionTreeClassifier()
####clf = clf.fit(iris.data, iris.target)
estimator = estimator.fit(one_hot_data, df['conversion_flag'])
#dot_data = tree.export_graphviz(clf, out_file=None, 
#                     feature_names=list(one_hot_data.columns.values),
                     #feature_names=['m_is_mailable','m_is_ported', 'w_is_mailable', 'w_is_ported', 'w_is_wireless','m_is_connected','w_is_connected', 'm_days_of_porting_since20180101', 'w_days_of_porting_since20180101'],           
                     #feature_names=iris.feature_names,  
                     #class_names=iris.target_names,  
#                     filled=True, rounded=True,  
#                     special_characters=True) 
#graph = graphviz.Source(dot_data)
#graph

################# Now we need to traverse fitted decision tree (stored in parallel arrays) #####################
#################       so to compute ratios of converted to diverted in each node  #####################################
################## We store the computed ratios in arrys with the same internal scikit learn node indexing as other ############
################### arrays of the fitted tree object follow ##################################################################


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

############ Now we go over arrays of computed ratios and stored by scikit array of sample sizes
############# to apply a filter : sample size > N (100) ;  the ratio of diverted/converted > M (4)
############## to identify potential knockout nodes
        
for i in range(n_nodes):
    if node_ratio[i]>3 and estimator.tree_.n_node_samples[i]>=200:
        print(i,node_ratio[i], estimator.tree_.n_node_samples[i])
        #print(estimator.tree_.value[i][0])
        
        
#############    We need now 1) to retrieve a path leading to each node from the tree structure
#############                2) to retrieve a rule corresponding to each split

#to retrieve just sequence of parents use :

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

retrieve_path(estimator, 515)

#to retrieve path with the directions left/right (where to go left to the splitted value or right, i.e. smaller or equal to
# splitted value (left) or larger (right) use :

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
        
path_515 = retrieve_directional_path(estimator, 515 )
print('path_515:', path_515)

#mapped_feature = features_map[estimator.tree_.feature[path_6363[5][0]]]
#print( estimator.tree_.feature[path_6363[5][0]], mapped_feature)


# to retrieve path of decisions (rules)  =  potential KO rule use:

features_map = list(one_hot_data.columns.values)

def get_ko_rule(clf, path):
    length = len(path)
    features =[]
    thresholds = []
    decision_list =[]
    for i in range(0,length):
        #features.append(features_map[clf.tree_.feature[path[i]]])
        #thresholds.append(clf.tree_.threshold[path[i]])
        print('\n \n \n ', path)
        mapped_feature = features_map[clf.tree_.feature[path[i][0]]]
        decision_list.append( ( mapped_feature, path[i][1], clf.tree_.threshold[path[i][0]] ) )                                
        #decision_list.append( ( features_map[clf.tree_.feature[path[i]]] , clf.tree_.threshold[path[i]]  )   )
    #return features, thresholds
    return decision_list

features_map = list(one_hot_data.columns.values)
get_ko_rule(estimator, path_515)

#######################################################################################################################
####################### to validate the potential KO rule on validation set use: ######################################
#####################################################################################################################

#get another 10000 loans
df_valid = pd.read_csv('https://app.periscopedata.com/api/creditninja/chart/csv/02f7e29a-03e0-89cb-7d4c-e99f9c2d4336')
df_valid.drop_duplicates('current_loan_leadid', inplace = True)

df_valid=df_valid.fillna(df_valid.mean())


#df_valid[ df_valid['m_is_mailable'==1] ]
n_converted = len(df_valid[ (df_valid['w_is_wireless']==1) &  (df_valid['m_is_connected']==0)
                   & (df_valid['w_is_mailable']==0)  &  (df_valid['w_contact_phone_type']!='RESIDENTIAL') & (df_valid['w_contact_phone_type']!='UNKNOWN') & (df_valid['w_is_ported']==0) &(df_valid['conversion_flag']==1) ]) 



n_diverted = len(df_valid[ (df_valid['w_is_wireless']==1) &  (df_valid['m_is_connected']==0)
                   & (df_valid['w_is_mailable']==0)  &  (df_valid['w_contact_phone_type']!='RESIDENTIAL') & (df_valid['w_contact_phone_type']!='UNKNOWN')    &(df_valid['w_is_ported']==0) &(df_valid['conversion_flag']==0) ]) 



proportion = n_diverted / n_converted


# is proportion bigger than prescribed threshold (=4) ?????????????????????????
##### NO, it is close to 2 (which is population average convergence rate). Overfitting. Knockout rule failed. 
