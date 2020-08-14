import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")

#load data
project_data = pd.read_csv('train.csv')
resources_data = pd.read_csv('resources.csv')

#make total resource cost column
resources_data['total_resource_cost'] = resources_data['quantity'] * resources_data['price']
#combine columns together
combined_df = resources_data.groupby(['id']).sum().reset_index()
#join this data set by id to the other data set
combined_df = combined_df.join(project_data.set_index('id'), on='id')
combined_df.head()

#cleaning:
#get rid of outliers method:
def getBoundsForOutliers(column):
  qOne = column.quantile(0.25)
  median = column.quantile(0.5)
  qThree = column.quantile(0.75)
  iqr = qThree - qOne
  lowerBound = qOne - 1.5*iqr
  upperBound = qThree + 1.5*iqr
  return lowerBound, upperBound

#data cleanup - NAN's
#1) remove all data w/ NaN in quantity, price, total resource cost, teacher id, prefix, state... anything that isnt essays 3 or 4
# - same as delete if more than 2 cols have missing stuff
thresholdValue = len(combined_df.columns) - 2
combined_df.dropna(thresh=thresholdValue).head()
#remove outliers in cols price, quantity, total resource cost, number of previously posted projects
#quantity
#remove outliers related to essay separately (have to separate depending on # essays)

def removeOutliersFromDf(dataframe, col_names):
  for name in col_names:
    bounds = getBoundsForOutliers(dataframe[name])
    dataframe = dataframe[bounds[0] <= dataframe[name]]
    dataframe = dataframe[bounds[1] >= dataframe[name]]
  return dataframe

#all the cols we want to go thru
col_names = ['quantity', 'price', 'total_resource_cost', 'teacher_number_of_previously_posted_projects']
for name in col_names:
  bounds = getBoundsForOutliers(combined_df[name])
  combined_df = combined_df[bounds[0] <= combined_df[name]]
  combined_df = combined_df[bounds[1] >= combined_df[name]]

#get the number of essays
def getNumberEssays(row):
  #return number of not NAN entries for the 4 essay cols
  numEssays = 0
  if not pd.isna(row['project_essay_1']):
    numEssays+=1
  if not pd.isna(row['project_essay_2']):
    numEssays+=1
  if not pd.isna(row['project_essay_3']):
    numEssays+=1
  if not pd.isna(row['project_essay_4']):
    numEssays+=1
  return numEssays
numEssays = combined_df.apply(getNumberEssays, axis=1)
numEssays.head()

#replace the nan w/ empty strings
combined_df['project_essay_3'] = combined_df['project_essay_3'].fillna("")
combined_df['project_essay_4'] = combined_df['project_essay_4'].fillna("")
#get data w/ only 2 essays vs data w/ > 2 essays
combined_df['number_essays'] = numEssays
#add essay length columns
combined_df['essay_1_length'] = combined_df['project_essay_1'].str.len()
combined_df['essay_2_length'] = combined_df['project_essay_2'].str.len()
combined_df['essay_3_length'] = combined_df['project_essay_3'].str.len()
combined_df['essay_4_length'] = combined_df['project_essay_4'].str.len()
#split data
two_essays = combined_df[combined_df['number_essays'] == 2]
four_essays = combined_df[combined_df['number_essays'] == 4]
#remove outliers in 2 essays & four essays separately.
# two essays:
#all the cols we want to go thru
col_names = ['essay_1_length', 'essay_2_length']
two_essays = removeOutliersFromDf(two_essays, col_names)
# 4 essays:
#all cols we want to go thru
col_names = ['essay_1_length', 'essay_2_length', 'essay_3_length', 'essay_4_length']
four_essays = removeOutliersFromDf(four_essays, col_names)
#combine the two essays and four essays back together.
combined_df = two_essays.append(four_essays, sort=False)
#finished cleaning
#SOMEHOW THERE R STILL NAN Values
combined_df = combined_df.dropna()
combined_df.head()

#we dont care abt teacher id, project resource summary, title or any of the essays.
combined_df = combined_df.drop(columns=['teacher_id', 'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_resource_summary'])
#replace teacher prefix, id, school state, project grade category, and project subject category w/ one hot encoded version
#51 unique values in both subject cats and school state, 393 subcats so i shouldnt OH encode that
categorical_variables = ['teacher_prefix', 'project_grade_category', 'project_subject_categories', 'school_state']
for col_name in categorical_variables:
  combined_df = pd.concat([combined_df, pd.get_dummies(combined_df[col_name])], axis=1)
#*DONT* remove the old columns
#practice_df = practice_df.drop(columns=col_names)
combined_df.head()

#make a balanced df from the combined df.
#split data
not_approved_df = combined_df[combined_df['project_is_approved'] == 0]
approved_df = combined_df[combined_df['project_is_approved'] == 1].sample(len(not_approved_df))
#now combine them
balanced_combined_df = approved_df.append(not_approved_df, sort=False)

#split into training and validation sets.
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(balanced_combined_df, test_size=0.2)

#try to make an unbalanced validation set.
unbalanced_valid_df = combined_df.sample(round(0.2*len(combined_df)))
unbalanced_valid_df['project_is_approved'].describe()

valid_df['project_is_approved'].describe()

train_df['project_is_approved'].describe()

def getTrainValidAccErr(minSamples, maxDepth, training_size):
  #all imports:
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import mean_squared_error
  from sklearn.metrics import accuracy_score

  #for predictor variables first put in all them numerical data
  explanatory_variables = ['total_resource_cost', 'teacher_number_of_previously_posted_projects', 'number_essays',
                           'essay_1_length', 'essay_2_length', 'essay_3_length', 'essay_4_length']
  #now add in all the categorical data column names to the list
  for name in categorical_variables:
    explanatory_variables = explanatory_variables + combined_df[name].unique().tolist()

  #training set
  y = train_df.project_is_approved[0:training_size]
  X = train_df[explanatory_variables][0:training_size]
  project_model = DecisionTreeClassifier(min_samples_leaf=minSamples, random_state=1, max_depth=maxDepth)
  project_model.fit(X, y)
  predicted_approval = project_model.predict(X)
  trainingError = mean_squared_error(y, predicted_approval)
  trainingAccuracy = accuracy_score(y, predicted_approval)
  #check w/ validation model.
  X_valid = valid_df[explanatory_variables]
  y_valid = valid_df.project_is_approved
  predicted_approval = project_model.predict(X_valid)
  validationError = mean_squared_error(y_valid, predicted_approval)
  validationAccuracy = accuracy_score(y_valid, predicted_approval)
  #check w/unbalanced valid
  X_unbal_valid = unbalanced_valid_df[explanatory_variables]
  y_unbal_valid = unbalanced_valid_df.project_is_approved
  predicted_approval = project_model.predict(X_unbal_valid)
  unbalValidAccuracy = accuracy_score(y_unbal_valid, predicted_approval)

  return trainingError, trainingAccuracy, validationError, validationAccuracy, unbalValidAccuracy

def getF1_score(minSamples, maxDepth, training_size):
  #all imports:
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import f1_score

 #for predictor variables first put in all them numerical data
  explanatory_variables = ['total_resource_cost', 'teacher_number_of_previously_posted_projects', 'number_essays',
                           'essay_1_length', 'essay_2_length', 'essay_3_length', 'essay_4_length']
  #now add in all the categorical data column names to the list
  for name in categorical_variables:
    explanatory_variables = explanatory_variables + combined_df[name].unique().tolist()

  #training set
  y = train_df.project_is_approved[0:training_size]
  X = train_df[explanatory_variables][0:training_size]
  project_model = DecisionTreeClassifier(min_samples_leaf=minSamples, random_state=1, max_depth=maxDepth)
  project_model.fit(X, y)
  predicted_approval = project_model.predict(X)
  trainingF1 = f1_score(y, predicted_approval)
  #check w/ validation model.
  X_valid = valid_df[explanatory_variables]
  y_valid = valid_df.project_is_approved
  predicted_approval = project_model.predict(X_valid)
  validationF1 = f1_score(y_valid, predicted_approval)

  return trainingF1, validationF1

  getTrainValidAccErr(minSamples=80, maxDepth=10, training_size=len(train_df))

  #Graph both maxDepth and minSamples
graphing_df = pd.DataFrame(columns=['max_Depth', 'min_Samples', 'training_Error', 'training_Accuracy', 'validation_Error', 'ValidationAccuracy'])
#graphing_df = pd.DataFrame(columns=['max_Depth', 'min_Samples', 'trainingF1', 'validationF1'])
for max_depth in range(1, 126, 9):
  for min_samples in range(1, 126, 9):
    trainValidAccErr = getTrainValidAccErr(minSamples=min_samples, maxDepth=max_depth, training_size=len(train_df))
    f1 = getF1_score(minSamples=min_samples, maxDepth=max_depth, training_size=len(train_df))
    new_row = {'max_Depth':max_depth, 'min_Samples': min_samples, 'training_Error':trainValidAccErr[0], 'training_Accuracy':trainValidAccErr[1], 'validation_Error':trainValidAccErr[2], 'ValidationAccuracy':trainValidAccErr[3]
               ,'train_f1':f1[0], 'valid_f1':f1[1]}
    #f1Score = getF1_score(minSamples=min_samples, maxDepth=max_depth, training_size=len(train_df))
    #new_row = {'max_Depth':max_depth, 'min_Samples': min_samples, 'trainingF1':f1Score[0], 'validationF1':f1Score[1]}
    #append row to the dataframe
    graphing_df = graphing_df.append(new_row, ignore_index=True)

graphing_df.head()

#focus on max depth samples from 1 to 30 and min samples from 100 to 130
focused_graphing_df = pd.DataFrame(columns=['max_Depth', 'min_Samples', 'training_Error', 'training_Accuracy', 'validation_Error', 'ValidationAccuracy'])
#graphing_df = pd.DataFrame(columns=['max_Depth', 'min_Samples', 'trainingF1', 'validationF1'])
for max_depth in range(1, 30):
  for min_samples in range(90, 120):
    trainValidAccErr = getTrainValidAccErr(minSamples=min_samples, maxDepth=max_depth, training_size=len(train_df))
    f1 = getF1_score(minSamples=min_samples, maxDepth=max_depth, training_size=len(train_df))
    new_row = {'max_Depth':max_depth, 'min_Samples': min_samples, 'training_Error':trainValidAccErr[0], 'training_Accuracy':trainValidAccErr[1], 'validation_Error':trainValidAccErr[2], 'ValidationAccuracy':trainValidAccErr[3]
               ,'train_f1':f1[0], 'valid_f1':f1[1]}
    #f1Score = getF1_score(minSamples=min_samples, maxDepth=max_depth, training_size=len(train_df))
    #new_row = {'max_Depth':max_depth, 'min_Samples': min_samples, 'trainingF1':f1Score[0], 'validationF1':f1Score[1]}
    #append row to the dataframe
    focused_graphing_df = focused_graphing_df.append(new_row, ignore_index=True)

focused_graphing_df.head()

focused_graphing_df['valid_f1'].describe()
graphing_df.iloc[graphing_df['valid_f1'].argmax()]
focused_graphing_df.iloc[focused_graphing_df['ValidationAccuracy'].argmax()]
sns.distplot(a=graphing_df['valid_f1'], kde=False)

cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots()
#x,y,color
points = ax.scatter(graphing_df['max_Depth'], graphing_df['min_Samples'], c=graphing_df['valid_f1'], s=50, cmap=cmap)
plt.xlabel("Maximum Tree Depth")
plt.ylabel("Minimum Samples at Leaf Node")
f.colorbar(points)

cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots()
#x,y,color
points = ax.scatter(focused_graphing_df['max_Depth'], focused_graphing_df['min_Samples'], c=focused_graphing_df['valid_f1'], s=50, cmap=cmap)
plt.xlabel("Maximum Tree Depth")
plt.ylabel("Minimum Samples at Leaf Node")
f.colorbar(points)

#Training Set Size Graph:
#maximum training size =  #len(train_df)
trainSize_df = pd.DataFrame(columns=['training_set_size', 'training_f1','valid_f1'])
for i in range(1000, len(train_df), 1000):
  f1_score = getF1_score(minSamples=118, maxDepth=10, training_size=i)
  new_row = {'training_set_size':i,'training_f1':f1_score[0], 'valid_f1':f1_score[1]}
  #append row to the dataframe
  trainSize_df = trainSize_df.append(new_row, ignore_index=True)
trainSize_df.head()

plt.figure(figsize=(10,10))
plt.ylabel("Accuracy")
sns.lineplot(data=trainSize_df[['valid_f1', 'training_f1', 'training_set_size']].set_index('training_set_size'))

#Max Depth Graph.:
#maxDepth_df = pd.DataFrame(columns=['max_Depth', 'training_Error', 'training_Accuracy', 'validation_Error', 'ValidationAccuracy'])
maxDepth_df = pd.DataFrame(columns=['max_Depth'])
for i in range(1, 126):
  #trainValidAccErr = getTrainValidAccErr(minSamples=1, maxDepth=i, training_size=len(train_df))
  f1_score = getF1_score(minSamples=118, maxDepth=i, training_size=len(train_df))
  new_row = {'max_Depth':i, 'training_f1':f1_score[0], 'valid_f1':f1_score[1]}
  #append row to the dataframe
  maxDepth_df = maxDepth_df.append(new_row, ignore_index=True)
maxDepth_df.head()

#plt.figure(figsize=(10,10))
sns.lineplot(data=maxDepth_df.set_index('max_Depth')[0:40])

#min samples at leaf node
#minSamples_df = pd.DataFrame(columns=['min_Samples', 'training_Error', 'training_Accuracy', 'validation_Error', 'ValidationAccuracy'])
minSamples_df = pd.DataFrame(columns=['min_Samples'])
for i in range(1, 126):
  #trainValidAccErr = getTrainValidAccErr(minSamples=i, maxDepth=17, training_size=len(train_df))
  f1_score = getF1_score(minSamples=i, maxDepth=15, training_size=len(train_df))
  new_row = {'min_Samples':i, 'training_f1':f1_score[0], 'valid_f1':f1_score[1]}
  #append row to the dataframe
  minSamples_df = minSamples_df.append(new_row, ignore_index=True)
minSamples_df.head()

plt.figure(figsize=(10,5))
sns.lineplot(data=minSamples_df.set_index('min_Samples')[0:125])

#use decision tree Classifier

#for predictor variables first put in all them numerical data
explanatory_variables = ['total_resource_cost', 'teacher_number_of_previously_posted_projects', 'number_essays',
                         'essay_1_length', 'essay_2_length', 'essay_3_length', 'essay_4_length']
#now add in all the categorical data column names to the list
for name in categorical_variables:
  explanatory_variables = explanatory_variables + combined_df[name].unique().tolist()
y = train_df.project_is_approved
X = train_df[explanatory_variables]

getF1_score(10,10,len(train_df))

#import the classifier
from sklearn.tree import DecisionTreeClassifier
#max features: , max_features=5
project_model = DecisionTreeClassifier(max_depth=10, random_state=1, min_samples_leaf=10)
project_model.fit(X, y)
predicted_approval = project_model.predict(X)
#from sklearn.metrics import mean_absolute_error
#print(mean_absolute_error(y, predicted_approval))
from sklearn.metrics import mean_squared_error
print("training error: "+str(mean_squared_error(y, predicted_approval)))
from sklearn.metrics import accuracy_score
print("training acc: "+str(accuracy_score(y, predicted_approval)))
#check w/ validation model.
X_valid = valid_df[explanatory_variables]
y_valid = valid_df.project_is_approved
predicted_approval = project_model.predict(X_valid)
#from sklearn.metrics import mean_absolute_error
#print(mean_absolute_error(y_valid, predicted_approval))
from sklearn.metrics import mean_squared_error
print("validation error: "+str(mean_squared_error(y_valid, predicted_approval)))
print("validation acc: "+str(accuracy_score(y_valid, predicted_approval)))

from matplotlib import pyplot as plt
from sklearn import tree
fig = plt.figure(figsize=(70,20))
_ = tree.plot_tree(project_model,
                   feature_names=explanatory_variables,
                   class_names=['Not_Approved', 'Approved'],
                   filled=True, fontsize=12)

def getRandomForestResults(number_of_trees, maxDepth, minSamples, maxFeatures):
  #Code for parameters if u want.
  #all imports:
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import mean_squared_error
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import f1_score

  #for predictor variables first put in all them numerical data
  explanatory_variables = ['total_resource_cost', 'teacher_number_of_previously_posted_projects', 'number_essays',
                          'essay_1_length', 'essay_2_length', 'essay_3_length', 'essay_4_length']
  #now add in all the categorical data column names to the list
  for name in categorical_variables:
    explanatory_variables = explanatory_variables + combined_df[name].unique().tolist()

  #training set
  y = train_df.project_is_approved
  X = train_df[explanatory_variables]
  project_model = RandomForestClassifier(random_state=1, n_estimators=number_of_trees, max_depth=maxDepth, min_samples_leaf=minSamples, max_features=maxFeatures) #put parameters here
  project_model.fit(X, y)
  predicted_approval = project_model.predict(X)
  #trainingError = mean_squared_error(y, predicted_approval)
  trainingAccuracy = accuracy_score(y, predicted_approval)
  training_f1 = f1_score(y, predicted_approval)
  #check w/ validation model.
  X_valid = valid_df[explanatory_variables]
  y_valid = valid_df.project_is_approved
  predicted_approval = project_model.predict(X_valid)
  #validationError = mean_squared_error(y_valid, predicted_approval)
  validationAccuracy = accuracy_score(y_valid, predicted_approval)
  validation_f1 = f1_score(y_valid, predicted_approval)

  return trainingAccuracy, validationAccuracy, training_f1, validation_f1

random_forest_df = pd.DataFrame(columns=[])
for number_of_trees in range(1, 12, 1):
  for max_depth in range(5, 12, 1):
    for max_features in range(2, 119, 9):
      for min_samples in range (1, 50, 5):
        randomForestResults = getRandomForestResults(number_of_trees, max_depth, min_samples, max_features)
        new_row = {'max_depth':max_depth, 'number_of_trees': number_of_trees,
                  'max_features':max_features, 'min_samples':min_samples,
                  'train_accuracy':randomForestResults[0], 'valid_accuracy':randomForestResults[1],
                  'train_f1':randomForestResults[2], 'valid_f1':randomForestResults[3]}
        random_forest_df = random_forest_df.append(new_row, ignore_index=True)

random_forest_df.head()

random_forest_df.iloc[random_forest_df['valid_accuracy'].argmax()]

#focused random forest
focused_rf_df = pd.DataFrame(columns=[])

#1) try to optimize accuracy score.
for number_of_trees in range(8, 12, 1):
  for max_depth in range(6, 10, 1):
    for max_features in range(45, 60, 1):
      for min_samples in range (14, 30, 1):
        randomForestResults = getRandomForestResults(number_of_trees, max_depth, min_samples, max_features)
        new_row = {'max_depth':max_depth, 'number_of_trees': number_of_trees,
                  'max_features':max_features, 'min_samples':min_samples,
                  'train_accuracy':randomForestResults[0], 'valid_accuracy':randomForestResults[1],
                  'train_f1':randomForestResults[2], 'valid_f1':randomForestResults[3]}
        focused_rf_df = focused_rf_df.append(new_row, ignore_index=True)

focused_rf_df.head()

random_forest_df.iloc[focused_rf_df['valid_accuracy'].argmax()]

cmap = sns.cubehelix_palette(as_cmap=True)

f, ax = plt.subplots()
#x,y,color
points = ax.scatter(random_forest_df['max_depth'], random_forest_df['number_of_trees'], c=random_forest_df['valid_accuracy'], s=50, cmap=cmap)
plt.xlabel("Maximum Tree Depth")
plt.ylabel("Number Of Trees")
f.colorbar(points)

#random forest code
#all imports:
from sklearn.ensemble import RandomForestClassifier
#for predictor variables first put in all them numerical data
explanatory_variables = ['total_resource_cost', 'teacher_number_of_previously_posted_projects', 'number_essays',
                            'essay_1_length', 'essay_2_length', 'essay_3_length', 'essay_4_length']
#now add in all the categorical data column names to the list
for name in categorical_variables:
  explanatory_variables = explanatory_variables + combined_df[name].unique().tolist()

#training set
y = train_df.project_is_approved
X = train_df[explanatory_variables]
project_model = RandomForestClassifier(random_state=1, n_estimators=10, max_depth=8, min_samples_leaf=21, max_features=56) #put parameters here
project_model.fit(X, y)
predicted_approval = project_model.predict(X)
"""#trainingError = mean_squared_error(y, predicted_approval)
trainingAccuracy = accuracy_score(y, predicted_approval)
training_f1 = f1_score(y, predicted_approval)"""
#check w/ validation model.
X_valid = valid_df[explanatory_variables]
y_valid = valid_df.project_is_approved
predicted_approval = project_model.predict(X_valid)
"""#validationError = mean_squared_error(y_valid, predicted_approval)
validationAccuracy = accuracy_score(y_valid, predicted_approval)
validation_f1 = f1_score(y_valid, predicted_approval)"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

confusion_matrix = pd.crosstab(y_valid, predicted_approval, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()

from sklearn.metrics import confusion_matrix
confusion_matrix(y_valid, predicted_approval)