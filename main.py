import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

"""
1. Load DSP_2.csv.
2. Drop following variables: 'Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope','FastingBS'.
3. Split data with test_size = 0.2. 
4. Run random forest (y = HeartDisease, n_estimators=10, max_depth = 4, random_state=0).
5. Provide visualisation (use graphviz).

Provide answer to following question - how can you describe the model? Is is good?
"""
df2 = pd.read_csv("DSP_2.csv", sep=",")

df2_removed = df2.dropna()

df2_ready = df2_removed.drop(["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"], axis=1)

X = df2_ready.drop(["HeartDisease"], axis=1)
y = df2_ready["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# n_estimators = number of decision trees created - more of them means better accuracy but requires more training data, time and memory
# max_depth = maximum depth of each decision tree i.e. the furthest possible distance from root node to the farthest leaf node.
# Generally higher depth can mean better accuracy but it can also lead to "overfitting" - this term means that we train our model
# "too well", making it find patterns specific to the data set we used. This can lead to situations where the model will perform poorly
# when working with new and unseen data sets
rfc = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)

# training data
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

# we have to choose one of the decision trees for visualization, we can choose any from index 0 to 9 as we have created a model with 10
# decision trees in total
# filled - to have colors on the nodes!!!!!
# rounder - makes boxes have rounded edges
estimator = rfc.estimators_[0]
dot_data = export_graphviz(estimator, out_file=None,
                           feature_names=X_train.columns,
                           class_names=['No heart disease', 'Heart disease'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("random_forest")

print(classification_report(y_test, y_pred, target_names=['No heart disease', 'Heart disease']))

"""
Notes about Random Forest render:
gini - purity of the leaf, it means how often a randomly chosen element for the set would be incorrectly labeled, the lower the purer the leaf
samples - the amount of samples used for this leaf
value - not that important anyway

Colors: They are impacted by 2 factors (IMPORTANT - SET "filled" TO True TO HAVE COLORS!!!!!)
- balance inside "value"
- pureness(gini)
For this render orange represents heart disease while blue lack of.. that. If "value" was [16, 16] we would get a white leaf regardless of the gini,
if it was [16, 0] with gini=0.0 we would get intense orange which stands for "heart disease"(which is on the left because APPARENTLY it's ordered 
alphabetically), while if we had [16, 0] but gini of 0.5 the leaf would be white again because of the impurity
"""

"""
Provide answer to following question - 
1. how can you describe the model? 
2.Is is good?
"""

# My answers:
# 1. Not really related but interestingly my first run produces different values in classification_report() despite me not remembering changing
# anything at all. The first value for precision for "No heart disease" was 0.77, but with subsequent runs it wound decrease to values such as
# 0.74 or 0.73. When I run it now all values stay the same, interesting

# I base my answers on these results:
# Heart disease:
# Precision: 0.82
# Recall: 0.77
# f1-score: 0.79
# Support: 107

# No heart disease:
# Precision: 0.70
# Recall: 0.77
# f1-score: 0.73
# Support: 77

# I haven't managed to improve the model by changing the depth or the number of trees. By analyzing the trees and the classification report I
# concluded that this model could use additional data as there was a significant number of (furthest)leafs with relatively high impurity(often >0.4) and
# the precision score for "No heart disease" isn't at all impressive, neither is 0.82 for "Heart disease" to my "statistically" inexperienced brain but
# it's at least close

# 2.
# In my critically uneducated opinion - it's mediocre, pretty much like the one from task 3(same data, though different distribution, but it checks out more
# or less), it's correct quite often but not often enough to be even close to "reliable", I would be curious to see its performance with more data of this kind

# I remembered that test_size was different in task 3, so I got curious and compared the results when test_size here was also set to 0.1, it seems
# that this method(at least for this data set) is actually inferior pretty much across the board, beaten by 3-5%
# with that said changing the test_size significantly improved precision for "No heart disease" so perhaps the settings for this model could still be
# optimized, or perhaps I just don't know what I'm doing...