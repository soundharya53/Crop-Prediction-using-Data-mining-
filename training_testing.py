import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pyswarm import pso
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import zero_one_loss as accuracy_score
from sklearn.metrics import confusion_matrix
from parameters import parametres
import matplotlib.pyplot as plt
fp = open(r'FinalDataset.csv', 'r')
data = pd.read_csv(fp)
# Print the first few rows to verify the data is loaded properly
print(data.head())
features = ['District ', 'Crop_Year', 'Season', 'Crop', 'Area ', 'Production', 'Temperature', 'humidity', 'ph', 'rainfall']
train_size = int(0.8 * len(data))
train_features = data[features][:train_size]
train_labels = data['Crop'][:train_size]
test_features = data[features][train_size:]
test_labels = data['Crop'][train_size:]
dt_model = DecisionTreeClassifier()
dt_model.fit(train_features, train_labels)
dt_predictions = dt_model.predict(test_features)
dt_accuracy = accuracy_score(test_labels, dt_predictions)
print("Decision tree accuracy:", dt_accuracy)
features = ['Temperature', 'humidity', 'ph', 'rainfall']
data_scaled = (data[features] - data[features].mean()) / data[features].std()
kmeans = KMeans(n_clusters=5)
kmeans.fit(data_scaled)
data['Cluster'] = kmeans.labels_
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def fitness_function(params, X_train, y_train):
    # Unpack the parameters
    n_clusters = int(params[0])
    max_depth = int(params[1])
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    # Create a new dataset with the cluster assignments as features
    X_train_new = kmeans.transform(X_train)
    # Train a decision tree on the new dataset
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    clf.fit(X_train_new, y_train)
    # Calculate the accuracy score on the training set
    y_pred_train = clf.predict(X_train_new)
    accuracy = accuracy_score(y_train, y_pred_train)
    # Return the negative of the accuracy score (since we want to maximize accuracy)
    return -accuracy
lb = [2, 1]
ub = [10, 12675]
n_particles = 50
#options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
parameters, value = pso(fitness_function, lb, ub, args=(X_train, y_train), swarmsize=n_particles)
print("Optimized parameters:")
print("Number of clusters: ", int(parameters[0]))
print("Max depth: ", int(parameters[1]))
kmeans = KMeans(n_clusters=int(parameters[0]), random_state=42)
kmeans.fit(X_train)
X_train_new = kmeans.transform(X_train)
clf = DecisionTreeClassifier(max_depth=int(parameters[1]), random_state=1)
clf.fit(X_train_new, y_train)
X_test_new = kmeans.transform(X_test)
y_pred = clf.predict(X_test_new)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the testing set: ", accuracy)
df = pd.read_csv("modified_dataset.csv")
print(df.columns)
X= df.drop(['Unnamed: 0','Id','District ','Crop_Year','Season','Area ','Production'],axis=1)
y = df.Crop
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)
km = KMeans(n_clusters=5, random_state=42)
km.fit(X_train)
X_train_new = km.transform(X_train)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy1 = classifier.score(X_test, y_test)
print('The classification accuracy score is:', accuracy1)
print(y_test.value_counts())
cm=confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
Sensitivity = TP / float(TP + FN)
print('Sensitivity : {0:0.4f}'.format(Sensitivity))
Specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(Specificity))
F_measure = 2*precision*Sensitivity / (precision+Sensitivity)
print('F-measure : {0:0.4f}'.format(Specificity))
def fitness_function(params, X_train, y_train):
    n_clusters = int(params[0])
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(X_train)
    X_train_new = km.transform(X_train)
    gb = GaussianNB()
    gb.fit(X_train_new, y_train)
    y_pred_train = gb.predict(X_train_new)
    a = accuracy_score(y_train, y_pred_train)
    return a
parameters, value = pso(fitness_function, lb, ub, args=(X_train, y_train),swarmsize=5)
print("Accuracy after pso before testing again: ", value)
selected_features=[]
for i in range(12):
    if parametres[1][i] == 1:
        selected_features.append(df.columns[i])
print(selected_features)
print("Clusters:",round(parameters[0],0))
kmeans = KMeans(n_clusters=int(parameters[0]), random_state=42)
kmeans.fit(X_train)
X_train_new = kmeans.transform(X_train)
gb = GaussianNB()
gb.fit(X_train_new, y_train)
X_test_new = kmeans.transform(X_test)
y_pred = gb.predict(X_test_new)
a1 = accuracy_score(y_test, y_pred)
print("Accuracy on the testing set: ", a1)
df2=X.drop(['Crop'], axis=1)
X=df2.to_numpy()
Y=y.to_numpy()
model=GaussianNB()
model.fit(X,Y)
input_data = [[24.52922681,80.54498576,7.070959995,260.2634026]]
proba = model.predict_proba(input_data)[0]
c = pd.read_csv("Crop.csv")
crop_probs = list(zip(c['Crop'], proba))
crop_probs.sort(key=lambda x: x[1], reverse=True) # sort by descending probability
top_n = 5
predicted_crops = [crop_probs[i][0] for i in range(top_n)]
crops = [crop.replace('\xa0', ' ') for crop in predicted_crops]
print(f"The top {top_n} predicted crops are: {crops}")
