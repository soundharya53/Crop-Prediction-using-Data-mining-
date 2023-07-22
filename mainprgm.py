import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data
import matplotlib.pyplot as plt
from tkinter import *
import tkinter.messagebox as mb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import zero_one_loss as accuracy_score
from sklearn.metrics import confusion_matrix
from pyswarm import pso
from parameters import parametres
import matplotlib.pyplot as plt

df = pd.read_csv("FinalDataset.csv")
df1=df.drop_duplicates()
pr=df1['Production'].mode()[0]
t=df1['Temperature'].mean()
h=df1['humidity'].mean()
ph=df1['ph'].mean()
r=df1['rainfall'].mean()
df1['Production']=df1['Production'].fillna(value=pr)
df1['Temperature']=df1['Temperature'].fillna(value=t)
df1['humidity']=df1['humidity'].fillna(value=h)
df1['ph']=df1['ph'].fillna(value=ph)
df1['rainfall']=df1['rainfall'].fillna(value=r)
df1=df1.dropna()  
df1.to_csv('modified_dataset.csv')
print("Mode of Production:",pr)
print("Mean of Temperature:",t)
print("Mean of Humidity:",h)
print("Mean of pH:",ph)
print("Mean of Rainfall:",r)

path =r"C:\Users\sound\source\repos\cropprediction\cropprediction\FinalDataset.csv"
training_set = pd.read_csv(path)
training_set = np.array(training_set)
idnum = int(max(training_set[:,0]))
param = int(max(training_set[:,1]))
def convert(data):
    new_data = []
    for paramnum in range(1, idnum+1):
        paramid = data[:,1][data[:,0] == paramnum]
        rate = data[:,2][data[:,0] == paramnum]
        ratings = np.zeros(param)
        ratings[paramid - 1] = rate
        new_data.append(list(ratings))
    return np.array(new_data)
training_set = torch.FloatTensor(convert(training_set))
def convert_likes(df):
    df[df == 0] = -1
    df[df == 1] = 0
    df[df == 2] = 0
    df[df >=3 ] = 1
    return df
training_set = convert_likes(training_set)
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nv, nh)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        
    def sample_h(self, x):
        wx = torch.mm(x, self.W)
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W.t())
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk, lr=0.01):
        self.W += lr * torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += lr * torch.sum((v0 - vk), 0)
        self.a += lr * torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 16
batch_size = 32
rbm = RBM(nv, nh)
nb_epoch = 10
lr = 0.03
losses = []

for epoch in range(1, nb_epoch + 1):
    
    train_loss = 0
    epoch_loss = []
    s = 0.0
    
    for idnum in range(0, idnum - batch_size, batch_size):
        
        vk = training_set[idnum:idnum+batch_size]
        v0 = training_set[idnum:idnum+batch_size]
        
        ph0,_ = rbm.sample_h(v0)
        for k in range(100):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
            
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk, lr)
        
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        epoch_loss.append(train_loss/s)
        s += 5.0
        
    losses.append(epoch_loss[-1])
    if(epoch %2 == 0):
        print('Epoch:{0:4d} Train Loss:{1:1.4f}'.format(epoch, train_loss/s))
root = Tk()

# Set size of the form
root.geometry("500x500")

#Providing title to the form
root.title('crop prediction')

#this creates 'Label' widget for Registration Form and uses place() method.
label_0 =Label(root,text="crop prediction", width=20,font=("bold",20))
#place method in tkinter is  geometry manager it is used to organize widgets by placing them in specific position
label_0.place(x=90,y=60)

#this creates 'Label' widget for Fullname and uses place() method.
label_1 =Label(root,text="Temperature", width=20,font=("bold",10))
label_1.place(x=80,y=130)

#this will accept the input string text from the user.
entry_1=Entry(root)
entry_1.pack()
entry_1.place(x=240,y=130)

#this creates 'Label' widget for Email and uses place() method.
label_3 =Label(root,text="Humidity", width=20,font=("bold",10))
label_3.place(x=68,y=180)

entry_3=Entry(root)
entry_3.pack()
entry_3.place(x=240,y=180)

label_4 =Label(root,text="PH", width=20,font=("bold",10))
label_4.place(x=68,y=220)

entry_4=Entry(root)
entry_4.pack()
entry_4.place(x=240,y=220)

label_5=Label(root,text="Rainfall", width=20,font=("bold",10))
label_5.place(x=58,y=270)

entry_5=Entry(root)
entry_5.pack()
entry_5.place(x=240,y=270)
    #importing tkinter module for GUI application
def open_popup():
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from pyswarm import pso

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
    ub = [10, 12467]

    n_particles =2
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

    accuracy1 = accuracy_score(y_test, y_pred)
    print("Accuracy on the testing set: ", accuracy1)


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
    accuracy = classifier.score(X_test, y_test)
    print('The classification accuracy score is:', accuracy)

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

    lb = [2, 1]
    ub = [11, 12675]
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
    a1= accuracy_score(y_test, y_pred)
    print("Accuracy on the testing set: ", a1)
    if(value>accuracy1):
        v1 = entry_1.get()
        v2 = entry_3.get()
        v3 = entry_4.get()
        v4 = entry_5.get()
        print(v1)
        print(v2)
    # Remove any leading or trailing spaces
        a = v1.strip()
        b = v2.strip()
        c = v3.strip()
        d = v4.strip()
    # Check if the string is empty
        if v1 == '':
          v11 = 0.000
        else:
            a = ''.join(c for c in a if c.isdigit() or c == '.')
            v11 = float(a)
        if v2 =='':
            v12 = 0.000
        else:
            b = ''.join(c for c in b if c.isdigit() or c == '.')
            v12 = float(b)
        if v3 == '':
            v13 = 0.000
        else:
            c = ''.join(c for c in c if c.isdigit() or c == '.')
            v13 = float(c)
        if v4 == '':
            v14 = 0.0000
        else:
            d = ''.join(c for c in d if c.isdigit() or c == '.')
            v14 = float(d)
        X_train_new = kmeans.transform(X_train)
        gb = GaussianNB()
        gb.fit(X_train_new, y_train)
        X_test_new = kmeans.transform(X_test)
        y_pred = gb.predict(X_test_new)

        mb.showinfo('Result', f'Temperature: {v11}, Humidity: {v12}, PH: {v13}, Rainfall: {v14}')
        input_data = [[v11,v12,v13,v14]]
        print(input_data)
        proba = model.predict_proba(input_data)[0]
        c = pd.read_csv("Crop.csv")
        crop_probs = list(zip(c['Crop'], proba))
        crop_probs.sort(key=lambda x: x[1], reverse=True) # sort by descending probability
        top_n = 5
        predicted_crops = [crop_probs[i][0] for i in range(top_n)]
        crops = [crop.replace('\xa0', ' ') for crop in predicted_crops]
        print(f"The top {top_n} predicted crops are: {crops}")
        mb.showinfo('crops:{crops}')
        print(value)
        print(accuracy1)
    else:
        v1 = entry_1.get()
        v2 = entry_3.get()
        v3 = entry_4.get()
        v4 = entry_5.get()
        print(v1)
        print(v2)
    # Remove any leading or trailing spaces
        a = v1.strip()
        b = v2.strip()
        c = v3.strip()
        d = v4.strip()
    # Check if the string is empty
        if v1 == '':
          v11 = 0.000
        else:
            a = ''.join(c for c in a if c.isdigit() or c == '.')
            v11 = float(a)
        if v2 =='':
            v12 = 0.000
        else:
            b = ''.join(c for c in b if c.isdigit() or c == '.')
            v12 = float(b)
        if v3 == '':
            v13 = 0.000
        else:
            c = ''.join(c for c in c if c.isdigit() or c == '.')
            v13 = float(c)
        if v4 == '':
            v14 = 0.0000
        else:
            d = ''.join(c for c in d if c.isdigit() or c == '.')
            v14 = float(d)
        df2=X.drop(['Crop'], axis=1)
        X=df2.to_numpy()
        Y=y.to_numpy()
        model=DecisionTreeClassifier()
        model.fit(X,Y)
        mb.showinfo('Result', f'Temperature: {v11}, Humidity: {v12}, PH: {v13}, Rainfall: {v14}')
        input_data = [[v11,v12,v13,v14]]
        input_data = [[v11,v12,v13,v14]]
        proba = model.predict_proba(input_data)[0]
        c = pd.read_csv("Crop.csv")
        crop_probs = list(zip(c['Crop'], proba))
        crop_probs.sort(key=lambda x: x[1], reverse=True) # sort by descending probability
        top_n = 5
        predicted_crops = [crop_probs[i][0] for i in range(top_n)]
        crops = [crop.replace('\xa0', '') for crop in predicted_crops]
        print(f"The top {top_n} predicted crops are: {crops}")
        mb.showinfo('crops',f'crops:{crops}')
        print(accuracy1)          
 
Button(root, text='Submit',width=15,bg="pink",command=open_popup).place(x=200,y=300)
root.mainloop()
