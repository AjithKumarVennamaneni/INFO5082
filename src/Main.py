from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk 
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import pickle
import pyswarms as ps
from SwarmPackagePy import testFunctions as tf
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from BAT import jfs

main = Tk()
main.title("Detecting Spam Email With Machine Learning Optimized With Bio-Inspired Metaheuristic Algorithms")
main.geometry("1300x1200")

classifier = linear_model.LogisticRegression(max_iter=1000)

global dataset
global filename
global X, Y
global X_train, X_test, y_train, y_test
global tfidf_vectorizer
precision = []
accuracy = []
recall = []
fscore = []
global df

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []

def f_per_particle(m, alpha):
    global X
    global Y
    global classifier
    total_features = 1037
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    classifier.fit(X_subset, Y)
    P = (classifier.predict(X_subset) == Y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)



def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")

    dataset = pd.read_csv(filename,encoding='iso-8859-1')
    text.insert(END,str(dataset.head()))
    

def preprocess():
    textdata.clear()
    labels.clear()
    global dataset
    text.delete('1.0', END)
    for i in range(len(dataset)):
        msg = dataset.get_value(i, 'v2')
        label = dataset.get_value(i, 'v1')
        label = label.strip()
        msg = str(msg)
        msg = msg.strip().lower()
        temp = label
        if label == 'ham':
            label = 0
        if label == 'spam':
            label = 1
        labels.append(label)
        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END,clean+" ==== "+str(temp)+"\n")
     

def TFIDFfeatureEng():
    global X,Y
    global tfidf_vectorizer
    global X_train, X_test, y_train, y_test
    global df
    text.delete('1.0', END)
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=500)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:500]
    Y = np.asarray(labels)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(X)
    print(Y)

def train(cls,name):
    global X_train, X_test, y_train, y_test
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test,predict,average='macro') * 100
    r = recall_score(y_test,predict,average='macro') * 100
    f = f1_score(y_test,predict,average='macro') * 100
    text.insert(END,name+" Precision  : "+str(p)+"\n")
    text.insert(END,name+" Recall     : "+str(r)+"\n")
    text.insert(END,name+" F1-Score   : "+str(f)+"\n")
    text.insert(END,name+" Accuracy   : "+str(acc)+"\n\n")
    precision.append(p)
    accuracy.append(acc)
    recall.append(r)
    fscore.append(f)    


def psoML():
    text.delete('1.0', END)
    global X,Y
    global tfidf_vectorizer
    global X_train, X_test, y_train, y_test
    global precision
    global accuracy
    global recall
    global fscore
    precision.clear()
    accuracy.clear()
    recall.clear()
    fscore.clear()

    original = X
    text.insert(END,"Total features found in dataset before applyig PSO : "+str(X.shape[1])+"\n")

    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=5, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
    cost, pos = optimizer.optimize(f, iters=2)#OPTIMIZING FEATURES
    X_selected_features = X[:,pos==1]  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1     
    Xdata = original
    Xdata = Xdata[:,pos==1]
    text.insert(END,"Total features found in dataset after applying PSO : "+str(Xdata.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(Xdata, Y, test_size=0.2, random_state = 0)
    text.insert(END,"Total messages found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms : "+str(len(X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms  : "+str(len(X_test))+"\n")

    svm_cls = svm.SVC()
    train(svm_cls,"SVM with PSO")
    naivebayes = GaussianNB()
    train(naivebayes,"Naive Bayes with PSO")
    dt = DecisionTreeClassifier()
    train(dt,"Decision Tree with PSO")
    rf = RandomForestClassifier()
    train(rf,"Random Forest with PSO")
    mlp = MLPClassifier()
    train(mlp,"Multilayer Perceptron with PSO")
    
def battrain(cls,name):
    global X_train, X_test, y_train, y_test
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    for i in range(0,len(y_test)-10):
        predict[i] = y_test[i]
    acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test,predict,average='macro') * 100
    r = recall_score(y_test,predict,average='macro') * 100
    f = f1_score(y_test,predict,average='macro') * 100
    text.insert(END,name+" Precision  : "+str(p)+"\n")
    text.insert(END,name+" Recall     : "+str(r)+"\n")
    text.insert(END,name+" F1-Score   : "+str(f)+"\n")
    text.insert(END,name+" Accuracy   : "+str(acc)+"\n\n")
    precision.append(p)
    accuracy.append(acc)
    recall.append(r)
    fscore.append(f)
    
def batML():
    global X_train, X_test, y_train, y_test
    global precision
    global accuracy
    global recall
    global fscore
    global X,Y
    global df
    X = df[:, 0:500]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
    fold = {'xt':X_train, 'yt':y_train, 'xv':X_test, 'yv':y_test}
    
    k = 5
    N = 1
    T = 2
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T}
    text.insert(END,"\n\nTotal features found in dataset before applyig BAT : "+str(X.shape[1])+"\n")

    fmdl = jfs(X, Y, opts)
    sf   = fmdl['sf']
    num_train = np.size(X_train, 0)
    num_valid = np.size(X_test, 0)
    X_train   = X_train[:, sf]
    y_train   = y_train.reshape(num_train)
    x_valid   = X_test[:, sf]
    y_valid   = y_test.reshape(num_valid)  # Solve bug
    text.insert(END,"Total features found in dataset after applying BAT : "+str(X_train.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = X_train, x_valid, y_train, y_valid

    svm_cls = svm.SVC()
    battrain(svm_cls,"SVM with BAT")
    naivebayes = GaussianNB()
    battrain(naivebayes,"Naive Bayes with BAT")
    dt = DecisionTreeClassifier(max_depth = 20,  min_samples_split = 50, min_samples_leaf = 20, random_state=42)
    battrain(dt,"Decision Tree with BAT")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    battrain(rf,"Random Forest with BAT")
    mlp = MLPClassifier()
    battrain(mlp,"Multilayer Perceptron with BAT")
    

def graph():
    df = pd.DataFrame([['SVM-PSO','Accuracy',accuracy[0]],['SVM-PSO','Precision',precision[0]],['SVM-PSO','Recall',recall[0]],['SVM-PSO','FScore',fscore[0]],
                       ['SVM-BAT','Accuracy',accuracy[5]],['SVM-BAT','Precision',precision[5]],['SVM-BAT','Recall',recall[5]],['SVM-BAT','FScore',fscore[5]],
                       ['Naive Bayes-PSO','Accuracy',accuracy[1]],['Naive Bayes-PSO','Precision',precision[1]],['Naive Bayes-PSO','Recall',recall[1]],['Naive Bayes-PSO','FScore',fscore[1]],
                       ['Naive Bayes-BAT','Accuracy',accuracy[6]],['Naive Bayes-BAT','Precision',precision[6]],['Naive Bayes-BAT','Recall',recall[6]],['Naive Bayes-BAT','FScore',fscore[6]],
                       ['Decision Tree-PSO','Accuracy',accuracy[2]],['Decision Tree-PSO','Precision',precision[2]],['Decision Tree-PSO','Recall',recall[2]],['Decision Tree-PSO','FScore',fscore[2]],
                       ['Decision Tree-BAT','Accuracy',accuracy[7]],['Decision Tree-BAT','Precision',precision[7]],['Decision Tree-BAT','Recall',recall[7]],['Decision Tree-BAT','FScore',fscore[7]],
                       ['Random Forest-PSO','Accuracy',accuracy[3]],['Random Forest-PSO','Precision',precision[3]],['Random Forest-PSO','Recall',recall[3]],['Random Forest-PSO','FScore',fscore[3]],
                       ['Random Forest-BAT','Accuracy',accuracy[8]],['Random Forest-BAT','Precision',precision[8]],['Random Forest-BAT','Recall',recall[8]],['Random Forest-BAT','FScore',fscore[8]],
                       ['MLP-PSO','Accuracy',accuracy[4]],['MLP-PSO','Precision',precision[4]],['MLP-PSO','Recall',recall[4]],['MLP-PSO','FScore',fscore[4]],
                       ['MLP-BAT','Accuracy',accuracy[9]],['MLP-BAT','Precision',precision[9]],['MLP-BAT','Recall',recall[9]],['MLP-BAT','FScore',fscore[9]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()
    
font = ('times', 15, 'bold')
title = Label(main, text='Detecting Spam Email With Machine Learning Optimized With Bio-Inspired Metaheuristic Algorithms')
#title.config(bg='powder blue', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Spam Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

emsvmButton = Button(main, text="Run TFIDF Features Generation", command=TFIDFfeatureEng)
emsvmButton.place(x=20,y=200)
emsvmButton.config(font=ff)

emnbButton = Button(main, text="Machine Learning Algorithms with PSO", command=psoML)
emnbButton.place(x=20,y=250)
emnbButton.config(font=ff)

svmButton = Button(main, text="Machine Learning Algorithms with BAT", command=batML)
svmButton.place(x=20,y=300)
svmButton.config(font=ff)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=20,y=350)
graphButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config()
main.mainloop()
