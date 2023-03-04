import os
import sys
import time
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import copy
import psutil
import warnings
import sigdirect
import cascade_forest 
import random
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report,precision_score,recall_score,f1_score
t1=time.time()
class _Preprocess:

    def __init__(self):
        self._label_encoder = None

    def preprocess_data(self, raw_data):
        """ Given one of UCI files specific to SigDirect paper data,
            transform it to the common form used in sklearn datasets"""
        transaction_data = [list(map(int, x.strip().split())) for x in raw_data]
        max_val = max([max(x[:-1]) for x in transaction_data])
        X, y = [], []

        for transaction in transaction_data:
            positions = np.array(transaction[:-1]) - 1
            transaction_np = np.zeros((max_val))
            transaction_np[positions] = 1
            X.append(transaction_np)
            y.append(transaction[-1])
        X = np.array(X)
        y = np.array(y)

        # converting labels
        if self._label_encoder is None:  # train time
            unique_classes = np.unique(y)
            self._label_encoder = defaultdict(lambda: 0, zip(unique_classes, range(len(unique_classes))))
            y = np.vectorize(self._label_encoder.get)(y)
        else:  # test time
            y = np.vectorize(lambda a: self._label_encoder[a])(y)

        return X, y

tt1 = time.time()
def test_uci():

    size=[30] #add more element here for multiple windows
    #data=["wave"]
    #data=["flare"]
    data=["pima","glass","hepati","anneal","horse","mushroom","adult","ionosphere"]
    data=["pageblocks"]
    for dataset_name in data:       
        print(dataset_name)
        train_filename = os.path.join('uci', '{}.txt'.format(dataset_name))
        all_pred_y = defaultdict(list)
        all_true_y = []
        # counting number of rules before and after pruning
        generated_counter = 0
        final_counter = 0
        avg = [0.0] * 4
        tt1 = time.time()
        prep = _Preprocess()
        # load the training data and pre-process it
        with open(train_filename) as f:
            raw_data = f.read().strip().split('\n')

        X, y = prep.preprocess_data(raw_data)
        print("len",len(X),len(X[0]))
        #X=X[:int(len(X)/10)]
        #y=y[:int(len(y)/10)]
        #print(len(X),len(X[0]))
        dmax=-1     

        pred1 = []
        pred2 = []
        pred3 = []
        check=[]
        acc=[]
        #print("Start")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #print("size",len(X_train[0]))
        for window_size in size:
            #print(len(X_train[0])-window_size)
            for i in range(len(X_train[0])-window_size):
                #print(window_size)
                print(i)
                sliced_X_train=X_train[:,i:i+window_size]
                sliced_X_test=X_test[:,i:i+window_size]
                sliced_x=X[:,i:i+window_size]
                #print(sliced_x.shape)                  
               
                try:
                    clf = sigdirect.SigDirect(get_logs=sys.stdout)
                    generated_c, final_c,memory = clf.fit(sliced_X_train, y_train)

               
              
                    prob=clf.predict_proba(X,1)
                    X=np.concatenate((X, prob), axis=1)
                except:
                    pass
                    #print("error")
        #print(len(X[0]))
        cascade(X,y)






def cascade(X,y):
    run=True
    max_acc=-1

    max_p=-1
    max_r=-1
    while run:
        t1=time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        generated_c = 0
        final_c = 0
        """
        clf = sigdirect.SigDirect(get_logs=sys.stdout)
        g, f = clf.fit(X_train, y_train)
        for i in (1, 2, 3):
            y_pred=clf.predict(X_test, i)
            print('ACC S{}:'.format(i), accuracy_score(y_test, y_pred))
        """

        pred2 = []
        #model = []
        no_of_predictor=0
        #print("hello")
        acc=[]
        #pre=[]
        #rec=[]
        memory=0
        preds=[]
        for i in range(100): #change this value for number of base learners
            print(i)
            try:
                clf = sigdirect.SigDirect(get_logs=sys.stdout)
                random.seed(1)


                sample, sample_test = subsample(X_train, X_test,.6)
                #print(type(sample))
                if(not isinstance(sample, pd.core.frame.DataFrame)):
                    break

                sample = np.array(sample)
                sample_test = np.array(sample_test)
                #print(sample.shape)
                # print("sample size",len(sample))
                # sample_y = np.array(sample_y)
                no_of_predictor+=1
                g, f,m = clf.fit(sample, y_train)
                if(m>memory):
                    memory=m
                prob=clf.predict_proba(X,2)
                temp2=tuple(map(tuple,prob))
                if(temp2 not in preds):
                    preds.append(temp2)
                    #print(temp2)
                    X=np.concatenate((X, prob), axis=1)
                    #print("accuracy:", accuracy_score(y_test,clf.predict(sample_test, 1)),accuracy_score(y_test,clf.predict(sample_test, 2)),accuracy_score(y_test,clf.predict(sample_test, 3)))
                    #acc.extend([accuracy_score(y_test,clf.predict(sample_test, 1)),accuracy_score(y_test,clf.predict(sample_test, 2)),accuracy_score(y_test,clf.predict(sample_test, 3))])
                else:
                    occurance=preds.count(temp2)
                    preds.append(temp2)
                    X=np.concatenate((X, prob*occurance), axis=1)
                    #X =np.delete(X,np.index(prob))


                pred2.append(clf.predict(sample_test, 2))
                model.append(clf)
                #print("accuracy:", accuracy_score(y_test,clf.predict(sample_test, 1)),accuracy_score(y_test,clf.predict(sample_test, 2)),accuracy_score(y_test,clf.predict(sample_test, 3)))
                acc.append(accuracy_score(y_test,clf.predict(sample_test, 2)))
                #pre.append(precision_score(y_test,clf.predict(sample_test, 2),average='macro'))
                #rec.append(recall_score(y_test,clf.predict(sample_test, 2),average='macro'))
                generated_c += g
                final_c += f
            except:
                pass






        final_prediction = []
        pred2 = np.array(pred2)
        pred2 = pred2.transpose()
        for i in pred2:
            i = list(i)
            final_prediction.append(max(i, key=i.count))
        run, max_acc=calculate_score(y_test,final_prediction,acc,max_acc)
        #max_p=pre
        #prev_r=rec


        #print("Accuracy:", accuracy_score(y_test, final_prediction), end =" ")
        #print( "precission",precision_score(y_test, final_prediction,average='macro'),"recall",recall_score(y_test, final_prediction,average='macro'), "f1 score",f1_score(y_test, final_prediction,average='macro'))






    # Create a random subsample from the dataset with replacement


def subsample(dataset, test_dataset,ov):
    df = pd.DataFrame(dataset)
    df2 = pd.DataFrame(test_dataset)
   
    sample = pd.DataFrame()
    test_sample = pd.DataFrame()
    y_sample = pd.DataFrame()
    index = []
    #n_feature = df.shape[1]
    #print("shape",df.shape[1])

    #change this for number of features in each of the base learners
    n_feature = 30 #


    #print("nfeature",n_feature)
    temp = []
    count=0
    random.seed()
    while (len(temp) < n_feature):
        index = random.randint(0, df.shape[1] - 1)
      
        #if (index not in temp):
        temp.append(index)
        # print(index)
        sample[sample.shape] = df[index]
        test_sample[test_sample.shape] = df2[index]

  
    return sample, test_sample







def calculate_score(y_test,final_prediction,acc,max_acc):
    acc.append(accuracy_score(y_test,final_prediction))
    #pre.append(precision_score(y_test,final_prediction,average='macro'))
    #rec.append(recall_score(y_test,final_prediction,average='macro'))
    if(max_acc<max(acc)):
        max_acc=max(acc)
        print("layer_accuracy:",max_acc)
        return True, max_acc

    else:
        m=max(acc)
        print("layer_max_accuracy:",m)
        print("final_accuracy:",max_acc)
        #m=prev_ac.index(max(prev_ac))
        #print("accuracy",prev_ac[m], "precision",prev_p[m], "recall", prev_r[m])
        print( "precission",precision_score(y_test, final_prediction,average='macro'),"recall",recall_score(y_test, final_prediction,average='macro'), "f1 score",f1_score(y_test, final_prediction,average='macro'))
        process = psutil.Process(os.getpid())
        print("memory",process.memory_info().rss/ 1024 ** 2, end=" ")
        print("time",time.time()-tt1)
        return False, max_acc


memory=0
if __name__ == '__main__':
    warnings.filterwarnings("ignore") #there are warning for some base learners where the base learners doen't learn anything
    start_time = time.time()
    #print("test")
    test_uci()
    end_time=time.time()
    print("sig memory:", memory)
    print("required_time:", end_time-start_time)