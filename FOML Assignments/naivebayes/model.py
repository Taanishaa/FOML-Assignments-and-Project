import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import statistics as stat
import warnings
warnings.filterwarnings('ignore')
class NaiveBayes:

    def gaussian_fit(self,X,c):
        logpri=np.log(self.priors[c])
        loglike=-0.5*np.sum(np.log(2 * np.pi * self.gaussian[c]['sig2']))-0.5*np.sum(((X - self.gaussian[c]['mu']) ** 2) / self.gaussian[c]['sig2'])
        return logpri+loglike


    def bernoulli_fit(self,X,c):
        logpri=np.log(self.priors[c])
        loglike = np.sum(X * np.log(self.bernoulli[c]['p']) + (1 - X) * np.log(1 - self.bernoulli[c]['p']))
        return logpri+loglike
    
    def laplace_fit(self,X,c):
        logpri=np.log(self.priors[c])
        loglike=len(X)*np.sum(np.log(np.sqrt(0.5/self.laplace[c]['scale']))-np.sqrt(2)*np.abs(X-self.laplace[c]['mean'])/self.laplace[c]['scale'])
        return logpri+loglike
    
    def exponential_fit(self,X,c):
        logpri=np.log(self.priors[c])
        loglike=np.sum(np.log(self.exponential[c]['lambda'])-self.exponential[c]['lambda']*X)
        return logpri+loglike
    
    def multinomial_fit(self,X,c):
        X=X.astype(int)
        logpri=np.log(self.priors[c])
        loglike=np.log(self.multinomial2[c][0][X[0]]*self.multinomial2[c][1][X[1]])
        return logpri+loglike

    def fit(self, X, y):
        self.priors={}
        self.gaussian={}
        self.num_classes=3
        self.bernoulli={}
        self.exponential={}
        self.laplace={}
        self.multinomial={}
        self.multinomial2={}
        """Start of your code."""
        """
        X : np.array of shape (n,10)
        y : np.array of shape (n,)
        Create a variable to store number of unique classes in the dataset.
        Assume Prior for each class to be ratio of number of data points in that class to total number of data points.
        Fit a distribution for each feature for each class.
        Store the parameters of the distribution in suitable data structure, for example you could create a class for each distribution and store the parameters in the class object.
        You can create a separate function for fitting each distribution in its and call it here.
        """
        n=y.shape[0]
        for cl in range(self.num_classes):
            self.priors[cl]=np.sum(y==cl)/n
            self.gaussian[cl]={'mu':np.mean(X[:,0:2][y==cl],axis=0),'sig2':np.var(X[:,0:2][y==cl],axis=0)}
            self.bernoulli[cl] = {"p":(np.sum(X[:,2:4][y==cl],axis=0))/(np.sum(y==cl))}
            self.laplace[cl]={'mean':np.mean(X[:,4:6][y==cl],axis=0),'scale':np.std(X[:,4:6][y==cl],axis=0)}
            self.exponential[cl]={'lambda':1.0/np.mean(X[:,6:8][y==cl],axis=0)}
            self.multinomial[cl]=[]
            t1=[]
            for i in range(4):
                t1.append((X[y==cl][:,8]==i).mean())

            self.multinomial[cl].append(t1)
            t2=[]
            for i in range(8):
                t2.append((X[y==cl][:,9]==i).mean())

            self.multinomial[cl].append(t2)


            a = np.bincount(X[:,8].astype(int)[y == cl], minlength=4)
            b = np.bincount(X[:,9].astype(int)[y == cl], minlength=8)
            likelihood_a =(a+1)/(a.sum()+4)
            likelihood_b=(b+1)/(b.sum()+8)
            self.multinomial2[cl]=[list(likelihood_a),list(likelihood_b)]




        """End of your code."""

    def predict(self, X):
        """Start of your code."""
        """
        X : np.array of shape (n,10)

        Calculate the posterior probability using the parameters of the distribution calculated in fit function.
        Take care of underflow errors suitably (Hint: Take log of probabilities)
        Return an np.array() of predictions where predictions[i] is the predicted class for ith data point in X.
        It is implied that prediction[i] is the class that maximizes posterior probability for ith data point in X.
        You can create a separate function for calculating posterior probability and call it here.
        """
        pred=[]
        for x in X:
            gposterior=[]
            bposterior=[]
            lposterior=[]
            eposterior=[]
            mposterior=[]

            for cl in range(self.num_classes):
                gposterior.append(self.gaussian_fit(x[0:2],cl))
                bposterior.append(self.bernoulli_fit(x[2:4],cl))
                lposterior.append(self.laplace_fit(x[4:6],cl))
                eposterior.append(self.exponential_fit(x[6:8],cl))
                mposterior.append(self.multinomial_fit(x[8:10],cl))


            gposterior=np.array(gposterior)
            bposterior=np.array(bposterior)
            lposterior=np.array(lposterior)
            eposterior=np.array(eposterior)
            mposterior=np.array(mposterior)
            prediction=gposterior+bposterior+lposterior+eposterior+mposterior
            pred.append(np.argmax(prediction))

        return np.array(pred)

    def getParams(self):
        """
        Return your calculated priors and parameters for all the classes in the form of dictionary that will be used for evaluation
        Please don't change the dictionary names
        Here is what the output would look like:
        priors = {"0":0.2,"1":0.3,"2":0.5}
        gaussian = {"0":[mean_x1,mean_x2,var_x1,var_x2],"1":[mean_x1,mean_x2,var_x1,var_x2],"2":[mean_x1,mean_x2,var_x1,var_x2]}
        bernoulli = {"0":[p_x3,p_x4],"1":[p_x3,p_x4],"2":[p_x3,p_x4]}
        laplace = {"0":[mu_x5,mu_x6,b_x5,b_x6],"1":[mu_x5,mu_x6,b_x5,b_x6],"2":[mu_x5,mu_x6,b_x5,b_x6]}
        exponential = {"0":[lambda_x7,lambda_x8],"1":[lambda_x7,lambda_x8],"2":[lambda_x7,lambda_x8]}
        multinomial = {"0":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"1":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"2":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]]}
        """

        priors = self.priors
        gaussian={}
        bernoulli = {}
        laplace = {}
        exponential = {}
        multinomial = self.multinomial
        for c in range(self.num_classes):
            gaussian[c] = [self.gaussian[c]['mu'][0],self.gaussian[c]['mu'][1],self.gaussian[c]['sig2'][0],self.gaussian[c]['sig2'][1]]
            bernoulli[c]= [self.bernoulli[c]['p'][0],self.bernoulli[c]['p'][0]]
            laplace[c]=[self.laplace[c]['mean'][0],self.laplace[c]['mean'][1],self.laplace[c]['scale'][0],self.laplace[c]['scale'][1]]
            exponential[c]=list(self.exponential[c]['lambda'])

        return (priors, gaussian, bernoulli, laplace, exponential, multinomial)        


def save_model(model,filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open("model.pkl","wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class 

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""
        
        n=(predictions==label).sum()
        return (true_labels[predictions==label]==label).sum()/n
    


        
        """End of your code."""
        


    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        
        n=(true_labels==label).sum()
        return (true_labels[predictions==label]==label).sum()/n



        """End of your code."""
        

    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""
        
        pre=precision(predictions,true_labels,label)
        rec=recall(predictions,true_labels,label)
        return (pre*rec)/(pre+rec)




        """End of your code."""
        return f1
    

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size



if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    # visualise(validation_datapoints, validation_predictions, "validation_predictions.png")

