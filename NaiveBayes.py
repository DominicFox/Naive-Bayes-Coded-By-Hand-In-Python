import pandas as pd
import math
from sklearn.model_selection import train_test_split

def Mean(data):
    return sum(data)/len(data)

def Std(data):
    mu = Mean(data)
    newList = [(x-mu)**2 for x in data]
    return math.sqrt((1/(len(data)-1))*sum(newList))

def GaussianPDF(number, mu, sigma):
    return (1/(sigma*math.sqrt(2*math.pi)))*math.exp(-0.5*((number - mu)/sigma)**2)

def CalculateProbabilityOfClassC(c, model):
    # Function computes the probability of a class c, using counts of how many samples are in each class.
    numberOfSamplesInClassC = model[c][0][2]
    numberOfSamplesInTotal = sum([model[x][0][2] for x in list(model.keys())])
    return numberOfSamplesInClassC/numberOfSamplesInTotal

def SeparateClasses(dataset):
    # Function separates the classes in a dataset, saving the result as a dictionary with classes as the keys.
    separateClassesDict = {}
    dataByClass = dataset.groupby('class')

    for group, data in dataByClass:
        separateClassesDict[group] = data
    
    return separateClassesDict

def CalculateColumnStatsForGivenClass(singleClassDataset):
    # Function calculates the mean, std and sample count for each column, given a single class
    requiredList = []

    # Get all the column names in this class as a list
    cols = list(singleClassDataset.columns)

    # Loop through the list of column names, calculating mean, std, and sample count
    for x in list(cols[0:len(cols)-1]): # -1 because we don't want to include the column for 'class'
        requiredList.append([Mean(singleClassDataset[x]), Std(singleClassDataset[x]), len(singleClassDataset[x])])
    return requiredList

def TrainModel(dataset):
    # Function produces a dictionary representing the Gaussian naive bayes model. 
    result = {}

    # First separate the classes
    dataSeparatedByClassDict = SeparateClasses(dataset)

    # Loop through each class
    for x in list(dataSeparatedByClassDict.keys()):
        # Calculate the mean and std for each attribute, within each class
        result[x] = CalculateColumnStatsForGivenClass(dataSeparatedByClassDict[x])
    return result

def PredictClass(model, newData):
    # This function uses the trained model to make a prediction on one new sample of data
    probabilityOfBelongingToEachClassDict = {}

    classes = list(model.keys())

    # Loop through all the possible classes the data could belong to
    for c in classes:
        listOfProbabilitiesToMultiply = []

        # First calculate the probability of the class (proportion of that class in training data)
        listOfProbabilitiesToMultiply.append(CalculateProbabilityOfClassC(c,model))

        # Loop through all attributes of new data and calculate the gaussian pdf value using relevant mean and std.
        for i in range(0, len(newData)):
            listOfProbabilitiesToMultiply.append(GaussianPDF(newData[i], model[c][i][0], model[c][i][1]))

        # Multiply all calculated values together to form probability data belongs to each class
        probabilityOfBelongingToEachClassDict[c] = math.prod(listOfProbabilitiesToMultiply)

    keys = list(probabilityOfBelongingToEachClassDict.keys())
    values = list(probabilityOfBelongingToEachClassDict.values())

    # Choose the class with highest probaiblity associated with it
    predictedClass = keys[values.index(max(values))]

    return predictedClass

def EvaluateModelAccuracy(model, X_test, y_test):
    # This function uses the trained model to make predictions over a test set, then calculates the accuracy
    predictions = []

    # Loop through all the test data
    for i in range(0,len(X_test)):

        # Make prediction for each new data point
        currentPrediction = PredictClass(model, list(X_test.iloc[i]))

        # Save whether or not the prediction was correct
        if currentPrediction == y_test.iloc[i]:
            predictions.append(1)
        else:
            predictions.append(0)

    # Calculate accuracy
    accuracy = sum(predictions)/len(X_test)
    return accuracy

if __name__ == '__main__':
    # Load the dataset
    dataFilepath = '/Users/account1/Documents/Blog/ML Algorithms/iris.data'
    columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(dataFilepath, names=columns)

    # Separate out the target variable
    X = dataset.drop('class', axis = 1)
    y = dataset['class']

    # Create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

    # Concatenate the target training variable back onto the training set
    trainingSet = pd.concat([X_train, y_train], axis = 1)

    # Train the model on training set
    model = TrainModel(trainingSet)

    # Use the trained model to make predictions on test set, and calculate accuracy of predictions
    modelAccuracy = EvaluateModelAccuracy(model, X_test, y_test)

    print(modelAccuracy)











    



