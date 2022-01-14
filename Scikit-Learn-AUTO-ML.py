# * Importing libraries
import pandas as pd
from joblib import dump

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# @ Classifiers!
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.kernel_approximation import RBFSampler # Kernal approximation

# @ Regressors!
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

"""### Defining Functions ###"""

# @ Function to get categorical features
def get_categorical_features(dataset):
    print("Scaning columns")
    catergorical_features = []

    for col in dataset.columns:
        try:
            for i in range(len(col)):
               dataset.loc[i, col].astype(float)

            print("{} is a numerical column".format(col))
        except:
            print("{} is not a numerical column".format(col))
            catergorical_features.append(col)

    return catergorical_features

# @ Function to get common elements out of 2 lists
def common_elements(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# @ Function to check missing values
def check_missing_vals(dataset):
    print("Checking for missing values")
    m = dataset.isna().sum()

    m_col = []
    for col in dataset.columns:
        if(m.get(col)):
            print("There are missing values in {}".format(col))
            m_col.append(col)
        else:
            print("There are no missing values in {}".format(col))

    return m_col

# @ Function to check percentage of missing data
def check_percentage_of_missing_data_column(col):
    m = dataset[col].isna().sum()
    p = (m/len(dataset[col])) * 100
    print("{:.2f}% Data is missing in {} column".format(p,col))

    return p

# @ Function to fill missing values
def fill_missing_values(dataset, m_col, target):
    if m_col:
        an = str(input("Do you want the AUTOML to FIX it?? Enter yes or no\n"))

        if (an == "yes"):

            catergorical_features = []

            for col in dataset.columns:
                try:
                    for i in range(len(col)):
                        dataset.loc[i, col].astype(float)

                except:
                    catergorical_features.append(col)

            print("We will handle this")
            print("Remove rows with missing target values")
            dataset.dropna(subset=[target], inplace=True)

            cat_m_cols = common_elements(m_col, catergorical_features)

            num_m_cols = [k for k in m_col if k not in cat_m_cols]

            for col in dataset.columns:
                p = check_percentage_of_missing_data_column(col)

                if(p>50.00):
                    print("Terminating {} column has it's {}% values are missing".format(col,p))

                    dataset.drop(col,axis=1,inplace=True)

                    try:
                       cat_m_cols.remove(col)
                    except:
                        try:
                            num_m_cols.remove(col)
                        except:
                            print("Facing an issue with {} column".format(col))

            try:
                num_m_cols.remove(target)
            except:
                pass

            # * Fill the Numerical columns

            for col in num_m_cols:
                dataset[col].fillna(dataset[col].mean(),inplace=True)


            # * Fill the Categorical columns

            for col in cat_m_cols:
                dataset[col].fillna("missing", inplace=True)

            return dataset

        else:
            print("All the best Bye!")
            quit()

    else:
        print("Your data doesn't has any missing values")
        return dataset

# @ Function to check the problem type
def check_problem_type(dataset, target_col):
    for i in range(len(dataset[target_col])):
        t = dataset.loc[i, target_col]

        if (t <= 1):
            problem_type = "Looks like a classification problem"
        elif (t > 1):
            problem_type = "Looks like a regression problem"
        elif (t <= 1.0):
            problem_type = "Looks like a classification problem"
        elif (t > 1.0):
            problem_type = "Looks like a regression problem"
        else:
            problem_type = "Looks like a clustering problem"

    print(problem_type)
    return problem_type

# @ Function to get the important features
def recursion_function_ab(imp_col,a):


    if (imp_col<3):

        a += 5.00

        for col in dataset.columns:
            p = check_percentage_of_missing_data_column(col)

            if (p < a):
                imp_col.append(col)

    else:
        return imp_col

    recursion_function_ab(imp_col,a)
    return imp_col

def get_imp_features(dataset):
    imp_col = []
    pp = 00.00

    for col in dataset.columns:
        p = check_percentage_of_missing_data_column(col)

        if (p == pp):
            imp_col.append(col)

    imp_col = recursion_function_ab(imp_col,pp)

    return len(imp_col)

### ? Welcoming

import warnings
warnings.filterwarnings("ignore")

print("Welcome to AUTOML")

# It can't process audios,images and videos

data_path = input("Enter the path where data is stored!\n")

data_path = data_path.replace("/","//")

print("Scaning Given data")
dataset = pd.read_csv(data_path, low_memory=False)

print("Calculating the Total Number of Samples")
samples = len(dataset)
print("There are {} Samples in the Given Dataset".format(samples))

# * Attaining the target column

target_column = input("Enter the Name of Target Column\n")

# * Checking for missing values

m_col =check_missing_vals(dataset)

dataset = fill_missing_values(dataset,m_col,target_column)

# * Defining x and y

print("Creating X(Feature Matrix) and Y(Labels)")

x = dataset.drop(target_column, axis=1)
y = dataset[target_column]

# * Turn the categories into numbers

catergorical_features = get_categorical_features(dataset)

transformed_dataset = 0

if catergorical_features:
    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                      one_hot,
                                      catergorical_features)],
                                    remainder="passthrough")

    x = transformer.fit_transform(x)

    print("All the non-numerical columns are now One hot encoded")
else:
    print("All the columns are numerical")

# * Split into training and test

print("Splitting data into training and test")

x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

print("Splitting data completed")

if(samples < 50):
    print("AUTOML requires more data to create an efficient machine learning model")
    quit()
else:
    print("Selecting a machine learning model based on the given data")

fpt = check_problem_type(dataset,target_column)

if(fpt == "Looks like a classification problem"):
    problem_type = "predicting_a_category"
    Data_type = "labeled_data"

elif(fpt == "Looks like a regression problem"):
    problem_type = "predicting_a_quantity"

elif(fpt == "Looks like a clustering problem"):
    problem_type = "predicting_a_quantity"
    Data_type = "unlabeled_data"

if(problem_type == "predicting_a_category"):
    if(Data_type == "labeled_data"):
        if(samples < 100000):

            try:
                print("Preparing Linear SVC Model")
                model = svm.LinearSVC()

                print("Training Linear SVC Model")

                model.fit(x_train,y_train)
                accuracy = model.score(x_test, y_test) * 100

            except:
               print("Unable to Train Linear SVC Model")

            if(accuracy<80):
                print("Linear SVC Model is showing {}% accuracy".format(accuracy))
                print("Performing Detailed Data Analysis")

                data = None

                if catergorical_features:
                    data = "text"

                if(data == "text"):

                    try:
                        print("Preparing Naive Bayes Model")
                        model = GaussianNB()

                    except:
                        print("Unable to prepare Naive Bayes Model")

                    try:
                        print("Training Naive Bayes Model")
                        model.fit(x_train, y_train)

                    except:
                        print("Unable to Train Naive Bayes Model")

                    accuracy = model.score(x_test, y_test) * 100
                    print("Model is showing {}% accuracy".format(accuracy))
                    print("Saving Model to the Desired Directory")
                    dump(model, filename="Naive Bayes Model.joblib")

                else:

                    try:
                        print("Preparing KNeighbors Classifier Model")
                        model = KNeighborsClassifier()

                    except:
                        print("Unable to prepare KNeighbors Classifier Model")

                    try:
                        print("Training KNeighbors Classifier Model")
                        model.fit(x_train, y_train)

                    except:
                        print("Unable to Train KNeighbors Classifier Model")

                    accuracy = model.score(x_test, y_test) * 100

                    if (accuracy < 80):
                        print("KNeighbors Classifier Model is showing {}% accuracy".format(accuracy))
                        print("Training SVC Ensemble Classifier Model")

                        try:
                            print("Preparing SVC Ensemble Classifier Model(RandomForestClassifier)")
                            model = RandomForestClassifier()

                        except:
                            print("Unable to prepare SVC Ensemble Classifier Model(RandomForestClassifier)")

                        try:
                            print("Training SVC Ensemble Classifier Model(RandomForestClassifier)")
                            model.fit(x_train, y_train)

                        except:
                            print("Unable to Train SVC Ensemble Classifier Model(RandomForestClassifier)")

                        print("Model is showing {}% accuracy".format(accuracy))
                        print("Saving Model to the Desired Directory")
                        dump(model, filename="SVC Ensemble Classifier Model(RandomForestClassifier).joblib")

                    else:
                        print("Model is showing {}% accuracy".format(accuracy))
                        print("Saving Model to the Desired Directory")
                        dump(model, filename="KNeighbors Classifier Model.joblib")

            else:
                print("Model is showing {}% accuracy".format(accuracy))
                print("Saving Model to the Desired Directory")
                dump(model, filename="Linear SVC Model.joblib")

        else:

            try:
                print("Preparing SGD Classifier Model")
                model = SGDClassifier()

            except:
                print("Unable to prepare SGD Classifier Model")

            try:
                print("Training SGD Classifier Model")
                model.fit(x_train,y_train)

            except:
               print("Unable to Train SGD Classifier Model")


            accuracy = model.score(x_test,y_test) * 100

            if (accuracy < 80):
                print("SGD Classifier Model is showing {}% accuracy".format(accuracy))

                try:
                    print("Preparing for kernal approximation")
                    rbf_feature = RBFSampler(gamma=1, random_state=1)

                except:
                    print("Unable to prepare Kernal approximation")

                try:
                    print("Performing Kernal approximation")
                    x_features = rbf_feature.fit_transform(x)

                    x_features_train , x_features_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

                except:
                    print("Unable to Perform Kernal approximation")

                try:
                    print("Preparing SGD Classifier Model after Kernal approximation")
                    model = SGDClassifier()

                except:
                    print("Unable to prepare SGD Classifier Model")

                try:
                    print("Training SGD Classifier Model after Kernal approximation")
                    model.fit(x_features_train, y_train)

                except:
                    print("Unable to Train SGD Classifier Model")

                accuracy = model.score(x_features_test, y_test) * 100
                print("Model is showing {}% accuracy".format(accuracy))
                print("Saving Model to the Desired Directory")
                dump(model, filename="SGD Classifier Model.joblib")

            else:
                print("Model is showing {}% accuracy".format(accuracy))
                print("Saving Model to the Desired Directory")
                dump(model, filename="SGD Classifier Model.joblib")

    else:
        print("Clustering will be done using tensorflow")

elif (problem_type == "predicting_a_quantity"):
    if (samples > 100000):

        inp = input(
            "if you know the number of important features in the dataset please write the number and if u don't just write no\n")
        try:
            imp_features = int(inp)
        except:
            print("Looking for important features")
            try:
                imp_features = get_imp_features(dataset)
                print("{} important features found".format(imp_features))
            except:
                print("Unable to find important features")

        if(imp_features<5):

            print("Preparing Lasso ELasticNet")
            model = linear_model.Lasso()

            print("Unable to prepare Lasso ELasticNet")

            print("Training Lasso ELasticNet")
            model.fit(x_train,y_train)

            accuracy = model.score(x_test,y_test)

            print("Model is showing {}% accuracy".format(accuracy))
            print("Saving Model to the Desired Directory")
            dump(model, filename="Lasso ELasticNet.joblib")

        else:

            print("Preparing RidgeRegression SVR(kernal='linear')")
            model = linear_model.Ridge()

            print("Training RidgeRegression SVR(kernal='linear')")
            model.fit(x_train, y_train)

            accuracy = model.score(x_test, y_test)

            if (accuracy < 80):

                print("RidgeRegression Model is showing {}% accuracy".format(accuracy))

                print("Preparing RandomForest Regressor")

                reg1 = GradientBoostingRegressor(random_state=1)
                reg2 = RandomForestRegressor(random_state=1)
                reg3 = LinearRegression()

                model = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])

                print("Training SVR(kernal='rbf') EnsembleRegressors")
                model.fit(x_train, y_train)

                accuracy = model.score(x_test, y_test)

                print("Model is showing {}% accuracy".format(accuracy))
                print("Saving Model to the Desired Directory")
                dump(model, filename="SVR EnsembleRegressors.joblib")

            else:
                print("Model is showing {}% accuracy".format(accuracy))
                print("Saving Model to the Desired Directory")
                dump(model, filename="RidgeRegression.joblib")

    else:
        model = SGDRegressor()
        model.fit(x_train, y_train)
        print("Training SGD Regressor")

        accuracy = model.score(x_test, y_test)

        print("Model is showing {}% accuracy".format(accuracy))
        print("Saving Model to the Desired Directory")
        dump(model, filename="SGD Regressor.joblib")