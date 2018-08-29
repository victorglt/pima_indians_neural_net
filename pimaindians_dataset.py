import pandas as pd

def normalize(x_train, x_test):
    mean = x_train.mean(axis=0)
    x_train -= mean
    std = x_train.std(axis=0)
    x_train /= std

    x_test -= mean
    x_test /= std


def load_data(path='diabetes.csv', test_percentage=0.5,
              relevant_rows=None):
    """ Loads the Pima Indians diabetes dataset 
        # Returns
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
            :type test_percentage: percentage of entries to use for test
    """

    if relevant_rows is None:
        relevant_rows = ["Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness",
                         "SerumInsulin", "BMI", "DiabetesPedigree", "Age", "Diabetic"]
    dataset = pd.read_csv(open(path, "rb"))[relevant_rows]

    # Get a percentage sample from the full dataset to use as testing
    test_dataset = dataset.sample(frac=test_percentage)

    # Training dataset will be the full dataset - the selected test samples
    train_dataset = dataset.drop(test_dataset.index)

    # The class to predict
    y_train = train_dataset["Diabetic"]
    y_test = test_dataset["Diabetic"]

    # Exclude the class to predict from the features
    x_train = train_dataset.drop(columns=["Diabetic"])
    x_test = test_dataset.drop(columns=["Diabetic"])

    # Normalize features to a mean
    normalize(x_train, x_test)

    return (x_train.values, y_train.values), (x_test.values, y_test.values)
