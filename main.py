# 0. Imports
from random import shuffle
from copy import deepcopy

# 1. Globals
useTensorFlow = True # if True we will use the tensorflow module as opposed to my NeuralNetwork module.

# 2. Helper functions
def read_data(fname,dlm=","):
    with open(fname,"r+") as f:
        return [[j.strip() for j in i.split(dlm) if j.strip() != ""] for i in f.readlines()]

def testtrain_split(data, fraction = 0.8):
    cols = ['ï»¿ID', 'Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases', 'Product_importance', 'Gender', 'Discount_offered', 'Weight_in_gms', 'Reached.on.Time_Y.N']
    test, train = [], []
    index = int(len(data)*fraction)
    for n, i in enumerate(data):
        row = [float(k) for k in [
            i[3], # customer calls
            i[5], # product cost
            i[6], # prior purchases
            i[-1], # reach on time [y/n]
            1 if int(i[4]) > 1 else 0, # customer rating int(i[4])
        ]]
        if n >= index:
            test.append(row)
        else:
            train.append(row)
    # print(train)
    return (test, train)

def xy_partition(test, train):
    # this function breaks the test/ train datasets into there X,Y components.
    trainX = [row[:-1] for row in train]
    trainY = [row[-1] for row in train]
    testX = [row[:-1] for row in train]
    testY = [row[-1] for row in train]
    return ((testX,testY),(trainX,trainY))

# zero rule algorithm for classification
def zero_rule_algorithm_classification(trainY, test_length):
    # used as a reference point for my neural networks.  How good and I doing relative to 
    # simple heuristics.  This heuristic makes its prediction based on the most frequent label
    # in the dataset.  Hard to beat if the dataset is highly disproportionate in favor of one label.
	prediction = max(set(trainY), key=trainY.count)
	return [prediction for i in range(test_length)]

# calculate the out of sample error
def compute_Eout(predictions, actual):
    correct = 0
    for i, j in zip(predictions, actual):
        i = i[0] if type(i) != type(1.1) else i # if a numpy array take first (only) element, else if float use i.
        # print(i, lookup[i], "|", j, lookup[j])
        correct += (1 if i == j else 0)
    return correct/ len(predictions)

# 3. main functions
def main():
    from NeuralNetwork import NeuralNetwork # my module
    # 1. read data and partition into test/ train sets.
    lookup = {0:"unsatisfied", 1: "satisfied"}
    raw_data = read_data("shipData.csv")
    #raw_data = raw_data[:len(raw_data)//6]
    shuffled_data = raw_data[1:]
    shuffle(shuffled_data)
    test, train = testtrain_split(shuffled_data, fraction = 0.9)
    print(len(test), len(train))
    
    # 2. initialize & train network.
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))

    #print("OUTPUTS: ", set([row[-1] for row in train]))
    NN = NeuralNetwork(n_inputs, 7, n_outputs)
    w0 = deepcopy(NN.network)
    NN.train_network(NN.network, train, 0.01, 20, n_outputs)

    correct = 0
    for tv in test:
        #rint("Customer satifaction: ")
        #print(tv)
        correct += 1 if NN.predict(NN.network, tv) == tv[-1] else 0
    print("out of sample accuracy: ",correct/len(test))

    print("zero rule Eout:")
    test, train = xy_partition(test, train)
    testX,testY = test
    trainX,trainY = train
    zpredictions = zero_rule_algorithm_classification(trainY, len(testY))
    Eout2 = compute_Eout(zpredictions, testY)
    print(f"out of sample error (zr): {Eout2:.4f}")
    print("learned weights: ", NN.network)
    print("\n\n")
    print("initial weights: ", w0)

def tensor_main():
    import tensorflow as tf # adding import here becuase of the annoying dl open messages.
    from tensorflow.keras.callbacks import LambdaCallback
    print("1: tensorflow import success ->")
    # 1. read data and partition into test/ train sets.
    lookup = {0:"unsatisfied", 1: "satisfied"}
    raw_data = read_data("shipData.csv")
    #raw_data = raw_data[:len(raw_data)//6]
    shuffled_data = raw_data[1:]
    shuffle(shuffled_data)
    test, train = testtrain_split(shuffled_data)
    print(len(test), len(train))
    
    print("2: data read success")
    # 2. initialize & train network.
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

    # 3. define model architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(7, input_dim=n_inputs, activation='relu')) #relu
    model.add(tf.keras.layers.Dense(n_inputs, activation='relu')) #relu
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #sigmoid for binary classifcation
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'],)
    w0 = deepcopy(model.weights)
    print("3: model create success")
    test, train = xy_partition(test, train)
    testX,testY = test
    trainX,trainY = train
    model.fit(trainX, trainY, epochs=50, batch_size=10, callbacks = [print_weights])
    print("4: model fit success")

    predictions = model.predict_classes(testX)
    print("5: attempting to predict")
    Eout1 = compute_Eout(predictions, testY)
    print(f"out of sample error (nn): {Eout1:.4f}")

    print("zero rule Eout:")
    zpredictions = zero_rule_algorithm_classification(trainY, len(testY))
    Eout2 = compute_Eout(zpredictions, testY)
    print(f"out of sample error (zr): {Eout2:.4f}")
    print(model.summary())

    print("Initial weights: ", model.weights)
    print("\n\n")
    print("Learned weights: ", model.weights)
    

if __name__ == "__main__": tensor_main() if useTensorFlow else main()