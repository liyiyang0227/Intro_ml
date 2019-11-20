import numpy as np
import pickle
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from keras.optimizers import Adadelta
import keras.backend as K
from sklearn.metrics import accuracy_score
from keras.metrics import binary_accuracy
class ClaimClassifier:
    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.model = self.create_model()
        self.history = None
    def create_model(self,optimizer='adadelta', init='normal',dropout_rate = 0.5):
        model = Sequential()
        model.add(Dense(8, kernel_initializer=init,input_dim=9, activation='relu'))
        # model.add(Dropout(dropout_rate))
        # model.add(Dense(16,kernel_initializer=init,activation='relu'))
        # model.add(Dropout(dropout_rate))
        # model.add(Dense(8, kernel_initializer=init, activation='relu'))
        # model.add(Dropout(dropout_rate))
        model.add(Dense(8,kernel_initializer=init,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))

        model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        return model
    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        X: numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE
        # Standardization
        clean_data = preprocessing.scale(X_raw)
        clean_data = preprocessing.normalize(clean_data)
        return clean_data

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded
        y_raw : numpy.ndarray (optional)
            A one dimensional numpy array, this is the binary target variable
        Returns
        -------
        ?
        """
        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_raw),
                                                          y_raw)
        class_weights_dict = dict(enumerate(class_weights))
        print(class_weights_dict)
        self.history = self.model.fit(X_clean,y_raw,nb_epoch=50,batch_size=100,class_weight=class_weights_dict,validation_split=0.2)

        # YOUR CODE HERE
    def fit_skl(self,X,Y):
        seed = 1997
        np.random.seed(seed)
        x = self._preprocessor(X)
        # evaluate using 10-fold cross validation
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(Y),
                                                          Y)
        class_weights_dict = dict(enumerate(class_weights))
        self.model = KerasClassifier(build_fn=self.create_model, nb_epoch=10, batch_size=32)
        self.model.fit(x,Y,class_weight = class_weights_dict,nb_epoch=20, batch_size=100,shuffle = True)
    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        y_pred = self.model.predict(X_clean)
        #y_pred = self.model.predict_classes(X_clean)
        return  y_pred

    def evaluate_architecture(self,X,Y):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        acc_values = self.history.history['acc']
        val_acc_values = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(acc_values) + 1)
        # bo for blue dot
        plt.plot(epochs, loss, 'b', label='trainning loss')
        plt.plot(epochs, val_loss, 'r', label='validating loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.clf()
        plt.plot(epochs, acc_values, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc_values, 'r', label='validating accuracy')
        plt.title('Training and validating accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # seed = np.random.seed(520)
        # X = self._preprocessor(X)
        # # evaluate using 10-fold cross validation
        # model = KerasClassifier(build_fn=self.model, nb_epoch=200, batch_size=32)
        # kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)
        # results = cross_val_score(model, X, Y, cv=kfold)
        # print(results.mean())

    def save_model(self):
        with open("part2_claim_classifier.pickle", "wb") as target:
            pickle.dump(self, target)


def ClaimClassifierHyperParameterSearch(X,Y):  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    nn = ClaimClassifier()
    model = KerasClassifier(build_fn=nn.create_model)
    X_clean = nn._preprocessor(X)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y),
                                                      Y)
    print(class_weights)
    class_weights_dict = dict(enumerate(class_weights))
    # grid search epochs, batch size and optimizer
    optimizers = ['SGD', 'RMSprop', 'Adagrad','Adam','Nadam']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = np.array([20,50,100,1000])
    batches = np.array([25, 50, 100])
    #dropout_rate = [0.0, 0.2, 0.5, 0.9]
    param_grid = dict(nb_epoch=epochs, batch_size=batches, init=init,optimizer = optimizers)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_clean, Y,class_weight=class_weights_dict)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.cv_results_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

    return  grid_result.best_params_
