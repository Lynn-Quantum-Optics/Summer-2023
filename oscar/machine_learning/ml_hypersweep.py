# file to perform hyperparameter sweep for XGBOOST and NN

import wandb
import numpy as np
from xgboost import XGBClassifier
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam

## compute accuracy ##
def evaluate_perf(model, X_train, Y_train, X_test, Y_test):
    ''' Function to measure accuracy on both train and test data.
    Params:
        model: trained model
        method: 'witness' or 'entangled' for prediction method
        data: 'train' or 'test' for which data to evaluate on
        '''
    Y_pred_test = model.predict(X_test)
    Y_pred_train = model.predict(X_train)

    N_correct_test = 0
    N_correct_train = 0
    for i, y_pred in enumerate(Y_pred_test):
        if Y_test[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
            N_correct_test+=1
    
    for i, y_pred in enumerate(Y_pred_train):
        if Y_train[i][np.argmax(y_pred)]==1: # if the witness is negative, i.e. detects entanglement
            N_correct_train+=1

    # if method == 'witness':
    #     Ud = Y_test.sum(axis=1) # undetectables: count the number of states w negative witness value
    #     return [N_correct_test / (len(Y_pred_test) - len(Ud[Ud==0])), N_correct_train / (len(Y_pred_train) - len(Ud[Ud==0]))]
    # elif method == 'entangled':
    #     return [N_correct_test / len(Y_pred_test), N_correct_train / len(Y_pred_train)]
    return [N_correct_test / len(Y_pred_test), N_correct_train / len(Y_pred_train)]


#######################################################
## XGBOOST ##
''' Total list of params: https://xgboost.readthedocs.io/en/stable/parameter.html'''

xgb_sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "max_depth": {"distribution": "int_uniform", "min":  1, "max": 10},
        "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 0.9},
        "n_estimators": {"distribution": "int_uniform", "min":  500, "max": 5000},
        "early_stopping": {"distribution": "int_uniform", "min": 5, "max": 30}
    },
    }

def train_xgb():
    ''' Function to run wandb sweep for XGBOOST. 
    Adapted from https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py 
    params:
        method: 'witness' or 'entangled' for prediction method
        data: 'train' or 'test' for which data to evaluate on
    '''
    wandb.init(config=xgb_sweep_config) # initialize wandb client

    # define the model
    model = XGBClassifier(
        max_depth=wandb.config.max_depth,
        learning_rate=wandb.config.learning_rate,
        n_estimators=wandb.config.n_estimators,
        # max_depth=int(wandb.config.max_depth),
        # learning_rate=wandb.config.learning_rate,
        # n_estimators=int(wandb.config.n_estimators),
    )

    # fit the model
    model.fit(X_train, Y_train, early_stopping_rounds = wandb.config.early_stopping, eval_set=[(X_test, Y_test)], callbacks=[wandb.xgboost.WandbCallback(log_model=True)])
    # early_stopping_rounds = int(wandb.config.early_stopping)
    # log test accuracy to wandb
    val_acc = evaluate_perf(model, X_train, Y_train, X_test, Y_test)[0]
    wandb.log({"val_acc": val_acc})

def custom_train_xgb(method, X_train, Y_train, X_test, Y_test, max_depth=6, learning_rate=0.3, n_estimators=1000, early_stopping=10):
    ''' Function to run XGBOOST with custom hyperparameters.
    params:
        method: 'witness' or 'entangled' for prediction method
        data: 'train' or 'test' for which data to evaluate on
    '''
    # define the model
    model = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators= n_estimators,
    )

    # fit the model
    model.fit(X_train, Y_train, tree_method='hist', early_stopping_rounds = early_stopping, random_state=42, eval_set=[(X_test, Y_test)])

    # print results
    acc = evaluate_perf(model, method, X_train, Y_train, X_test, Y_test)
    print('Accuracy on test, train', acc)

#######################################################
## NN ##
nn3h_sweep_config = {
    'method': 'random',
    'name': 'val_accuracy',
    'goal': 'maximize',
'parameters':{
    'epochs': {
       'distribution': 'int_uniform',
       'min': 20,
       'max': 100
    },
    # for build_dataset
     'batch_size': {
       'values': [x for x in range(32, 161, 32)]
    },
    'size_1': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'size_2': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'size_3': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256,
    },
    'dropout': {
      'distribution': 'uniform',
       'min': 0,
       'max': 0.6
    },
    'learning_rate':{
         #uniform distribution between 0 and 1
         'distribution': 'uniform', 
         'min': 0,
         'max': 0.1
     }
},
}


nn5h_sweep_config = {
    'method': 'random',
    'name': 'val_accuracy',
    'goal': 'maximize',
    'metric':'val_accuracy',
'parameters':{
    'epochs': {
       'distribution': 'int_uniform',
       'min': 20,
       'max': 100
    },
    # for build_dataset
     'batch_size': {
       'values': [x for x in range(32, 161, 32)]
    },
    'size_1': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'size_2': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'size_3': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'size_4': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'size_5': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'dropout': {
      'distribution': 'uniform',
       'min': 0,
       'max': 0.6
    },
    'learning_rate':{
         #uniform distribution between 0 and 1
         'distribution': 'uniform', 
         'min': 0,
         'max': 0.1
     }
},
}

def train_nn3h():
    ''' Function to run wandb sweep for NN.'''
    
    wandb.init(config=nn3h_sweep_config) # initialize wandb client

    
    def build_model(size1, size2, size3, dropout, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size1))
        model.add(layers.Dense(size2))
        model.add(layers.Dense(size3))

        model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('softmax'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        return model
    
    model = build_model(wandb.config.size_1,  wandb.config.size_2, wandb.config.size_3, 
              wandb.config.dropout, wandb.config.learning_rate)
    
    # now train
    history = model.fit(
        X_train, Y_train,
        batch_size = wandb.config.batch_size,
        validation_data=(X_test,Y_test),
        epochs=wandb.config.epochs,
        callbacks=[wandb.keras.WandbCallback()] #use callbacks to have w&b log stats; will automatically save best model                     
      )

def train_nn5h():
    ''' Function to run wandb sweep for NN.'''
    
    wandb.init(config=nn5h_sweep_config) # initialize wandb client

    
    def build_model(size1, size2, size3, size4, size5, dropout, learning_rate):
        model = Sequential()

        model.add(layers.Dense(size1))
        model.add(layers.Dense(size2))
        model.add(layers.Dense(size3))
        model.add(layers.Dense(size4))
        model.add(layers.Dense(size5))

        model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(len(Y_train[0])))
        model.add(layers.Activation('softmax'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        return model
    
    model = build_model(wandb.config.size_1,  wandb.config.size_2, wandb.config.size_3, 
              wandb.config.size_4, wandb.config.size_5, 
              wandb.config.dropout, wandb.config.learning_rate)
    
    # now train
    history = model.fit(
        X_train, Y_train,
        batch_size = wandb.config.batch_size,
        validation_data=(X_test,Y_test),
        epochs=wandb.config.epochs,
        callbacks=[wandb.keras.WandbCallback()] #use callbacks to have w&b log stats; will automatically save best model                     
      )


## run on our data ##
if __name__=='__main__':
    import pandas as pd
    from os.path import join

    from train_prep import prepare_data

    # load data here
    JS_DATA_PATH = 'jones_simplex_data'
    X_train, Y_train, X_test, Y_test, _ , _ = prepare_data(datapath=join(JS_DATA_PATH, 'simplex_100000_0.csv'), savename='simp_100k', method='witness_s')
    
    def run_sweep(wtr=0):
        ''' Function to run hyperparam sweep.
        Params:
            wtr: int, 0 for XGB, 1 for NN5H, 2 for NN3H
        '''
        if wtr==0:
            sweep_id = wandb.sweep(xgb_sweep_config, project="Lynn Quantum Optics3")
            wandb.agent(sweep_id=sweep_id, function=train_xgb)
        elif wtr==1:
            sweep_id = wandb.sweep(nn5h_sweep_config, project="Lynn Quantum Optics-NN5h")
            wandb.agent(sweep_id=sweep_id, function=train_nn5h)
        elif wtr==2:
            sweep_id = wandb.sweep(nn5h_sweep_config, project="Lynn Quantum Optics-NN3h")
            wandb.agent(sweep_id=sweep_id, function=train_nn3h)
        else:
            raise ValueError('wtr must be 0, 1, or 2.')
    wtr = int(input('Enter 0 for XGB, 1 for NN5H, 2 for NN3H:'))
    run_sweep(wtr)