# file to revamp becca and laney's neural net from last semester
import os # for managing paths
from os.path import join
import matplotlib.pyplot as plt # for plots
from scipy.stats import sem # to compute standard error of mean
import numpy as np # for formatting data
import pandas as pd
from xgboost import XGBRegressor # the ml algorithm we're using
import keras # for neural nets

# read in and prepare data
DATA_PATH = 'S22_data'

def prepare_data(df_path_ls, random_seed, p): # do target prep is binary. p is what fraction is used for training
    print('preparing data.....')
    df_ls = []
    for df_path in df_path_ls:
        df_ls.append(pd.read_csv(join(DATA_PATH, df_path)))
    
    df = pd.concat(df_i for df_i in df_ls)
    df = df.sample(frac=1, random_state=random_seed).reset_index() # randomize rows

    # define inputs and outputs for model
    inputs = ['HH probability', 'HV probability', 'VH probability', 'VV probability',
        'DD probability', 'DA probability', 'AD probability', 'AA probability',
        'RR probability', 'RL probability', 'LR probability', 'LL probability']
    outputs = ['XY min', 'YZ min','XZ min']

    # function to simplify the outputs by replacing the most negative value with 1 and setting others to 0
    def simplify_targ(df):
        df_out = df.applymap(lambda x: 1 if x < 0 else 0)
        return df_out

    # function to build the dataset: p is proportion in training vs test; inputs is list of strings of input states, same for outputs
    def prep_data_targ(df, p, inputs, outputs):
        split_index = int(p*len(df))
        df_train = df.iloc[:split_index, :]
        df_test = df.iloc[split_index:, :]

        X_train = df_train[inputs]
        Y_train = simplify_targ(df_train[outputs])
        # Y_train = df_train[outputs]

        X_test = df_test[inputs]
        Y_test = simplify_targ(df_test[outputs])

        return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()

    # functions to build and train the network
    # split_frac = 0.8 # split some portion to be train vs test
   
    X_train, Y_train, X_test, Y_test = prep_data_targ(df, p, inputs, outputs)

    # save!
    np.save(join(DATA_PATH, 'x_train'), X_train)
    np.save(join(DATA_PATH,'y_train'), Y_train)
    np.save(join(DATA_PATH,'x_test'), X_test)
    np.save(join(DATA_PATH,'y_test'), Y_test)

# read in data
def read_data():
    X_train = np.load(join(DATA_PATH, 'x_train.npy'), allow_pickle=True)
    Y_train = np.load(join(DATA_PATH, 'y_train.npy'), allow_pickle=True)
    X_test = np.load(join(DATA_PATH, 'x_test.npy'), allow_pickle=True)
    Y_test = np.load(join(DATA_PATH, 'y_test.npy'), allow_pickle=True)

    return X_train, Y_train, X_test, Y_test

## call to load and save dfs ##
df_path_ls = []
for path in os.listdir(DATA_PATH):
    if path.endswith('.csv'):
        df_path_ls.append(path)

## code for neural network ##
def do_nn(X_train, Y_train, X_test, Y_test):

    import tensorflow as tf
    from keras import layers
    from keras.models import Sequential
    from keras.optimizers import Adam

    ## model creation ##
    def build_model_test(size=50, dropout=0.1, learning_rate=0.001):
        output_len = 3
        def witness_loss_fn(y_true, y_pred):
            # y_pred is a 1 by batch size tensor with all the predicted values for each state
            # y_true is a 3 by batch size tensor, with the three ground truth outputs per state
            
            # making y_pred 1 output when evaluating loss
            y_pred_xy = y_pred[0]
            y_pred_yz = y_pred[1]
            y_pred_xz = y_pred[2]

            # for y_true, getting labels
            xy_qual = y_true[0]
            yz_qual = y_true[1]
            xz_qual = y_true[2]

            loss = y_pred_xy*xy_qual + y_pred_yz*yz_qual + y_pred_xz*xz_qual

            return tf.reduce_mean(loss, axis=-1)

        model = Sequential()

        model.add(layers.Dense(size))
        model.add(layers.Dense(size))
        model.add(layers.Dense(size))
        model.add(layers.Dense(size))
        model.add(layers.Dense(size))

        model.add(layers.Dropout(dropout))

        # return len of class size
        model.add(layers.Dense(output_len))
        model.add(layers.Activation('softmax'))

        optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')

        return model

    def train_manual(X_train, Y_train, X_test, Y_test):
        global model
        model = build_model_test()
        
        #now run training
        history = model.fit(
        X_train, Y_train,
        batch_size = 100,
        validation_data=(X_test, Y_test),
        epochs=10,
        shuffle=True
        )
        return model

    ## call training ##
    model = train_manual(X_train, Y_train, X_test, Y_test)
    return model

## code for implementing xgboost model ##
def do_xgboost(X_train, Y_train, X_test, Y_test):
    # define the model
    model = XGBRegressor(n_estimators = 1000, random_state=0) 

    # fit the model
    model.fit(X_train, Y_train, early_stopping_rounds = 10, eval_set = [(X_test, Y_test)], verbose=False) 

    # make prediction
    # predictions = model.predict(X_test)
    # mae = mean_absolute_error(predictions, Y_test)
    # print('mae',mae)
    return model

## returns accuracy as defined by becca and laney in the spring write up ##
def evaluate_perf(model):
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
    Ud = Y_test.sum(axis=1) # undetectables: count the number of states w negative witness value
    return [N_correct_test / (len(Y_pred_test) - len(Ud[Ud==0])), N_correct_train / (len(Y_pred_train) - len(Ud[Ud==0]))] # return both the test and train results

## for testing single state ##
# prepare_data(df_path_ls, 0)
# X_train, Y_train, X_test, Y_test = read_data()

## initialize data ##
p_ls = [0.8 - 0.15*x for x in range(5)]
p_df = pd.DataFrame({'xgb_test_mean':[], 'xgb_test_sem': [],'xgb_train_mean':[], 'xgb_train_sem': [], 
'bl_test_mean':[], 'bl_test_sem': [],'bl_train_mean':[], 'bl_train_sem': []})
for j, p in enumerate(p_ls):
    # pick random states; test performance
    random_arr = np.random.randint(1,100, size=(1,50))
    xgb_test =[]
    xgb_train =[]
    bl_test = []
    bl_train = []

    ## load trained bl model ##
    json_file = open('models/model_qual_v2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_bl = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model_bl.load_weights("models/model_qual_v2.h5")
    for i, rand in enumerate(random_arr[0]):
        prepare_data(df_path_ls, rand, p)
        X_train, Y_train, X_test, Y_test = read_data()

        model_xgb = do_xgboost(X_train, Y_train, X_test, Y_test)

        # stats for xgboost
        frac_xgb = evaluate_perf(model_xgb)
        frac_bl = evaluate_perf(model_bl)
        print('fraction correct of xgb for seed '+str(rand)+', '+ str(i / len(random_arr[0])*100)+'%', frac_xgb[0], frac_xgb[1])
        xgb_test.append(frac_xgb[0])
        xgb_train.append(frac_xgb[1])
        bl_test.append(frac_bl[0])
        bl_train.append(frac_bl[1])

    # visualize performance
    xgb_test_mean = np.mean(xgb_test)
    xgb_test_sem = sem(xgb_test)
    xgb_train_mean = np.mean(xgb_train)
    xgb_train_sem = sem(xgb_train)
    print('mean for xgb:', xgb_test_mean, 'sem:', xgb_test_sem)
    bl_test_mean = np.mean(bl_test)
    bl_test_sem = sem(bl_test)
    bl_train_mean = np.mean(bl_train)
    bl_train_sem = sem(bl_train)
    print('mean for bl:', bl_test_mean, 'sem for bl:', bl_test_sem)

    xgb_test =np.array(xgb_test)
    xgb_train = np.array(xgb_train)
    bl_test =  np.array(bl_test)
    bl_train =  np.array(bl_train)

    np.save(join(DATA_PATH, 'accuracy', 'xgb_test_%i.npy'%j), xgb_test)
    np.save(join(DATA_PATH, 'accuracy', 'xgb_train_%i.npy'%j), xgb_train)
    np.save(join(DATA_PATH, 'accuracy', 'bl_test_%i.npy'%j), bl_test)
    np.save(join(DATA_PATH, 'accuracy', 'bl_train_%i.npy'%j), bl_train)

    plt.figure(figsize=(10,7))
    plt.scatter(random_arr[0], xgb_test, label='XGB Test')
    plt.scatter(random_arr[0], xgb_train, label='XGB Train')
    plt.scatter(random_arr[0], bl_test, label='BL Test')
    plt.scatter(random_arr[0], bl_train, label='BL Train')
    plt.xlabel('Random Seed', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Comparative Performance of XGB vs BL, p=%.3g'%p, fontsize=16)
    plt.legend()
    plt.savefig(join(DATA_PATH, 'accuracy','xgb_bl_compare_%i.pdf'%j))

    p_df = p_df.append({'xgb_test_mean':xgb_test_mean, 'xgb_test_sem': xgb_test_sem,'xgb_train_mean':xgb_train_mean, 'xgb_train_sem': xgb_train_sem, 
'bl_test_mean':bl_test_mean, 'bl_test_sem': bl_test_sem,'bl_train_mean':bl_train_mean, 'bl_train_sem': bl_train_sem}, ignore_index=True)

    print(str( j / len(p_ls) * 100)+' complete')

# plot overall performance as p varies
plt.figure(figsize=(10,7))
plt.errorbar(p_ls, p_df['xgb_test_mean'].values, yerr=xgb_test_sem, label='XGB Test')
plt.errorbar(p_ls, p_df['xgb_train_mean'].values, yerr=xgb_train_sem, label='XGB Train')
plt.errorbar(p_ls, p_df['bl_test_mean'].values, yerr=bl_test_sem, label='BL Test')
plt.errorbar(p_ls, p_df['bl_train_mean'].values, yerr=bl_train_sem, label='BL Train')
plt.xlabel('Proportion of Data $p$ for Training', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Performance of XGB vs BL For Various $p$', fontsize=16)
p_df.to_csv(join(DATA_PATH, 'accuracy', 'summary.csv'))

np.save(join(DATA_PATH, 'accuracy', 'random_arr.np'), random_arr)


# to save model; this saves the last one
# print('saving model using seed = %i with test accuracy %.3g'%(rand, frac_xgb[0]))
# model_xgb.save_model("models/model_xgb_%i_0.json"%rand) # first int is the seed; second is what version

# # to load model
# model_xgb = XGBRegressor() # create xgb object
# model_xgb.load_model("models/model_xgb_47_0.json")


# frac_bl = evaluate_perf(bl_model)
# print('fraction correct of bl model', frac_bl[0], frac_bl[1])

## print stats for neural net ##
# model = do_nn(X_train, Y_train, X_test, Y_test)
# frac_me = evaluate_perf(model)
# print('fraction correct of o nn', frac_me[0], frac_me[1])

