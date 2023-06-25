import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.debugging.experimental.disable_dump_debug_info
#-----------------------------------------------------------------------------------------#
from keras import Sequential
from keras import layers
from tensorflow import keras
from sklearn import metrics
import numpy as np
import argparse
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description="FeGAN OD")
    parser.add_argument("--path", default="../Resources/Datasets/Arrhythmia_withoutdupl_norm_02_v01.arff",
                        help="Data path")
    parser.add_argument("--lr_gen", type=float, default=0.01, help="Learning rate generator")
    parser.add_argument("--lr_dis", type=float, default=0.01, help="Learning rate discriminator")
    parser.add_argument("--stop_epochs", type=int, default = 30, help="Generator stops training after stop_epochs")
    parser.add_argument("--k", type=int, default=30 , help="Number of discriminators")
    
    return parser.parse_args()

'''
    Calculate AUC and print it
'''
def AUC(truth, decision):
    output = metrics.roc_auc_score(truth, decision)
    print("AUC: " + str(output))
    return output

def set_seed(seed):
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    keras.utils.set_random_seed(seed) #seeds numpy, random and tf all at once
    os.environ["PYTHONHASSEED"] = str(seed)

#-----------------------------------------------------------------------------------------#


'''
    Create the generator. Uses two dense layers and relu activatoin
'''
def create_gen():
    generator = Sequential()
    generator.add(layers.Dense(latent_size, input_dim=latent_size, activation="relu", kernel_initializer=keras.initializers.Identity(gain=1.0)))
    generator.add(layers.Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    latent = keras.Input(shape=(latent_size,))
    fake_data = generator(latent)
    return keras.Model(latent, fake_data)

'''
    Create the discriminator. USes two dense layers and relu activation
'''
def create_dis(sub_size):
    discriminator = Sequential()
    discriminator.add(layers.Dense(np.ceil(np.sqrt(data_size)), input_dim=sub_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    discriminator.add(layers.Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    data = keras.Input(shape=(sub_size,))
    fake = discriminator(data)
    return keras.Model(data, fake)

def load_data():
    arff_data = arff.loadarff(args.path)
    df = pd.DataFrame(arff_data[0])
    df["outlier"] = pd.factorize(df["outlier"], sort=True)[0] #maybe flip
    data_x = df.iloc[:,:-2]
    data_y = df.iloc[:,-1]
    
    return data_x, data_y

'''
    Plot the loss of the models. Generator in blue. AUC in Yellow
'''
def plot(train_history):
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    auc_y = train_history['auc']
    for i in range(k):
        names['dy_' + str(i)] = train_history['sub_discriminator{}_loss'.format(i)]
    x = np.linspace(1, len(gy), len(gy))
    fig, ax = plt.subplots()
    ax.plot(x, gy, color='blue')
    ax.plot(x, dy,color='red')
    ax.plot(x, auc_y, color='yellow', linewidth = '3')
    for i in range(k):
        ax.plot(x, names['dy_' + str(i)], color='green', linewidth='0.5')
    plt.show()
    
'''
    Randomly draw subspaces for each sub_discriminator. Store them in names[]
'''
def draw_subspaces(dimension, k):
    dims = random.sample(range(1,dimension), k)
    for i in range(k):
        names["subspaces"+str(i)] = random.sample(range(dimension), dims[i])
    

if __name__ == '__main__':
    set_seed(777)
    train = True
    
    args = parse_arguments()
    data_x, data_y = load_data()
    data_size = data_x.shape[0] # n := number of samples
    latent_size = data_x.shape[1] # dimension of the data set
    
    if train:
        train_history = defaultdict(list)
        names = locals()
        epochs = args.stop_epochs * 3
        stop = 0
        k = args.k

        
        generator = create_gen()
        generator.compile(optimizer=keras.optimizers.SGD(learning_rate=args.lr_gen), loss='binary_crossentropy')
        latent = keras.Input(shape=(latent_size,))
        create = 0 # used to initialize sum of the sub_discriminators
        
        draw_subspaces(latent_size,k)
        
        names["sub_discriminator_sum"] = 0
        # create sub_discriminators, sum is used to then take the average of all sub_discriminators' decisions
        for i in range(k):
            names["sub_discriminator" + str(i)] = create_dis(len(names["subspaces"+str(i)]))
            names["fake" + str(i)] = generator(latent) # generate the fake data of the generator
            #names["sub_discriminator" + str(i)].trainable = False
            print(i)
            names["fake" + str(i)] = names["sub_discriminator" + str(i)](tf.gather(names["fake"+str(i)],names["subspaces"+str(i)],axis=1))
            names["sub_discriminator_sum"] += names["fake" + str(i)]
            names["sub_discriminator" + str(i)].compile(optimizer=keras.optimizers.SGD(learning_rate=args.lr_dis), loss='binary_crossentropy')
            
        names["sub_discriminator_sum"] /= k
        names["combine_model"] = keras.Model(latent, names["sub_discriminator_sum"]) # model with the average decision. Used to train the generator.
        names["combine_model"].compile(optimizer=keras.optimizers.SGD(learning_rate=args.lr_gen), loss='binary_crossentropy')
        
        
        for epoch in range(epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(500, data_size)
            num_batches = int(data_size / batch_size)
        
            for idx in range(num_batches):
                print('\nTesting for epoch {} index {}:'.format(epoch + 1, idx + 1))

                # Generate noise
                noise_size = batch_size
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                
                data_batch = data_x[idx * batch_size: (idx + 1) * batch_size]
                
                names["generated_data"] = generator.predict(noise, verbose = 1)
                
                X = np.concatenate((data_batch, names["generated_data"]))
                Y = np.array([1] * batch_size + [0] * int(noise_size)) # 1 real, fake
                
                discriminator_loss = 0
                for i in range(k):
                    names["sub_discriminator" + str(i) + "_loss"] = names["sub_discriminator" + str(i)].train_on_batch(X[:,names["subspaces"+str(i)]],Y)
                    train_history['sub_discriminator{}_loss'.format(i)].append(names['sub_discriminator' + str(i) + '_loss'])
                    discriminator_loss += names["sub_discriminator" + str(i) + "_loss"]
                discriminator_loss /= k
                train_history["discriminator_loss"].append(discriminator_loss)
                    
                counter = 0
                for i in range(k):
                    if counter == 0:
                        p_value = names["sub_discriminator" + str(i)].predict(data_x.to_numpy()[:,names["subspaces"+str(i)]])
                    else:
                        p_value += names["sub_discriminator" + str(i)].predict(data_x.to_numpy()[:,names["subspaces"+str(i)]])
                        
                p_value /= k
                
                if stop == 0:
                    trick = np.array([1] * noise_size)
                    generator_loss = names["combine_model"].train_on_batch(noise, trick)
                    train_history['generator_loss'].append(generator_loss)
                else:
                    trick = np.array([1] * noise_size)
                    generator_loss = names["combine_model"].evaluate(noise, trick)
                    train_history['generator_loss'].append(generator_loss)

                if epoch +1 > args.stop_epochs:
                        stop = 1
                
            data_y = pd.DataFrame(data_y)
            result = np.concatenate((p_value,data_y), axis=1)
            result = pd.DataFrame(result, columns=["p","y"])
            result = result.sort_values("p", ascending=True)
            
            inlier_parray = result.loc[lambda df: df.y == 0, 'p'].values
            outlier_parray = result.loc[lambda df: df.y == 1, 'p'].values
            sum = 0.0
            for o in outlier_parray:
                for i in inlier_parray:
                    if o < i:
                        sum += 1.0
                    elif o == i:
                        sum += 0.5
                    else:
                        sum += 0
            print(result)
            print(len(inlier_parray))
            print(len(outlier_parray))
            AUC = '{:.4f}'.format(sum / (len(inlier_parray) * len(outlier_parray)))
            for i in range(num_batches):
                train_history['auc'].append((sum / (len(inlier_parray) * len(outlier_parray))))
            print('AUC:{}'.format(AUC))

    plot(train_history)