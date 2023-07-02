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
from datetime import date
import csv

def parse_arguments():
    parser = argparse.ArgumentParser(description="FeGAN OD")
    parser.add_argument("--gpu", type=int,default=0)
    parser.add_argument("--data", default="C")
    parser.add_argument("--inlier", type=int, default=0)
    #parser.add_argument("--path", default="../Resources/Datasets/Arrhythmia_withoutdupl_norm_02_v01.arff",
    #                    help="Data path")
    #parser.add_argument("--lr_gen", type=float, default=0.01, help="Learning rate generator")
    #parser.add_argument("--lr_dis", type=float, default=0.01, help="Learning rate discriminator")
    #parser.add_argument("--stop_epochs", type=int, default = 30, help="Generator stops training after stop_epochs")
    #parser.add_argument("--k", type=int, default=30 , help="Number of discriminators")
    
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
def create_gen(latent_size):
    generator = Sequential()
    generator.add(layers.Dense(latent_size, input_dim=latent_size, activation="relu", kernel_initializer=keras.initializers.Identity(gain=1.0)))
    generator.add(layers.Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    input_shape = (latent_size,)
    latent = keras.Input(input_shape)
    fake_data = generator(latent)
    return keras.Model(latent, fake_data)

'''
    Create the discriminator. USes two dense layers and relu activation
'''
def create_dis(sub_size,data_size):
    discriminator = Sequential()
    discriminator.add(layers.Dense(np.ceil(np.sqrt(data_size)), input_dim=sub_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    discriminator.add(layers.Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    input_shape=(sub_size,)
    data = keras.Input(input_shape)
    fake = discriminator(data)
    return keras.Model(data, fake)

def load_data():
    if args.data == "C":
        (train_prior, prior_labels), (test_prior, test_labels) = tf.keras.datasets.cifar10.load_data() 
    else:
        (train_prior, prior_labels), (test_prior, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    idx = np.where(prior_labels == inlier)
    train = train_prior[idx[0]].copy() / 255
    test = test_prior.copy() / 255

    nx,ny,nz =(1,1,1)
    if args.data == "C":
        train_samples, nx, ny, nz = np.shape(train)
        train = train.reshape(train_samples, nx*ny*nz)
        test_samples, nx, ny, nz = np.shape(test)
        test = test.reshape(test_samples, nx*ny*nz)
    else: 
        train_samples, nx, ny = np.shape(train)
        train = train.reshape(train_samples, nx*ny)
        test_samples, nx, ny = np.shape(test)
        test = test.reshape(test_samples, nx*ny)  
        
    ground_truth = np.ones(len(test_labels))
    inlier_idx = np.where(test_labels == inlier)
    ground_truth[inlier_idx[0]] = 0
    return train, test, ground_truth, train_samples, nx*ny*nz

'''
    Plot the loss of the models. Generator in blue. AUC in Yellow
'''
def plot(train_history,names,k,result_path):
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    auc_y = train_history['auc']
    for i in range(k):
        names['dy_' + str(i)] = train_history['sub_discriminator{}_loss'.format(i)]
    xg = np.linspace(1, len(gy), len(gy))
    xd = np.linspace(1, len(dy), len(dy))
    xa = np.linspace(1, len(auc_y), len(auc_y))
    fig, ax = plt.subplots()
    ax.plot(xg, gy, color='blue', label="Generator loss")
    ax.plot(xd, dy,color='red', label="Avg discriminator loss")
    ax.plot(xa, auc_y, color='yellow', linewidth = '3', label="AUC")
    # dont show loss for sub discriminators. Gets very messy.
    #for i in range(k):
    #    ax.plot(x, names['dy_' + str(i)], color='green', linewidth='0.5')
    ax.legend(loc="upper left")
    plt.savefig(result_path + "/" + str(k))
    
'''
    Randomly draw subspaces for each sub_discriminator. Store them in names[]
'''
def draw_subspaces(dimension, ks,names):
    dims = random.choices(range(1,dimension), k=ks)
    for i in range(ks):
        names["subspaces"+str(i)] = random.sample(range(dimension), dims[i])
        

def start_training(seed,stop_epochs,k,path,lr_g,lr_d,result_path):
    set_seed(seed)
    train = True
    
    train_set,test_set,ground_truth,data_size,latent_size= load_data()
    
    if train:
        train_history = defaultdict(list)
        names = locals()
        epochs = stop_epochs * 3
        stop = 0

        generator = create_gen(latent_size)
        generator.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_g), loss='binary_crossentropy')
        latent_shape = (latent_size,)
        latent = keras.Input(latent_shape)
        
        draw_subspaces(latent_size,k,names)
        
        names["sub_discriminator_sum"] = 0
        # create sub_discriminators, sum is used to then take the average of all sub_discriminators' decisions
        for i in range(k):
            names["sub_discriminator" + str(i)] = create_dis(len(names["subspaces"+str(i)]),data_size)
            names["fake" + str(i)] = generator(latent) # generate the fake data of the generator
            #names["sub_discriminator" + str(i)].trainable = False

            names["fake" + str(i)] = names["sub_discriminator" + str(i)](tf.gather(names["fake"+str(i)],names["subspaces"+str(i)],axis=1))
            names["sub_discriminator_sum"] += names["fake" + str(i)]
            names["sub_discriminator" + str(i)].compile(optimizer=keras.optimizers.SGD(learning_rate=lr_d), loss='binary_crossentropy')
            
        names["sub_discriminator_sum"] /= k
        names["combine_model"] = keras.Model(latent, names["sub_discriminator_sum"]) # model with the average decision. Used to train the generator.
        names["combine_model"].compile(optimizer=keras.optimizers.SGD(learning_rate=lr_g), loss='binary_crossentropy')
        
        
        for epoch in range(epochs):
            print('Epoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(500, data_size)
            num_batches = int(data_size / batch_size)
        
            for idx in range(11,num_batches):
                print('\nTesting for epoch {} index {}:'.format(epoch + 1, idx + 1))

                # Generate noise
                noise_size = batch_size
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                
                data_batch = train_set[idx * batch_size: (idx + 1) * batch_size]
                
                names["generated_data"] = generator.predict(noise, verbose = 1)
                
                X = np.concatenate((data_batch, names["generated_data"]))
                Y = np.array([1] * batch_size + [0] * int(noise_size)) # 1 real data, 0 fake data
             
                discriminator_loss = 0
                for i in range(k):
                    names["sub_discriminator" + str(i) + "_loss"] = names["sub_discriminator" + str(i)].train_on_batch(X[:,names["subspaces"+str(i)]],Y)
                    train_history['sub_discriminator{}_loss'.format(i)].append(names['sub_discriminator' + str(i) + '_loss'])
                    discriminator_loss += names["sub_discriminator" + str(i) + "_loss"]
                discriminator_loss /= k
                train_history["discriminator_loss"].append(discriminator_loss)
                    
                p_value = names["sub_discriminator" + str(0)].predict(train_set[:,names["subspaces"+str(0)]])
                for i in range(1,k):
                    p_value += names["sub_discriminator" + str(i)].predict(train_set[:,names["subspaces"+str(i)]])
                        
                p_value /= k
                
                if stop == 0:
                    trick = np.array([1] * noise_size)
                    generator_loss = names["combine_model"].train_on_batch(noise, trick)
                    train_history['generator_loss'].append(generator_loss)
                else:
                    trick = np.array([1] * noise_size)
                    generator_loss = names["combine_model"].evaluate(noise, trick)
                    train_history['generator_loss'].append(generator_loss)

                if epoch + 1 > stop_epochs:
                        stop = 1
                        
            p_value = names["sub_discriminator" + str(0)].predict(test_set[:,names["subspaces"+str(0)]])
            for i in range(1,k):
                p_value += names["sub_discriminator" + str(i)].predict(test_set[:,names["subspaces"+str(i)]])
                
            data_y = pd.DataFrame(ground_truth)
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
            AUC = '{:.4f}'.format(sum / (len(inlier_parray) * len(outlier_parray)))
            for i in range(num_batches):
                train_history['auc'].append((sum / (len(inlier_parray) * len(outlier_parray))))
            print('AUC:{}'.format(AUC))

    plot(train_history,names,k,result_path)
    return AUC

def get_dim(path):
    _,_,_,_,latent_size = load_data()
    return latent_size
    
def start(path,result_path,csv_path):
    dimension = get_dim(path)
    sqrt = int(np.sqrt(dimension))
    seeds =[777, 45116, 4403, 92879, 34770]
    lrs_g = [0.01, 0.001]
    lrs_d = [0.01,0.001]
    ks =[sqrt,2*sqrt] #Decision between 2*sqrt and dim, 2^sqrt is way too much
    stop_epochs = [40]
    
    seed = 777
    
    with open(result_path + csv_path, "a", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer. writerow(["Seed", "LR_G", "LR_D", "k", "stop_epochs", "AUC"])
    
    for k in ks:
        for stop_epoch in stop_epochs:
                for lr_g in lrs_g:
                    for lr_d in lrs_d:
                        AUC = start_training(seed,stop_epoch,k,path,lr_g,lr_d,result_path)
                        output = [seed, lr_g, lr_d, k,stop_epoch,AUC]
                        with open(result_path + csv_path, "a", newline = "") as csv_file:
                            writer = csv.writer(csv_file)
                            writer. writerow(output)
    
def buildPath(dataset):
    result_path = "./Results/FeGAN_Results/Run_" + str(date.today()) + "_"+dataset
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path
    
if __name__ == '__main__':
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    args = parse_arguments()
    inlier = args.inlier
    
    
    gpu = "/device:GPU:" + str(args.gpu)
    
    with tf.device(gpu):
        if args.data == "C":
            start("C",buildPath("CIFAR" + str(inlier)),"/CIFAR"+str(inlier)+".csv")
        if args.data == "F":
            start("F",buildPath("FMNIST" + str(inlier)),"/FMNIST"+str(inlier)+".csv")
