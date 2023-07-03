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
    parser.add_argument("--data", default="W")
    
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

def load_data(path):
    arff_data = arff.loadarff(path)
    df = pd.DataFrame(arff_data[0])
    df["outlier"] = pd.factorize(df["outlier"], sort=True)[0] #maybe flip
    data_x = df.iloc[:,:-2]
    data_y = df.iloc[:,-1]
    
    return data_x, data_y

'''
    Plot the loss of the models. Generator in blue. AUC in Yellow
'''
def plot(train_history,names,k,seed,result_path):
    plt.style.use('ggplot')
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    auc_y = train_history['auc']
    for i in range(k):
        names['dy_' + str(i)] = train_history['sub_discriminator{}_loss'.format(i)]
    x = np.linspace(1, len(gy), len(gy))
    fig, ax = plt.subplots()
    ax.plot(x, gy, color="cornflowerblue", label="Generator loss", linewidth=2)
    ax.plot(x, dy,color="crimson", label="Average discriminator loss", linewidth=2)
    ax.plot(x, auc_y, color="yellow", linewidth = 3, label="ROC AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax.legend(loc="lower right")
    plt.savefig(result_path + "/" + str(seed)+".svg",format="svg",dpi=1200)
    
    fig_all,ax_all = plt.subplots()
    # dont show loss for sub discriminators. Gets very messy.
    for i in range(k):
        ax_all.plot(x, names['dy_' + str(i)], color="fuchsia", linewidth='0.5',alpha=0.3)
        
    ax_all.plot(x, gy, color="cornflowerblue", label="Generator loss", linewidth=2)
    ax_all.plot(x, dy,color="crimson", label="Average discriminator loss", linewidth=2)
    ax_all.plot(x, auc_y, color="yellow", linewidth = 3, label="ROC AUC")
    plt.savefig(result_path+"/"+str(seed)+"_all.svg",format="svg",dpi=1200)
    
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
    
    data_x, data_y = load_data(path)
    data_size = data_x.shape[0] # n := number of samples
    latent_size = data_x.shape[1] # dimension of the data set
    
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
                    
                p_value = names["sub_discriminator" + str(0)].predict(data_x.to_numpy()[:,names["subspaces"+str(0)]])
                for i in range(1,k):
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

                if epoch + 1 > stop_epochs:
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
            AUC = '{:.4f}'.format(sum / (len(inlier_parray) * len(outlier_parray)))
            for i in range(num_batches):
                train_history['auc'].append((sum / (len(inlier_parray) * len(outlier_parray))))
            print('AUC:{}'.format(AUC))

    plot(train_history,names,k,seed,result_path)
    return AUC,train_history['auc'],train_history['discriminator_loss'],train_history['generator_loss']

def get_dim(path):
    return load_data(path)[0].shape[1]

def plot_avg(auc_avg, disc_avg,gen_avg,result_path):
    plt.style.use('ggplot')
    dy = disc_avg
    gy = gen_avg
    auc_y = auc_avg

    x = np.linspace(1, len(gy), len(gy))
    fig, ax = plt.subplots()
    ax.plot(x, gy, color="cornflowerblue", label="Generator loss",linewidth = 2)
    ax.plot(x, dy,color="crimson", label="Average discriminator loss",linewidth = 2)
    ax.plot(x, auc_y, color="yellow", linewidth = 3, label="ROC AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax.legend(loc="lower right")
    plt.savefig(result_path + "/" +"average"+".svg",format="svg",dpi=1200)
    
    
def start(path,result_path,csv_path):
    dimension = get_dim(path)
    sqrt = int(np.sqrt(dimension))
    seeds =[777, 45116, 4403, 92879, 34770]
    lr_g = 0.001
    lr_d= 0.01
    k = 2*sqrt
    stop_epoch = 30
    
    starter=0
    
    with open(result_path + csv_path, "a", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer. writerow(["Seed", "LR_G", "LR_D", "k", "stop_epochs", "AUC"])
        
    for seed in seeds:
        AUC,temp_auc, temp_gen, temp_disc = start_training(seed,stop_epoch,k,path,lr_g,lr_d,result_path)
        output = [seed, lr_g, lr_d, k,stop_epoch,AUC]
        if starter == 0:
            auc_avg = np.array(temp_auc)
            disc_avg = np.array(temp_disc)
            gen_avg = np.array(temp_gen)
        else:
            auc_avg += temp_auc
            disc_avg += temp_disc
            gen_avg += temp_gen
        starter = -1
        with open(result_path + csv_path, "a", newline = "") as csv_file:
            writer = csv.writer(csv_file)
            writer. writerow(output)
    auc_avg /= len(seeds)
    disc_avg /= len(seeds)
    gen_avg /= len(seeds)
    plot_avg(auc_avg, disc_avg,gen_avg,result_path)

    
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
    
    
    gpu = "/device:GPU:" + str(args.gpu)
    
    with tf.device(gpu):
        if args.data == "I":
            start("../Resources/Datasets/InternetAds_withoutdupl_norm_02_v01.arff",buildPath("InternetAds"),"/InternetAds.csv")
        if args.data == "S":
            start("../Resources/Datasets/SpamBase_withoutdupl_norm_02_v01.arff",buildPath("SpamBase"),"/SpamBase.csv")
        if args.data == "A":
            start("../Resources/Datasets/Arrhythmia_withoutdupl_norm_02_v01.arff",buildPath("Arrythmia"),"/Arrythmia.csv")
        if args.data == "W":
            start("../Resources/Datasets/Waveform_withoutdupl_norm_v01.arff",buildPath("Waveform"),"/Waveform.csv")
