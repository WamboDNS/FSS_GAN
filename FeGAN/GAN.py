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
    latent = keras.Input(shape=(latent_size,))
    fake_data = generator(latent)
    return keras.Model(latent, fake_data)

'''
    Create the discriminator. USes two dense layers and relu activation
'''
def create_dis(sub_size,data_size):
    discriminator = Sequential()
    discriminator.add(layers.Dense(np.ceil(np.sqrt(data_size)), input_dim=sub_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    discriminator.add(layers.Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    data = keras.Input(shape=(sub_size,))
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
def plot(train_history,names,k,result_path):
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    auc_y = train_history['auc']
    for i in range(k):
        names['dy_' + str(i)] = train_history['sub_discriminator{}_loss'.format(i)]
    x = np.linspace(1, len(gy), len(gy))
    fig, ax = plt.subplots()
    ax.plot(x, gy, color='blue', label="Generator loss")
    ax.plot(x, dy,color='red', label="Avg discriminator loss")
    ax.plot(x, auc_y, color='yellow', linewidth = '3', label="AUC")
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
        latent = keras.Input(shape=(latent_size,))
        
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

    plot(train_history,names,k,result_path)
    return AUC

def get_dim(path):
    return load_data(path)[0].shape[1]
    
def start(path,result_path,data_path):
    dimension = get_dim(path)
    sqrt = int(np.sqrt(dimension))
    seeds =[777, 45116, 4403, 92879, 34770]
    lrs_g = [0.001]
    lrs_d = [0.01,0.001]
    ks =[2*sqrt,dimension,2**sqrt] #Decision between 2*sqrt and dim
    stop_epochs = [40]
    
    seed = 777
    
    with open(result_path + data_path, "a", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer. writerow(["Seed", "LR_G", "LR_D", "k", "stop_epochs", "AUC"])
    
    for k in ks:
        for stop_epoch in stop_epochs:
                for lr_g in lrs_g:
                    for lr_d in lrs_d:
                        AUC = start_training(seed,stop_epoch,k,path,lr_g,lr_d,result_path)
                        output = [seed, lr_g, lr_d, k,stop_epoch,AUC]
                        with open(result_path + data_path, "a", newline = "") as csv_file:
                            writer = csv.writer(csv_file)
                            writer. writerow(output)
    
def buildPath(dataset):
    result_path = "./Results/FeGAN_Results/Run_" + str(date.today()) + "_"+dataset
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path

def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    args = parse_arguments()
    
    use = -1
    
    if args.gpu == 0:
        use = 0
    elif args.gpu == 1:
        use = 1
    elif args.gpu == 2:
        use = 0
    elif args.gpu == 3:
        use = 1
    
    gpu = "/device:GPU:" + str(use)
    
    with tf.device(gpu):
        if args.gpu == 0:
            start("../Resources/Datasets/InternetAds_withoutdupl_norm_02_v01.arff",buildPath("InternetAds"),"/InternetAds.csv")
        if args.gpu == 1:
            start("../Resources/Datasets/SpamBase_withoutdupl_norm_02_v01.arff",buildPath("SpamBase"),"/SpamBase.csv")
        if args.gpu == 2:
            start("../Resources/Datasets/Arrhythmia_withoutdupl_norm_02_v01.arff",buildPath("Arrythmia"),"/Arrythmia.csv")
        if args.gpu == 3:
            start("../Resources/Datasets/Waveform_withoutdupl_norm_v01.arff",buildPath("Waveform"),"/Waveform.csv")

if __name__ == '__main__':
    main()