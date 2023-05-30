import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.lof import LOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.knn import KNN
from pyod.models.anogan import AnoGAN
from pyod.models.deep_svdd import DeepSVDD
from sklearn import metrics
import tensorflow as tf
import numpy as np
import csv
import warnings
import sys
from datetime import date
import argparse
tf.debugging.experimental.disable_dump_debug_info
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="One class classification")
parser.add_argument("--gpu", help="GPU to be used, indexed from 0",default="0")
parser.add_argument("--data", help="Data set to be used. C for Cifar, F for Fashion MNIST")
parser.add_argument("--start", help = "Start class, count down from here to end class incl", type=int, default=9)
parser.add_argument("--end", help = "Last class of the run", type = int, default = 0)
args = parser.parse_args()


'''
    Calculate AUC and print it
'''
def AUC(truth, decision):
    output = metrics.roc_auc_score(truth, decision)
    print("AUC: " + str(output))
    return output
    
    
'''
    Set seeds in every possible package (and pray that it works)
'''
def set_seed(seed):
    tf.keras.utils.set_random_seed(seed) #seeds numpy, random and tf all at once
    os.environ["PYTHONHASSEED"] = str(seed)
    
    
'''
    Pipeline to run the models one by one. Store AUC values in an array. Reseed before every new model.
'''
def pipeline(dataset, seeds, inlier_class, ground_truth, testset, result_path):
    learning_rates_g = [0.01,0.1]#[0.0001, 0.001, 0.01]
    stop_epochs = [1]#,40,60,80,100]
    
    avg_AUC = np.empty(0)
    params = []
    
    for lr in learning_rates_g:
        for n in stop_epochs:
            AUC_scores = np.empty((0))
            for seed in seeds:
                set_seed(seed)
                mogaal_model = MO_GAAL(lr_g = lr, stop_epochs=n)
                mogaal_model.fit(dataset)
                AUC_scores = np.append(AUC_scores, AUC(ground_truth, mogaal_model.decision_function(testset)))
            avg_AUC = np.append(avg_AUC, np.average(AUC_scores))
            params.append((lr,n))
    
    # store params of each model in a file
        with open(result_path + "/Params_" + str(inlier_class) + "_" + args.data  +".txt", "a", newline = "") as txt_file:
            for i in range(len(avg_AUC)):
                txt_file.writelines("lr_g=" + str(params[i][0]) + "; stop_epochs=" + str(params[i][1])+ "; average AUC=" + str(avg_AUC[i]) + "\n")


'''
    The backbone of the experiment. Choose seeds, load and prepare data, start the pipeline and
    write AUC values to a csv.
'''
def experiment(inlier, result_path):
    seeds =[777, 45116]#, 4403, 92879, 34770]

    # Load data set, convert data to fit the experiment (inlier class -> 0, other classes -> 1)
    #--------------------------------------------------------------
    if args.data == "C":
        (train_prior, prior_labels), (test_prior, test_labels) = tf.keras.datasets.cifar10.load_data() 
    else:
        (train_prior, prior_labels), (test_prior, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    idx = np.where(prior_labels == inlier)
    train = train_prior[idx[0]].copy() / 255
    test = test_prior.copy() / 255

    if args.data == "C":
        nsamples, nx, ny, nz = np.shape(train)
        train = train.reshape(nsamples, nx*ny*nz)
        nsamples, nx, ny, nz = np.shape(test)
        test = test.reshape(nsamples, nx*ny*nz)
    else: 
        nsamples, nx, ny = np.shape(train)
        train = train.reshape(nsamples, nx*ny)
        nsamples, nx, ny = np.shape(test)
        test = test.reshape(nsamples, nx*ny)  
        
    ground_truth = np.ones(len(test_labels))
    inlier_idx = np.where(test_labels == inlier)
    ground_truth[inlier_idx[0]] = 0
    
    
    
    # start pipeline and write to csv
    #----------------------------------------------------------------     
    pipeline(train, seeds, inlier, ground_truth, test, result_path)

'''
    Build path used to store results and params automatically
'''
def buildPath(inlier,data_path):
    result_path = "./Results/Tune_MO_GAAL_" + str(date.today()) + "/class_"+str(inlier)+data_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path

def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    fashion_mnist_path = "/Fashion_MNIST.csv"
    cifar_path = "/Cifar10.csv"
    
    gpu = "/device:GPU:" + args.gpu
    start_class = args.start
    #inclusive
    end_class = args.end
    
    with tf.device(gpu):
        if args.data == "F":
            for inlier in range(start_class,end_class-1,-1):
                result_path = buildPath(inlier, "_F")
                experiment(inlier, result_path)
        if args.data == "C":
            for inlier in range(start_class,end_class-1,-1):
                result_path = buildPath(inlier, "_C")
                experiment(inlier, result_path)
        print("End")
    
    
if __name__ == "__main__":
    main()
