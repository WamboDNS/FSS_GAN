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
def pipeline(dataset, seed, inlier_class, ground_truth, testset, result_path, printer):
    AUC_scores = np.empty((0))
    AUC_scores = np.append(AUC_scores, seed)
    AUC_scores = np.append(AUC_scores, inlier_class)
    
    set_seed(seed)
    lof_model = LOF()
    lof_model.fit(dataset)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, lof_model.decision_function(testset)))
    
    set_seed(seed)
    fb50_model = FeatureBagging(n_estimators=50)
    fb50_model.fit(dataset)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, fb50_model.decision_function(testset)))
    
    set_seed(seed)
    fb100_model = FeatureBagging(n_estimators=100)
    fb100_model.fit(dataset)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, fb100_model.decision_function(testset)))
    
    set_seed(seed)
    fb500_model = FeatureBagging(n_estimators=500)
    fb500_model.fit(dataset)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, fb500_model.decision_function(testset)))
    
    set_seed(seed)
    knn_model = KNN()
    knn_model.fit(dataset)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, knn_model.decision_function(testset)))
    
    set_seed(seed)
    mogaal_model = MO_GAAL(lr_g = 0.01, stop_epochs=50)
    mogaal_model.fit(dataset)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, mogaal_model.decision_function(testset)))
    
    set_seed(seed)
    anogan_model = AnoGAN()
    anogan_model.fit(dataset)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, anogan_model.decision_function(testset)))
    
    set_seed(seed)
    svdd_model = DeepSVDD()
    svdd_model.fit(dataset)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, svdd_model.decision_function(testset)))
    
    # store params of each model in a file
    if printer == 1:
        with open(result_path + "/Params_" + str(inlier_class) + "_" + args.data  +".txt", "a", newline = "") as txt_file:
            txt_file.writelines("LOF: " + str(lof_model.get_params()) + "\n")
            txt_file.writelines("FB50: " + str(fb50_model.get_params()) + "\n")
            txt_file.writelines("FB100: " + str(fb100_model.get_params()) + "\n")
            txt_file.writelines("FB500: " + str(fb500_model.get_params()) + "\n")
            txt_file.writelines("KNN: " + str(knn_model.get_params()) + "\n")
            txt_file.writelines("MO_GAAL: " + str(mogaal_model.get_params()) + "\n")
            txt_file.writelines("AnoGAN: " + str(anogan_model.get_params()) + "\n")
            txt_file.writelines("Deep SVDD: " + str(svdd_model.get_params()) + "\n")
        
    return AUC_scores


'''
    The backbone of the experiment. Choose seeds, load and prepare data, start the pipeline and
    write AUC values to a csv.
'''
def experiment(data_path, inlier, result_path):
    printer = 1
    seeds =[777, 45116, 4403, 92879, 34770]

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
    with open(result_path + data_path, "a", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer. writerow(["Seed", "Class", "LOF_AUC", "LOF_50", "LOF_100", "LOF_500", "KNN_AUC", "MO_GAAL_AUC", "AnoGAN_AUC", "DeepSVDD_AUC"])
        
    for i in range(len(seeds)):
        print("---------- " + "start run " + data_path + " " + str(i) + " ----------")
        output = pipeline(train, seeds[i], inlier, ground_truth, test, result_path, printer)
        printer = -1
        with open(result_path + data_path, "a", newline = "") as csv_file:
            writer = csv.writer(csv_file)
            writer. writerow(output)
        print("---------- " + "end run " + data_path + " " + str(i) + " ----------")

'''
    Build path used to store results and params automatically
'''
def buildPath(inlier):
    result_path = "./Results/Run_" + str(date.today()) + "/class_"+str(inlier)
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
                result_path = buildPath(inlier)
                experiment(fashion_mnist_path, inlier, result_path)
        if args.data == "C":
            for inlier in range(start_class,end_class-1,-1):
                result_path = buildPath(inlier)
                experiment(cifar_path, inlier, result_path)
        print("End")
    
    
if __name__ == "__main__":
    main()
