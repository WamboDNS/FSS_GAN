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
tf.debugging.experimental.disable_dump_debug_info
warnings.filterwarnings("ignore")

result_path = "./Results/Run_" + str(date.today())
if not os.path.exists(result_path):
    os.mkdir(result_path)


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
    tf.config.experimental.enable_op_determinism()
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ["PYTHONHASSEED"] = str(seed)
    
    
'''
    Pipeline to run the models one by one. Store AUC values in an array. Reseed before every new model.
'''
def pipeline(dataset, seed, ground_truth, testset):
    AUC_scores = np.empty((0))
    AUC_scores = np.append(AUC_scores, seed)
    
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
    
    return AUC_scores


'''
    The backbone of the experiment. Choose seeds, load and prepare data, start the pipeline and
    write AUC values to a csv.
'''
def experiment(data_path):
    seeds =[777, 45116, 4403, 92879, 34770]
    #--------------------------------------------------------
    # prepare data
    ''' The following block is for Fashion MNIST'''
    '''
    (prior, prior_labels), (test, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    inlier = 6
    
    idx = np.where(prior_labels == inlier)
    train = prior[idx].copy() / 255
    nsamples, nx, ny = np.shape(train)
    train = train.reshape(nsamples, nx*ny)
    
    test_copy = test.copy() / 255
    nsamples, nx, ny = np.shape(test_copy)
    test_copy = test_copy.reshape(nsamples, nx*ny)
    
    # DONT USE 1 OR 0 AS INLIER
    ground_truth = test_labels.copy()
    ground_truth[ground_truth != inlier] = 1
    ground_truth[ground_truth == inlier] = 0
    '''
    #--------------------------------------------------------
    
    (prior, prior_labels), (test, test_labels) = tf.keras.datasets.cifar10.load_data() #tf.keras.datasets.fashion_mnist.load_data()
    inlier = 6
    idx = np.where(prior_labels == inlier)

    train = prior[idx[0]].copy()

    print(np.shape(train))
    print(len(train))
    nsamples, nx, ny, nz = np.shape(train)
    train = train.reshape(nsamples, nx*ny*nz) / 255
        

    test_copy = test.copy() / 255
    nsamples, nx, ny, nz = np.shape(test_copy)
    test_copy = test_copy.reshape(nsamples, nx*ny*nz)
        
        # DONT USE 1 OR 0 AS INLIER
    ground_truth = test_labels.copy()
    ground_truth[ground_truth != inlier] = 1
    ground_truth[ground_truth == inlier] = 0
    
    
    # start pipeline and write to csv
    with open(result_path + data_path, "a", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer. writerow(["Seed","LOF_AUC", "LOF_50", "LOF_100", "LOF_500", "KNN_AUC", "MO_GAAL_AUC", "AnoGAN_AUC", "DeepSVDD_AUC"])
        
    for i in range(len(seeds)):
        print("---------- " + "start run " + data_path + " " + str(i) + " ----------")
        output = pipeline(train, seeds[i], ground_truth, test_copy)
        with open(result_path + data_path, "a", newline = "") as csv_file:
            writer = csv.writer(csv_file)
            writer. writerow(output)
        print("---------- " + "end run " + data_path + " " + str(i) + " ----------")


def main():
    
    fashion_mnist_path = "/Fashion_MNIST.csv"
    cifar_path = "/Cifar10.csv"
    experiment(cifar_path)
    
    
if __name__ == "__main__":
    main()