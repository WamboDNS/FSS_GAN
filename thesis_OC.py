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

tf.debugging.experimental.disable_dump_debug_info
warnings.filterwarnings("ignore")

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
def initialize(seed):
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
    
    initialize(seed)
    lof_model = LOF()
    lof_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, lof_model.decision_function(testset)))
    
    initialize(seed)
    fb50_model = FeatureBagging(n_estimators=50)
    fb50_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, fb50_model.decision_function(testset)))
    
    initialize(seed)
    fb100_model = FeatureBagging(n_estimators=100)
    fb100_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, fb100_model.decision_function(testset)))
    
    initialize(seed)
    fb500_model = FeatureBagging(n_estimators=500)
    fb500_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, fb500_model.decision_function(testset)))
    
    initialize(seed)
    knn_model = KNN()
    knn_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, knn_model.decision_function(testset)))
    
    initialize(seed)
    mogaal_model = MO_GAAL(lr_g = 0.01, stop_epochs=70)
    mogaal_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, mogaal_model.decision_function(testset)))
    
    initialize(seed)
    anogan_model = AnoGAN()
    anogan_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, anogan_model.decision_function(dataset.data)))
    
    initialize(seed)
    svdd_model = DeepSVDD()
    svdd_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(ground_truth, svdd_model.decision_function(dataset.data)))
    
    return AUC_scores

'''
    The backbone of the experiment. Choose seeds, load and prepare data, start the pipeline and
    write AUC values to a csv.
'''
def experiment(data_path, result_path):
    seeds =[777, 45116, 4403, 92879, 34770]
    
    #--------------------------------------------------------
    # prepare data
    (prior, prior_labels), (test, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    outlier = 3
    
    idx = prior_labels == outlier
    train = prior[idx].copy() / 255
    nsamples, nx, ny = np.shape(train)
    train = train.reshape(nsamples, nx*ny)
    
    test_copy = test.copy() / 255
    nsamples, nx, ny = np.shape(test_copy)
    test_copy = test_copy.reshape(nsamples, nx*ny)
    
    # DONT USE 1 OR 0 AS INLIER
    ground_truth = test_labels.copy()
    ground_truth[ground_truth != outlier] = 1
    ground_truth[ground_truth == outlier] = 0
    
    #--------------------------------------------------------
    # start pipeline and write to csv
    with open(result_path, "a", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer. writerow(["Seed","LOF_AUC", "LOF_50", "LOF_100", "LOF_500", "KNN_AUC", "MO_GAAL_AUC", "AnoGAN_AUC", "DeepSVDD_AUC"])
        
    for i in range(len(seeds)):
        print("---------- " + "start run " + data_path + " " + str(i) + " ----------")
        output = pipeline(train, seeds[i], ground_truth, test_copy)
        with open(result_path, "a", newline = "") as csv_file:
            writer = csv.writer(csv_file)
            writer. writerow(output)
        print("---------- " + "end run " + data_path + " " + str(i) + " ----------")

def main():
    
    iteration = "5"
    result_fashion_mnist = "./Results/Run_" + iteration+ "/Fashion_MNIST.csv"
    
    if str(sys.argv[1]) != "0":
        experiment("Fashion_MNIST", result_fashion_mnist)
    
    
if __name__ == "__main__":
    main()
