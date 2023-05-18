from scipy.io import arff
import pandas as pd
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.anogan import AnoGAN
from sklearn import metrics
import tensorflow as tf
import numpy as np
import random
import csv
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.debugging.experimental.disable_dump_debug_info
warnings.filterwarnings("ignore")

class CustomData():
    def __init__(self, path):
        arff_data = arff.loadarff(path)
        df = pd.DataFrame(arff_data[0])
        df["outlier"] = pd.factorize(df["outlier"], sort=True)[0]
        
        self.data = df.iloc[:,:-2]
        self.ground_truth = df.iloc[:,-1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
        
def AUC(truth, decision):
    output = metrics.roc_auc_score(truth, decision)
    print("AUC: " + str(output))
    return output
    
# set seeds in every possible package (and pray that it works)
def initialize(seed):
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ["PYTHONHASSEED"] = str(seed)
    
# run the models and calculate AUC
def run(dataset, seed):
    AUC_scores = np.empty((0))
    AUC_scores = np.append(AUC_scores, seed)
    
    lof_model = LOF()
    lof_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, lof_model.decision_function(dataset.data)))
    
    knn_model = KNN()
    knn_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, knn_model.decision_function(dataset.data)))
    
    mogaal_model = MO_GAAL(lr_d=0.01, lr_g=0.01, stop_epochs=50) #arrythmia is 50
    mogaal_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, mogaal_model.decision_function(dataset.data)))
    
    anogan_model = AnoGAN()
    anogan_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, anogan_model.decision_function(dataset.data)))
    
    return AUC_scores

# the main experiment. Load the given dataset, run it, write AUC in a csv.
def experiment(data_path, result_path):
    dataset = CustomData(data_path)
    seed = 0
    
    with open(result_path, "a", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        writer. writerow(["Seed","LOF_AUC", "KNN_AUC", "MO_GAAL_AUC", "AnoGAN_AUC"])
        
    for i in range(10):
        print("---------- " + "start run " + data_path + " " + str(i) + " ----------")
        seed += 111
        initialize(seed)
        output = run(dataset, seed)
        with open(result_path, "a", newline = "") as csv_file:
            writer = csv.writer(csv_file)
            writer. writerow(output)
        print("---------- " + "end run " + data_path + " " + str(i) + " ----------")

def main():
    
    arrythmia_path = "./Resources/Datasets/Arrhythmia_withoutdupl_norm_02_v01.arff"
    wave_path = "./Resources/Datasets/Waveform_withoutdupl_norm_v01.arff"
    internet_ads_path = "./Resources/Datasets/InternetAds_withoutdupl_norm_02_v01.arff"
    result_arrythmia = "./Results/Run_2/Arrythmia.csv"
    result_waveform = "./Results/Run_2/Waveform.csv"
    result_internet_ads = "./Results/Run_2/Internet_ads.csv"

    experiment(arrythmia_path,result_arrythmia)
    experiment(wave_path,result_waveform)
    experiment(internet_ads_path,result_internet_ads)
        
    
    
if __name__ == "__main__":
    main()
    
    