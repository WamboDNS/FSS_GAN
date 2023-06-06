from scipy.io import arff
import pandas as pd
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
args = parser.parse_args()
input_path = "./Resources/Datasets"
result_path = "./Results/Run_" + str(date.today())



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
        
        
'''
    Calculate AUC and print it
'''
def AUC(ground_truth, decision):
    output = metrics.roc_auc_score(ground_truth, decision)
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
def run(dataset, seed, printer):
    AUC_scores = np.empty((0))
    AUC_scores = np.append(AUC_scores, seed)
    '''
    set_seed(seed)
    lof_model = LOF()
    lof_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, lof_model.decision_function(dataset.data)))
    
    set_seed(seed)
    fb50_model = FeatureBagging(n_estimators=50)
    fb50_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, fb50_model.decision_function(dataset.data)))
    
    set_seed(seed)
    fb100_model = FeatureBagging(n_estimators=100)
    fb100_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, fb100_model.decision_function(dataset.data)))
    
    set_seed(seed)
    fb500_model = FeatureBagging(n_estimators=500)
    fb500_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, fb500_model.decision_function(dataset.data)))
    
    set_seed(seed)
    knn_model = KNN()
    knn_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, knn_model.decision_function(dataset.data)))
    
    set_seed(seed)
    mogaal_model = MO_GAAL(lr_g = 0.01, stop_epochs=70)
    mogaal_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, mogaal_model.decision_function(dataset.data)))
    
    set_seed(seed)
    anogan_model = AnoGAN()
    anogan_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, anogan_model.decision_function(dataset.data)))
    '''
    set_seed(seed)
    svdd_model = DeepSVDD()
    svdd_model.fit(dataset.data)
    AUC_scores = np.append(AUC_scores, AUC(dataset.ground_truth, svdd_model.decision_function(dataset.data)))
    
    if printer == 1:
        with open(result_path + "/Params.txt", "a", newline = "") as txt_file:
            #txt_file.writelines("LOF: " + str(lof_model.get_params()) + "\n")
            #txt_file.writelines("FB50: " + str(fb50_model.get_params()) + "\n")
            #txt_file.writelines("FB100: " + str(fb100_model.get_params()) + "\n")
            #txt_file.writelines("FB500: " + str(fb500_model.get_params()) + "\n")
            #txt_file.writelines("KNN: " + str(knn_model.get_params()) + "\n")
            #txt_file.writelines("MO_GAAL: " + str(mogaal_model.get_params()) + "\n")
            #txt_file.writelines("AnoGAN: " + str(anogan_model.get_params()) + "\n")
            txt_file.writelines("Deep SVDD: " + str(svdd_model.get_params()) + "\n")
        
    return AUC_scores


'''
    The backbone of the experiment. Choose seeds, load and prepare data, start the pipeline and
    write AUC values to a csv.
'''
def experiment(data_path, result_csv):
    printer=1
    dataset = CustomData(input_path + data_path)
    seeds =[777, 45116, 4403, 92879, 34770]
    
    with open(result_path + result_csv, "a", newline = "") as csv_file:
        writer = csv.writer(csv_file)
        #writer. writerow(["Seed","LOF_AUC", "LOF_50", "LOF_100", "LOF_500", "KNN_AUC", "MO_GAAL_AUC", "AnoGAN_AUC"])
        writer.writerow(["Deep_SVDD_AUC"])
        
    for i in range(len(seeds)):
        print("---------- " + "start run " + data_path + " " + str(i) + " ----------")
        output = run(dataset, seeds[i], printer)
        printer = -1
        with open(result_path + data_path, "a", newline = "") as csv_file:
            writer = csv.writer(csv_file)
            writer. writerow(output)
        print("---------- " + "end run " + data_path + " " + str(i) + " ----------")



def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    gpu = "/device:GPU:" + args.gpu
    
    arrythmia_path = "/Arrhythmia_withoutdupl_norm_02_v01.arff"
    wave_path = "/Waveform_withoutdupl_norm_v01.arff"
    internet_ads_path = "/InternetAds_withoutdupl_norm_02_v01.arff"
    spambase_path = "/SpamBase_withoutdupl_norm_02_v01.arff"
    result_arrythmia = "/Arrythmia.csv"
    result_waveform = "/Waveform.csv"
    result_internet_ads = "/Internet_ads.csv"
    result_spambase = "/Spambase.csv"
    
    with tf.device(gpu):
        experiment(arrythmia_path, result_arrythmia)
        experiment(wave_path,result_waveform)
        experiment(internet_ads_path,result_internet_ads)
        experiment(spambase_path,result_spambase)
    
    
if __name__ == "__main__":
    main()