import warnings #suppress warnings
import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.io import arff
#from pyod.models.anogan import AnoGAN
from pyod.models.mo_gaal import MO_GAAL

warnings.simplefilter("ignore")
#_______________________________________________________________________________________

# Probably doesn't work for images our high dim data yet
class CustomDataset(Dataset):
    def __init__(self, path):
        # start preprocessing 
        self.arff_data = arff.loadarff(path)
        self.df = pd.DataFrame(self.arff_data[0])
        #0 is outlier, 1 is normal data
        self.df["outlier"] = pd.factorize(self.df["outlier"])[0]
        #end preprocessing
        
        self.data_tensor = torch.tensor(self.df.to_numpy()).float()
        self.data_numpy = self.df.to_numpy()
        self.n = self.df.shape[0]
        
    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, i):
        return self.data_tensor[i]
#_______________________________________________________________________________________

#Variables, data settings etc.

path = "/home/denis/Documents/Resources_Thesis/Waveform_withoutdupl_norm_v01.arff"
seed = 777
torch.manual_seed(seed)
random.seed(seed)
num_workers = 2
batch_size = 128
#number of used GPUs
gpu = 0 

usedDevice = torch.device("cpu" if gpu == 0 else "cuda")
dataset = CustomDataset(path)
train_set , eval_set, test_set = torch.utils.data.random_split(dataset.data_numpy, [0.6,0.2,0.2]) #PFUSCH!! NUR ALS TEST FUER MOGAAL
#maybe data loader for each category?
dataloader = DataLoader(dataset=dataset.data_tensor, batch_size = batch_size, shuffle=True, num_workers=num_workers)

#_______________________________________________________________________________________

#Implementation of other models using the PyOD library for reference

mogaal = MO_GAAL()
mogaal.fit(train_set)

test_od_pred = mogaal.predict(test_set)


