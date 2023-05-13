import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.io import arff

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
        
        self.data = torch.tensor(self.df.to_numpy()).float()
        self.n = self.df.shape[0]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
#_______________________________________________________________________________________

path = "/home/denis/Documents/Resources_Thesis/Waveform_withoutdupl_norm_v01.arff"
seed = 777
torch.manual_seed(seed)
random.seed(seed)
num_workers = 2
batch_size = 128
gpu = 0 # number of used GPUs

usedDevice = torch.device("cpu" if gpu == 0 else "cuda")
dataset = CustomDataset(path)
train_set , eval_set, test_set = torch.utils.data.random_split(dataset.data, [0.6,0.2,0.2])
#maybe data loader for each category?
dataloader = DataLoader(dataset=dataset.data, batch_size = batch_size, shuffle=True, num_workers=num_workers)