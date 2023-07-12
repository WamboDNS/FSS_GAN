import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
t = [1,2,3,4,5,6,7,8,9]
x = pd.DataFrame()
x["epochs"] = t
x["values"] = t
fst= [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
upper = [0.11,0.18,0.33,0.41,0.53,0.64,0.75,0.82,0.94]
lower = [0.05,0.17,0.28,0.4,0.485,0.6,0.72,0.84,0.91]

def plot_avg(disc, gen,result_path):
    plt.cla()
    color = sns.color_palette("husl", 8)
    sns.set(style="darkgrid",palette=color)
    sns.lineplot(disc,x="epochs", y="values",linewidth = 2,label="Discriminator")
    sns.lineplot(gen,x="epochs", y="values",linewidth = 2,label="Generator")
    #sns.lineplot(auc,x="epochs", y="values",linewidth = 2,label="AUC")
    plt.savefig("./" +"average"+".svg",format="svg",dpi=1200)
    
def plot(dy,gy,auc_y,k,seed,result_path):
    plt.cla()
    plt.style.use('ggplot')

    x = np.linspace(1, len(gy), len(gy))
    fig, ax = plt.subplots()
    ax.plot(x, gy, color="cornflowerblue", label="Generator loss", linewidth=2)
    ax.plot(x, dy,color="crimson", label="Average discriminator loss", linewidth=2)
    ax.plot(x, auc_y, color="yellow", linewidth = 3, label="ROC AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax.legend(loc="lower right")
    plt.savefig("./" +"avg"+".svg",format="svg",dpi=1200)
    
bound = pd.DataFrame()
bound["epochs"] = t
bound["values"] = fst
temp = pd.DataFrame()
temp["epochs"] = t
temp["values"] = upper
bound = pd.concat([bound,temp])
temp["epochs"] = t
temp["values"] = lower
bound = pd.concat([bound,temp])

anti = bound.copy()
anti["values"] = 1 - anti["values"]

plot(fst,upper,lower,1,1,"")

plot_avg(bound, anti,"")
