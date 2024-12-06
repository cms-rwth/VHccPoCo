import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import argparse
import glob
import os, gc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, log_loss

from imblearn.over_sampling import SMOTE
import numpy as np
import awkward as ak
from coffea.util import load
from concurrent.futures import ThreadPoolExecutor
cpucount = len(os.sched_getaffinity(0))
print(f"Found {cpucount} CPUs.")

def get_inputs(channel,model_type='gnn'):
    if model_type in ["lgbm","dnn"]:
        if channel == "ZLL":
            inps = ["dilep_m","dilep_pt","dilep_dr","dilep_deltaPhi","dilep_deltaEta",
                    "dijet_m","dijet_pt","dijet_dr","dijet_deltaPhi","dijet_deltaEta",
                    "dijet_CvsL_max","dijet_CvsL_min","dijet_CvsB_max","dijet_CvsB_min",
                    "dijet_pt_max","dijet_pt_min",
                    "ZH_pt_ratio","ZH_deltaPhi","deltaPhi_l2_j1","deltaPhi_l2_j2",
                    "PuppiMET_pt","PuppiMET_phi","nPV","LeptonCategory"]
        if channel == "WLnu":
            inps = ["dijet_m","dijet_pt","dijet_dr","dijet_deltaPhi","dijet_deltaEta",
                    "dijet_CvsL_max","dijet_CvsL_min","dijet_CvsB_max","dijet_CvsB_min",
                    "dijet_pt_max","dijet_pt_min",
                    "W_mt","W_pt","pt_miss","WH_deltaPhi",
                    "deltaPhi_l1_j1","deltaPhi_l1_MET","deltaPhi_l1_b","deltaEta_l1_b","deltaR_l1_b",
                    "b_CvsL","b_CvsB","b_Btag","top_mass"]
        if channel == "ZNuNu":
            inps = ["dijet_m","dijet_pt","dijet_dr","dijet_deltaPhi","dijet_deltaEta",
                    "dijet_CvsL_max","dijet_CvsL_min","dijet_CvsB_max","dijet_CvsB_min",
                    "dijet_pt_max","dijet_pt_min",
                    "ZH_pt_ratio","ZH_deltaPhi","Z_pt"]
    
    elif model_type == "gnn":
        if channel == "ZLL":
            inps = ["JetGood_btagCvL","JetGood_btagCvB",
                    "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                    "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                    "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                    "ll_pt","ll_eta","ll_phi","ll_mass",
                    "PuppiMET_pt","PuppiMET_phi","nPV","LeptonCategory"]
        if channel == "WLNu":
            inps = ["JetGood_btagCvL","JetGood_btagCvB",
                    "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                    "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                    "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                    "W_pt","W_eta","W_phi","W_mt",
                    "PuppiMET_pt","PuppiMET_phi","nPV","W_m","LeptonCategory"]
        if channel == "ZNuNu":
            inps = ["JetGood_btagCvL","JetGood_btagCvB",
                    "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                    "Z_pt","Z_eta","Z_phi","Z_m",
                    "PuppiMET_pt","PuppiMET_phi","nPV"]


    inps = ["EventNr"]+inps
    return ["events_"+i for i in inps]
    print(f"Channel name {channel} for model tpye {model_type} not found!")
    exit(1)

def get_SR_name(channel):
    if channel=="ZLL":
        return "SR_ll_2J_cJ"
    elif channel=="WLNu":
        return "SR_Wln_2J_cJ"
    elif channel=="ZNuNu":
        return "SR_Znn_2J_cJ"
    raise NotImplementedError("Need to hardcode the names of the channels.")

def load_files(dr, SR, cols, test=False):
    all_files = glob.glob(f"{dr}/**/{SR}/*.parquet", recursive=True)
    if any([f for f in all_files if "_bx" in f]):
        print("\tWill explicity remove bc/cx/ll directories.")
        all_files = [f for f in all_files if "DiJet_bx" not in f and "DiJet_cx" not in f and "DiJet_ll" not in f]
    ln = len(all_files)
    if test:        
        all_files = all_files[:min(5,ln)]
        ln = len(all_files)

    def read_file(file):
        readarr =  ak.from_parquet(file, columns=cols)
        return readarr

    batchsize = 4000
    if ln > batchsize:
        print(f"\t{ln} files! That's a lot of files!")
        arrs = []
        
        for i in range(ln//batchsize+1):
            frm = i*batchsize
            to = min(ln,(i+1)*batchsize)
            if to>=frm:
                arrs.append(read_file(all_files[frm:to]))
        return ak.concatenate(arrs)
    else:
        return read_file(all_files)

def apply_norm(dr,SR,coffeafile,df):
    sampname = dr.split('/')[-1]
    sumwtdict = coffeafile['sumw'][SR][sampname]
    signofwt = coffeafile['sum_signOf_genweights'][sampname]

    sumw = sumwtdict[list(sumwtdict.keys())[0]]
    print(f"\tEffective sumw = {sumw}")

    # thiswt = df['weight'].sum()
    # print(f"\tGenweight sum = {thiswt}")

    df['weight'] =  df['weight']/signofwt
    thiswt = ak.sum(df['weight'])
    print(f"\tReweighted sum = {thiswt}")
    return df

    #Following is to use on the signs of weights, i.e. +1 or -1 per event

    df['weight'] = np.where(df['weight']>0,1,-1)
    thiswt = df['weight'].sum()
    print(f"\tOnly signed weight sum = {thiswt}")

    df['weight'] *= sumw/thiswt
    thiswt = df['weight'].sum()
    print(f"\tReweighted sum = {thiswt}")
    return df

def addmissingcols(df):
    keys = df.fields
    ln = len(df)
    optionaljagged = ['events_LeptonGood_miniPFRelIso_all',
       'events_LeptonGood_pfRelIso03_all', 'events_LeptonGood_pt',
       'events_LeptonGood_eta', 'events_LeptonGood_phi',
       'events_LeptonGood_mass']
    optionalnames = ['events_W_m','events_LeptonCategory']

    rename = {"events_ll_pt" : "events_V_pt",
            "events_ll_eta" : "events_V_eta",
            "events_ll_phi" : "events_V_phi",
            "events_ll_mass" : "events_V_mass",
            "events_W_pt" : "events_V_pt",
            "events_W_eta" : "events_V_eta",
            "events_W_phi" : "events_V_phi",
            "events_W_mt" : "events_V_mass",
            "events_Z_pt" : "events_V_pt",
            "events_Z_eta" : "events_V_eta",
            "events_Z_phi" : "events_V_phi",
            "events_Z_m" : "events_V_mass",
            }

    for opt in optionalnames:
        if opt not in keys:
            df[opt] = 0
    for opt in optionaljagged:
        if opt not in keys:
            df[opt] = np.zeros([ln, 1]).tolist()
    for ren in rename:
        if ren in keys:
            df[rename[ren]] = df[ren]
    for key in df.fields:
        if key.startswith("events_"):
            df[key.replace("events_","")] = df[key] 

    cols =      ['EventNr',"JetGood_btagCvL","JetGood_btagCvB",
                "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                "V_pt","V_eta","V_phi","V_mass",
                "PuppiMET_pt","PuppiMET_phi","nPV","W_m",
                "LeptonCategory","weight"]

    return df[cols]

def getchannel(dirname):
    if "ZLL" in dirname:
        return "ZLL", 2
    if "WLNu" in dirname:
        return "WLNu", 1
    if "ZNuNu" in dirname:
        return "ZNuNu", 0
    raise NotImplementedError(f"Could not determine channel for directory {dirname}")

def getera(dirname):
    if "2022_preEE" in dirname:
        return 0
    if "2022_postEE" in dirname:
        return 1
    if "2023_preBPix" in dirname:
        return 2
    if "2023_postBPix" in dirname:
        return 3
    raise NotImplementedError(f"Could not determine era for directory {dirname}")

def getcols(channel,model_type='gnn'):
    cols = get_inputs(channel,model_type)
    if model_type == "gnn":
        colsMC = cols + ["weight"]
    else:
        colsMC = cols
    colsdata = cols
    return colsMC,colsdata

# Function to load data
def load_data(dir_path, test, coffea):
    signal_files = [] 

    if isinstance(dir_path,str):
        dir_path = [dir_path]
    alldirs = []
    channelnames = []
    for dr in dir_path:
        if "Saved_columnar_arrays" not in dr:
            subdirs = [d for d in glob.glob(f"{dr}/*") if "Saved_columnar_arrays" in d]
            if len(subdirs) == 0:
                raise ValueError(f"Input director {dr} is neither a `Saved_columnar_arrays*` directory, nor has a subdirectory of that name.")
            else:
                dr = subdirs[0]
        newlist = glob.glob(f"{dr}/*")
        alldirs.extend(newlist)
        thischannel = getchannel(dr)
        channelnames.extend([thischannel]*len(newlist))

    data_dfs = []
    bkg_dfs = []
    sig_dfs = []

    

    for idr, dr in enumerate(alldirs):
        samp = '/'.join(dr.split('/')[-2:])
        print(f"Process {samp}:")
        channel, ich = channelnames[idr]
        era = getera(dr)
        SR = get_SR_name(channel)
        colsMC,colsdata = getcols(channel,'gnn')        
        if "DATA_" in dr:
            print("\tLoading data is deprecated.")
            continue    #deprecate this
        else:
            coffeafile = load(f"{dr}/../../{coffea}")
            if test:
                if len(sig_dfs) >= 1 and "Hto2C" in dr: continue
                if len(bkg_dfs) >= 1 and "Hto2C" not in dr: continue

            thisdf = load_files(dr,SR,colsMC,test)
            thisdf = apply_norm(dr,SR,coffeafile,thisdf)
            del coffeafile
            thisdf = addmissingcols(thisdf)
            thisdf["channel"] = ich
            thisdf["era"] = era
            if "Hto2C" in dr:
                sig_dfs.append(thisdf)
                print(f"\tAdded to signal and assigned era {era}, channel {ich}.")
            else:
                bkg_dfs.append(thisdf)
                print(f"\tAdded to background and assigned era {era}, channel {ich}.")


    signal_df = ak.concatenate(sig_dfs)
    background_df = ak.concatenate(bkg_dfs)

    return signal_df, background_df


def splitdata(X,y,e=None):
    if e is None:
        print("Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        print("Using even entries for training.")
        train_indices = e[e % 2 == 0].index
        test_indices = e[e % 2 == 1].index
        
        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]
    return X_train, X_test, y_train, y_test


def evaluate_model(y, input_y_pred, input_y_wts=None, plot_dir=None, suff=""):

    y_pred = input_y_pred
    data_pred = None
    model_type = 'gnn'

    fpr, tpr, _ = roc_curve(y, y_pred,sample_weight=input_y_wts)
    auc = roc_auc_score(y, y_pred)
    if input_y_wts is None:        
        print(f'AUC: {auc}')
    else:
        print(f'AUC, unweighted: {auc}')
        fpr2, tpr2, _ = roc_curve(y, y_pred)

    loss = log_loss(y, y_pred)

    print(f'Log Loss: {loss}')

    if plot_dir is None:
        plot_dir = f'Models/{signal_name}/Plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure()
    plt.hist(y_pred[y == 0], bins=50, alpha=0.5, label='Background', density=True)
    plt.hist(y_pred[y == 1], bins=50, alpha=0.5, label='Signal', density=True)
    plt.xlabel('Score')
    plt.legend()
    plt.savefig(f'{plot_dir}/Score_{model_type}_{suff}.png')

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    if input_y_wts is not None: 
        plt.plot(fpr2, tpr2, label=f'ROC curve (unweighted)')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(suff)
    plt.legend(loc='lower right')
    plt.savefig(f'{plot_dir}/roc_curve_{model_type}_{suff}.png')

    plt.close()
    return auc


def resize_arr(arr,target):
    m, n, p = arr.shape
    
    if n > target:
        return arr[:, :target, :]
    elif n < target:
        padding = numpy.zeros([m, target - n, p])
        return numpy.concatenate([arr, padding], dim=1)
    else:
        return arr

def process_jagged(arr,maxelems=6):
    N = np.max(ak.num(arr, axis=1))  
    padded = ak.pad_none(arr, max(N,maxelems), axis=1)
    unzipped = ak.unzip(padded)
    padded = [ak.fill_none(p,0.).to_numpy() for p in unzipped]
    arr = np.array(padded)
    arr = np.transpose(arr, (1,2,0))
    arr = resize_arr(arr,maxelems)
    return arr

def process_gnn_inputs(df,intype="pd",removenans=True,verbose=False):
    #TODO: intype "ak" does not work yet

    import torch
    
    tensors = {}

    jetelems =  ["JetGood_btagCvL","JetGood_btagCvB",
                "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass"]
    lepelems =  ["LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass"]
    flatelems = ["V_pt","V_eta","V_phi","V_mass",
                "PuppiMET_pt","PuppiMET_phi","nPV","W_m",
                "LeptonCategory","channel","era"]
    
    X = df[jetelems + lepelems + flatelems]
    del df

    if verbose: print("Processing jets...")
    thisX = X[jetelems]  
    thisX = process_jagged(thisX,6)
    tensors["jet"] = torch.tensor(thisX[:,:,:2],dtype=torch.float32)
    tensors["jetP4"] = torch.tensor(thisX[:,:,2:],dtype=torch.float32)

    if verbose: print("Processing leptons...")
    thisX = X[lepelems]
    thisX = process_jagged(thisX,2)
    tensors["lep"] = torch.tensor(thisX[:,:,:2],dtype=torch.float32)
    tensors["lepP4"] = torch.tensor(thisX[:,:,2:],dtype=torch.float32)

    if verbose: print("Processing other variables...")
    thisX = ak.unzip(X[flatelems])
    thisX = [a.to_numpy() for a in thisX]
    thisX = np.transpose(np.array(thisX))


    tensors["VP4"] = torch.tensor(thisX[:,:4],dtype=torch.float32)
    tensors["global"] = torch.tensor(thisX[:,4:8],dtype=torch.float32)
    tensors["category"] = torch.tensor(thisX[:,8:],dtype=torch.int64)

    nanmask = None
    if removenans:
        for t in tensors:
            if torch.any(torch.isnan(tensors[t])):
                print(f"Spotted NaN value(s) in {t} tensor!")
                thismask = torch.any(torch.isnan(tensors[t]),-1)
                if len(thismask.shape) > 1:   #for 3-dim tensors
                    thismask = torch.any(thismask,-1)
                if nanmask is None:
                    nanmask = thismask
                else:
                    nanmask = (nanmask) | (thismask)
        if nanmask is not None:
            for t in tensors:
                tensors[t] = tensors[t][~nanmask]

    return tensors, nanmask

def split_list(lst, m):
    n = len(lst)
    avg_size = n // m  
    remainder = n % m 

    splits = []
    start = 0
    for i in range(m):
        end = start + avg_size + (1 if i < remainder else 0)  # Distribute remainder
        splits.append(lst[start:end])
        start = end
    
    return splits

def train_gnn(X, y, signame, test, w=None, e=None, hyperparameter=False, ntrials=20, loadmodel=None):
    import torch, optuna
    from gnnmodels import runGNNtraining

    print("Processing dataframes into tensors.")

    tensors, nanmask = process_gnn_inputs(X,verbose=True)
    del X
    gc.collect()
    if nanmask is not None:
        y = y[~nanmask.numpy()]
        if w is not None: w = w[~nanmask.numpy()]
        if e is not None: e = e[~nanmask.numpy()]

    outdir = f"Models/global/gnn"
    print("Will store model in:",outdir)

    if hyperparameter:        
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100, interval_steps=10)
        study = optuna.create_study(direction="minimize", pruner=pruner)
        ngpu = torch.cuda.device_count()
        print("NGPU =",ngpu)
        if ngpu==0:
            print("ERROR: Could not detect CUDA devices. Will set it to 4.")
            ngpu=4
        cpu_list = list(os.sched_getaffinity(0))
        print("Available cpus:",cpu_list)
        cpu_allocations = split_list(cpu_list,ngpu)
        func = lambda trial: runGNNtraining(tensors,y,outdir,test,w,e,trial=trial,ngpu=ngpu,cpulist=cpu_allocations)
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print(f"OPTUNA will run {ngpu} jobs in parallel.")
        print("Starting jobs...\n\n")
        study.optimize(func, n_trials=ntrials, n_jobs=ngpu)
        print("\n\nResult:", study.best_params)
    else:
        runGNNtraining(tensors,y,outdir,test,w,e,loadmodel=loadmodel)
        

# Main function to load data and train/evaluate model
def main():
    parser = argparse.ArgumentParser(description='Train or evaluate a GNN model.')
    parser.add_argument('--dir_path', '-d', type=str, nargs='+', help='Path to the directory containing input files.')
    parser.add_argument('--model', type=str, help='Path to the model file (for evaluation only).')
    parser.add_argument('--coffea', type=str, default='output_all.coffea', help='Name of the coffea hist file.')
    parser.add_argument('--test', action='store_true', help='Run with only one file.')
    parser.add_argument('--hyperparameter', action='store_true', help='Do hyperparameter optimization.')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter optimization.')
    parser.add_argument('--loadmodel', type=str, default=None, help='Load a previously trained .pt model.')
    args = parser.parse_args() 

    if args.loadmodel and args.hyperparameter:
        raise ValueError("Cannot do hyperparameter optimization while loadmodel is specified.") 

    signal_df, background_df= load_data(args.dir_path, args.test, args.coffea)
    
    common_signal_name = "Hto2C"

    signal_df['target'] = 1
    background_df['target'] = 0

    df = ak.concatenate([signal_df, background_df])    

    print(f"Length of signal_df: {len(signal_df)}")
    print(f"Length of background_df: {len(background_df)}")
    
    fields = df.fields

    X = df[[f for f in fields if f!='target' and f!='EventNr' and f!='weight']]
    y = df['target']
    e = df['EventNr']
    w = df["weight"]

    train_gnn(X, y, common_signal_name, args.test, w, e, args.hyperparameter, args.ntrials, args.loadmodel)

if __name__ == "__main__":
    main()
