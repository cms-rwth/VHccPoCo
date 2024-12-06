import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, Subset
from alive_progress import alive_bar
import numpy as np
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os, math
import matplotlib.pyplot as plt
import mplhep as hep
from contextlib import nullcontext
plt.style.use(hep.style.CMS)
import optuna
import copy, pickle, gc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GraphDataset(Dataset):
    def __init__(self, datadict, labels, weights=None, weightsigns=None, dvc=None):
        if dvc:
            for d in datadict:
                datadict[d] = datadict[d].to(dvc)
        self.data = datadict

        self.labels = labels
        if weights is None:
            self.weights = torch.ones_like(labels)
            if weightsigns is not None:
                self.weights = torch.where(weightsigns>=0, self.weights, self.weights*-1)
                print("Using signs of input weights.")
                print(f"There are {(self.weights>0).sum()} positive weights and {(self.weights<0).sum()} negative weights.")
        else:
            self.weights = weights

        bkgwt = (self.weights[self.labels==0]).sum()
        sigwt = (self.weights[self.labels==1]).sum()
        print(f"Total background weight: {bkgwt}, total signal weight: {sigwt}")
        self.weights = torch.where(self.labels==1, self.weights*bkgwt/sigwt, self.weights)

        bkgwt = (self.weights[self.labels==0]).sum()
        sigwt = (self.weights[self.labels==1]).sum()
        print(f"Total background weight after reweighting: {bkgwt}, total signal weight after reweighting: {sigwt}")
        
        if dvc:
            self.labels = self.labels.to(dvc)
            self.weights = self.weights.to(dvc)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        orderoftensors = ['jet','jetP4','lep','lepP4','VP4','global','category']
        return *[self.data[d][idx] for d in orderoftensors], self.labels[idx], self.weights[idx]

class GraphAttentionClassifier(nn.Module):
    def __init__(self, jet_dim=2, lep_dim=2, ll_dim=4, num_heads=4, attention_dim=128, cat_embed_dim=2, num_classes=2, globdim=4, dropout=0.2, pairwisefeats = 7, hyperembeddim=16):
        super(GraphAttentionClassifier, self).__init__()
        self.num_heads = num_heads
        self.pairwisefeats = pairwisefeats

        self.embedjets = nn.Linear(jet_dim+4+cat_embed_dim, hyperembeddim)
        self.multihead_attention_jet = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)

        self.edgejet = nn.Linear(pairwisefeats+cat_embed_dim, hyperembeddim)
        self.conv1d_edgejet = nn.Conv1d(in_channels=pairwisefeats, out_channels=1, kernel_size=1)
        self.multihead_attention_edgejet = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)        
        self.edgejetbn = nn.BatchNorm1d(1)

        
        self.embedleps = nn.Linear(lep_dim+4+cat_embed_dim, hyperembeddim)
        self.multihead_attention_lep = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)

        self.edgelep = nn.Linear(pairwisefeats+cat_embed_dim, hyperembeddim)
        self.multihead_attention_edgelep = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)     

        self.embedll = nn.Linear(ll_dim+cat_embed_dim, hyperembeddim)     
        
        self.embedjl = nn.Linear(pairwisefeats+cat_embed_dim, hyperembeddim)
        self.multihead_attention_jl = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)
        self.embedjjl = nn.Linear(pairwisefeats+cat_embed_dim, hyperembeddim)
        self.multihead_attention_jjl = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)

        self.embedjll = nn.Linear(pairwisefeats+cat_embed_dim, hyperembeddim)
        self.multihead_attention_jll = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)
        self.embedjjll = nn.Linear(pairwisefeats+cat_embed_dim, hyperembeddim)
        self.multihead_attention_jjll = nn.MultiheadAttention(embed_dim=hyperembeddim, num_heads=num_heads, batch_first=True)

        self.embedglob = nn.Linear(globdim+cat_embed_dim, hyperembeddim)

        self.layer_norm = nn.LayerNorm(hyperembeddim)

        catlen = 2+3+4
        self.embedcat = nn.Linear(catlen, cat_embed_dim)

        self.fc1 = nn.Linear(hyperembeddim*10+catlen, attention_dim)
        self.bn1 = nn.BatchNorm1d(attention_dim)
        self.fc2 = nn.Linear(attention_dim+catlen, attention_dim)
        self.bn2 = nn.BatchNorm1d(attention_dim)
        self.fc3 = nn.Linear(attention_dim+catlen, attention_dim)
        self.bn3 = nn.BatchNorm1d(attention_dim)
        self.fc4 = nn.Linear(attention_dim+catlen, num_classes)

        self.dropout = nn.Dropout(dropout)
    
    def selfinteraction(self, x):
        pT = x[:, :, 0]        
        eta = x[:, :, 1]
        phi = x[:, :, 2]
        mass = x[:, :, 3]

        # Compute momentum components
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)

        # Compute energy for each node
        energy = torch.sqrt(pT**2 + pz**2 + mass**2)

        # Compute pairwise momentum and energy differences
        px1 = px.unsqueeze(2)
        py1 = py.unsqueeze(2)
        pz1 = pz.unsqueeze(2)
        E1 = energy.unsqueeze(2)

        px2 = px.unsqueeze(1)
        py2 = py.unsqueeze(1)
        pz2 = pz.unsqueeze(1)
        E2 = energy.unsqueeze(1)


        # Compute invariant mass squared
        E_sum = E1 + E2 
        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2
        p_sum2 = px_sum**2 + py_sum**2 + pz_sum**2

        invariant_mass_squared = E_sum**2 - p_sum2
        invariant_mass_squared = torch.clamp(invariant_mass_squared, min=0.0)

        pairwise_pT = torch.sqrt(px_sum**2 + py_sum**2 + 1e-10)
        

        # Combined eta and phi of the pair
        combined_eta = torch.arctanh(pz_sum/torch.sqrt(p_sum2+1e-10))
        combined_eta = torch.where(torch.isinf(combined_eta),20,combined_eta)
        combined_phi = torch.atan2(py_sum, px_sum)

        # Create mask for invalid edges
        nodemask = torch.all(x == 0, dim=-1).bool()
        edgemask = nodemask.unsqueeze(1) | nodemask.unsqueeze(2) 

        
        deta = eta.unsqueeze(2) - eta.unsqueeze(1)
        dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi  # Wrap to range [-pi, pi]
        distances = torch.sqrt(deta**2 + dphi**2)

        # Compute k_T
        pT1 = pT.unsqueeze(2)  # shape [batch, n, 1]
        pT2 = pT.unsqueeze(1)  # shape [batch, 1, n]
        min_pT = torch.min(pT1, pT2)  # shape [batch, n, n]
        kT = min_pT * distances

        # Compute z
        sum_pT = pT1 + pT2
        z = min_pT / sum_pT

        # Apply clamp to avoid issues with log(0)
        distances = torch.clamp(distances, min=1e-10)
        kT = torch.clamp(kT, min=1e-10)
        z = torch.clamp(z, min=1e-10)
        invariant_mass_squared = torch.clamp(invariant_mass_squared, min=1e-10)
        pairwise_pT = torch.clamp(pairwise_pT, min=1e-10)

        # Compute logarithms of the features
        log_distances = torch.log(distances)
        log_kT = torch.log(kT)
        log_z = torch.log(z)
        log_invariant_mass_squared = torch.log(invariant_mass_squared)
        log_pairwise_pT = torch.log(pairwise_pT)

        # Combine logarithms of edge features
        ftlist = [log_distances, log_kT, log_z, log_invariant_mass_squared, log_pairwise_pT, combined_eta, combined_phi]
        log_edge_features = torch.stack(ftlist[:self.pairwisefeats], dim=-1)
        edgeP4 = torch.stack([pairwise_pT, combined_eta, combined_phi, torch.sqrt(invariant_mass_squared)], dim=-1)

        # Apply mask to edge features
        log_edge_features_masked = log_edge_features.masked_fill(edgemask.unsqueeze(-1), -1e6)

        return log_edge_features_masked, edgeP4


    def pairwise_interaction(self, x, y):
        # Extract features for x and y
        pT_x, eta_x, phi_x, mass_x = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
        pT_y, eta_y, phi_y, mass_y = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]

        # Compute momentum components for x and y
        px_x = pT_x * torch.cos(phi_x)
        py_x = pT_x * torch.sin(phi_x)
        pz_x = pT_x * torch.sinh(eta_x)
        px_y = pT_y * torch.cos(phi_y)
        py_y = pT_y * torch.sin(phi_y)
        pz_y = pT_y * torch.sinh(eta_y)

        # Compute energy for each node in x and y
        energy_x = torch.sqrt(pT_x**2 + pz_x**2 + mass_x**2)
        energy_y = torch.sqrt(pT_y**2 + pz_y**2 + mass_y**2)

        # Compute pairwise momentum and energy sums for invariant mass and kinematic properties
        px1, py1, pz1, E1 = px_x.unsqueeze(2), py_x.unsqueeze(2), pz_x.unsqueeze(2), energy_x.unsqueeze(2)
        px2, py2, pz2, E2 = px_y.unsqueeze(1), py_y.unsqueeze(1), pz_y.unsqueeze(1), energy_y.unsqueeze(1)

        # Calculate invariant mass squared
        E_sum = E1 + E2
        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2
        p_sum2 = px_sum**2 + py_sum**2 + pz_sum**2
        invariant_mass_squared = E_sum**2 - p_sum2
        invariant_mass_squared = torch.clamp(invariant_mass_squared, min=0.0)

        # Calculate pairwise transverse momentum (pT)
        pairwise_pT = torch.sqrt(px_sum**2 + py_sum**2 + 1e-10)

        # Combined eta and phi for the pair
        combined_eta = torch.arctanh(pz_sum / torch.sqrt(p_sum2 + 1e-10))
        combined_eta = torch.where(torch.isinf(combined_eta),20,combined_eta)
        combined_phi = torch.atan2(py_sum, px_sum)

        # Create mask for invalid edges
        nodemask_x = torch.all(x == 0, dim=-1).bool()
        nodemask_y = torch.all(y == 0, dim=-1).bool()
        edgemask = nodemask_x.unsqueeze(2) | nodemask_y.unsqueeze(1)

        # Calculate pairwise differences in eta and phi
        deta = eta_x.unsqueeze(2) - eta_y.unsqueeze(1)
        dphi = phi_x.unsqueeze(2) - phi_y.unsqueeze(1)
        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi  # Wrap to range [-pi, pi]
        distances = torch.sqrt(deta**2 + dphi**2)

        # Compute k_T
        pT1 = pT_x.unsqueeze(2)
        pT2 = pT_y.unsqueeze(1)
        min_pT = torch.min(pT1, pT2)
        kT = min_pT * distances

        # Compute z
        sum_pT = pT1 + pT2
        z = min_pT / sum_pT

        # Apply clamp to avoid issues with log(0)
        distances = torch.clamp(distances, min=1e-10)
        kT = torch.clamp(kT, min=1e-10)
        z = torch.clamp(z, min=1e-10)
        invariant_mass_squared = torch.clamp(invariant_mass_squared, min=1e-10)
        pairwise_pT = torch.clamp(pairwise_pT, min=1e-10)

        # Compute logarithms of the features
        log_distances = torch.log(distances)
        log_kT = torch.log(kT)
        log_z = torch.log(z)
        log_invariant_mass_squared = torch.log(invariant_mass_squared)
        log_pairwise_pT = torch.log(pairwise_pT)

        # Combine logarithms of edge features
        ftlist = [log_distances, log_kT, log_z, log_invariant_mass_squared, log_pairwise_pT, combined_eta, combined_phi]
        log_edge_features = torch.stack(ftlist[:self.pairwisefeats], dim=-1)
        edgeP4 = torch.stack([pairwise_pT, combined_eta, combined_phi, torch.sqrt(invariant_mass_squared)], dim=-1)

        # Apply mask to edge features
        log_edge_features_masked = log_edge_features.masked_fill(edgemask.unsqueeze(-1), -1e6)

        return log_edge_features_masked, edgeP4

    def makeUT(self,tensor):
        nnodes = tensor.shape[1]
        indices = torch.triu_indices(nnodes, nnodes, offset=1, dtype=torch.int32)
        if len(tensor.shape) == 3:
            return tensor[:, indices[0], indices[1]]
        return tensor[:, indices[0], indices[1], :]
        

    def pairprocess(self,p4_1,p4_2,embed,mha=None,pool=True,catembed=None):
        ij, _ = self.pairwise_interaction(p4_1,p4_2)
        ij = ij.reshape(ij.shape[0],ij.shape[1]*ij.shape[2],ij.shape[3])
        if catembed is not None:
            catinput = catembed.unsqueeze(1).expand(-1, ij.shape[1], -1)
            ij = torch.cat([ij,catinput],dim=2)
        ij_embed = embed(ij)
        if mha is not None:
            ij_embed, _ = mha(ij_embed,ij_embed,ij_embed)
            ij_embed = self.layer_norm(ij_embed)
        if pool:
            ij_pooled = torch.mean(ij_embed, dim=1)
            return ij_pooled
        return ij_embed

    def forward(self, jet, jetp4, lep, lepp4, ll, glo, cat):
        #categorical
        lepcat = F.one_hot(cat[:,0],num_classes=2).float()
        channelcat = F.one_hot(cat[:,1],num_classes=3).float()
        eracat = F.one_hot(cat[:,2],num_classes=4).float()

        #categorical
        if len(lepcat.shape) == 1:          #This happens when batchsize is 1
            lepcat = lepcat.unsqueeze(0)
            channelcat = lepcat.unsqueeze(0)
            eracat = lepcat.unsqueeze(0)

        catconc = torch.cat([lepcat,channelcat,eracat], dim=1)
        cat_embed = self.embedcat(catconc)

        # Jets
        jet_edge_features, dijetP4 = self.selfinteraction(jetp4)            # BxNxNxF
        nnodes = jet_edge_features.shape[1]
        jet_edge_features_conc = jet_edge_features.view(jet_edge_features.shape[0], nnodes*nnodes, jet_edge_features.shape[3])   # BxN^2xF

        e_attn = self.conv1d_edgejet(jet_edge_features_conc.permute(0,2,1))      #BxFxN^2
        e_attn = F.relu(self.edgejetbn(e_attn))
        e_attn = e_attn.view(e_attn.shape[0],e_attn.shape[1],nnodes,nnodes) #BxFxNxN
        e_attn = e_attn.repeat(1, self.num_heads, 1, 1)
        e_attn = e_attn.view(e_attn.shape[0]*e_attn.shape[1],e_attn.shape[2],e_attn.shape[3])  #B*FxNxN

        jetmask = torch.all(jet == 0, dim=-1)
        edgemask = jetmask.unsqueeze(1) | jetmask.unsqueeze(2)
        # edgemask = edgemask.view(edgemask.shape[0],edgemask.shape[1]*edgemask.shape[2])     #BxN^2
        
        catinput = cat_embed.unsqueeze(1).expand(-1, jet.shape[1], -1)
        jet = torch.cat([jet,jetp4,catinput],dim=2)
        jet = self.embedjets(jet)
        attn_output_jet, _ = self.multihead_attention_jet(jet, jet, jet, key_padding_mask=jetmask.float(), attn_mask=e_attn)
        attn_output_jet = self.layer_norm(attn_output_jet)
        pooled_output_jets = torch.sum(attn_output_jet, dim=1)

        
        jet_edge_features = self.makeUT(jet_edge_features)
        edgemask = self.makeUT(edgemask)
        catinput = cat_embed.unsqueeze(1).expand(-1, jet_edge_features.shape[1], -1)
        e = self.edgejet(torch.cat([jet_edge_features,catinput],dim=2))
        attn_output_edgejet, _ = self.multihead_attention_edgejet(e, e, e, key_padding_mask=edgemask)
        attn_output_edgejet = self.layer_norm(attn_output_edgejet)        
        pooled_output_edgejet = torch.mean(attn_output_edgejet, dim=1)


        # Leps
        lep_edge_features, _ = self.selfinteraction(lepp4)
        # nnodes = lep_edge_features.shape[1]
        # lep_edge_features_avg = lep_edge_features.view(lep_edge_features.shape[0], nnodes*nnodes, lep_edge_features.shape[3])

        catinput = cat_embed.unsqueeze(1).expand(-1, lep.shape[1], -1)
        lep = torch.cat([lep,lepp4,catinput],dim=2)
        lep_emb = self.embedleps(lep)
        attn_output_lep, _ = self.multihead_attention_lep(lep_emb, lep_emb, lep_emb)
        attn_output_lep = self.layer_norm(attn_output_lep)
        pooled_output_leps = torch.sum(attn_output_lep, dim=1)


        lep_edge_features = self.makeUT(lep_edge_features)
        catinput = cat_embed.unsqueeze(1).expand(-1, lep_edge_features.shape[1], -1)
        e = self.edgelep(torch.cat([lep_edge_features,catinput],dim=2))
        attn_output_edgelep, _ = self.multihead_attention_edgelep(e, e, e)
        attn_output_edgelep = self.layer_norm(attn_output_edgelep)        
        pooled_output_edgelep = torch.mean(attn_output_edgelep, dim=1)

        # ll; particularly important for 0L channel
        ll_emb = self.embedll(torch.cat([ll,cat_embed],dim=1))

        jjp4 = self.makeUT(dijetP4)
        #jet with lep interactions
        jl_out = self.pairprocess(jetp4,lepp4,self.embedjl,self.multihead_attention_jl,catembed=cat_embed)

        # jj with lep interactions            
        jjl_out = self.pairprocess(jjp4,lepp4,self.embedjjl,self.multihead_attention_jjl,catembed=cat_embed)

        # j with ll interactions
        ll = ll.unsqueeze(1)
        jll_out = self.pairprocess(jetp4,ll,self.embedjll,self.multihead_attention_jll,catembed=cat_embed)

        # jj with ll interactions
        jjll_out = self.pairprocess(jjp4,ll,self.embedjjll,self.multihead_attention_jjll,catembed=cat_embed)

        #global
        global_out = self.embedglob(torch.cat([glo,cat_embed],dim=1))

        mhalist = [pooled_output_jets,pooled_output_edgejet,pooled_output_leps,pooled_output_edgelep,ll_emb,jl_out,jjl_out,jll_out,jjll_out,global_out,catconc]
        allout = torch.cat(mhalist,dim=1)

        # Feedforward neural network
        x = self.fc1(allout)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = torch.cat([x,catconc],dim=1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = torch.cat([x,catconc],dim=1)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = torch.cat([x,catconc],dim=1)
        x = self.fc4(x)
        
        return torch.sigmoid(x)

def getinputs(data,device):
    jet,jetp4,lep,lepp4,llp4,glo,cat,label,weight = data
    jet,jetp4,lep,lepp4,llp4,glo,cat,label,weight = jet.to(device),jetp4.to(device),lep.to(device),lepp4.to(device),llp4.to(device),glo.to(device),cat.to(device),label.to(device),weight.to(device)
    return jet,jetp4,lep,lepp4,llp4,glo,cat,label,weight

def runGNNtraining(tensordict, y, outdir, test=False, w=None, e=None, weightedsampling=False, trial=None, ngpu=1, cpulist=None, loadmodel=None):
    from training import evaluate_model
    outhandle = outhandler(outdir)
    outhandle.addcustom("signweighted")
    ncpu=4

    if trial is not None:
        lr = trial.suggest_float('lr', 5e-3, 1e-2)
        dropout = 0 #trial.suggest_float('dropout', 0., 0.1)
        attention_dim = trial.suggest_int('attn_dim', 128, 512, step=64)
        hyperembeddim = trial.suggest_int('hyper_dim', 16, 48, step=8)

        if not torch.cuda.is_available():
            raise NotImplementedError("Hyperparameter optimization expects GPU availability.")

        if ngpu > 1:
            gpu_id = trial.number % ngpu  # ngpu GPUs, alternate between GPU 0 and GPU 1
            device = torch.device(f"cuda:{gpu_id}") # Set the GPU for this trial
            print(f"Optuna trial number {trial.number}, using device cuda:{gpu_id}")
            
            outhandle.addcustom(f"hyperparameter_LR{lr}_dropout{dropout}_attn{attention_dim}_emb{hyperembeddim}")

            # Set CPU affinity: Allow ncpu CPUs per job
            process_id = os.getpid()
            cputouse = cpulist[gpu_id]
            ncpu = len(cputouse)
            os.sched_setaffinity(process_id, cputouse)
            print(f"Optuna trial number {trial.number}, using cpus:",cputouse)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr = outhandle.getenv("LR",0.007)
        dropout = outhandle.getenv("DROPOUT",0.)
        attention_dim = int(outhandle.getenv("ATTENTION",512))        
        hyperembeddim = int(outhandle.getenv("DIM",48))
    # if test:
    print("Device:", device)

    y = torch.tensor(y.to_numpy(),dtype=torch.float32)
    if w is not None:
        w = torch.tensor(w.to_numpy(),dtype=torch.float32)
    if e is not None:
        e = torch.tensor(e.to_numpy(),dtype=torch.int32)

    graph = GraphDataset(tensordict, y, w) 

    if device=="cpu":
        nloaders = os.cpu_count()
    else:
        nloaders = ncpu

    nepochs = 200
    schedulerpatience = 5
    decayrate = 0.7
    earlystop = 20
    reportevery = 1

    batch_size = 1024*64
    val_batch_size = 1024*32
    if test:
        batch_size = 16
        val_batch_size = 16
    

    print(f"Will use {nloaders} cpus for dataloaders.")

    if loadmodel is None:
        modelparams = {"attention_dim": attention_dim, "hyperembeddim": hyperembeddim, "dropout": dropout}        
    else:
        modeldir = '/'.join(loadmodel.split('/')[:-1])
        modelparams = pickle.load(open(f'{modeldir}/modelparams.pkl','rb'))
        print("Loaded saved model parameters.")

    print("Model parameters:",modelparams)
    model = GraphAttentionClassifier(**modelparams).to(device)

    modelsize = count_parameters(model)
    print("Using model of size:",modelsize)
    
    '''
    TODO: Currently multi-GPU training does not work; it requires memory-contiguous tensors and fails in the makeUT function:
    line 283, in makeUT
    return tensor[:, indices[0], indices[1], :]
    RuntimeError: CUDA error: misaligned address
    '''
    # if torch.cuda.device_count() > 1:
    #     print(f"Will use {torch.cuda.device_count()} GPUs.")
    #     model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decayrate, patience = schedulerpatience)

    if weightedsampling:
        criterion = nn.BCELoss()
    else:
        criterion = nn.BCELoss(reduction="none")

    if e is not None:
        print("Splitting events by even and odd.")
        train_indices = (e % 2 == 0).nonzero(as_tuple=True)[0]
        val_indices = (e % 2 == 1).nonzero(as_tuple=True)[0]

        train_size = len(train_indices)
        val_size = len(val_indices)
        print(f"Train size: {train_size}, Val size: {val_size}")

        # Create subsets of the dataset for training and validation
        train_subset = Subset(graph, train_indices)
        val_subset = Subset(graph, val_indices)
    else:
        print("Using random split.")
        trainfrac = 0.8
        dataset_size = len(graph)
        train_size = int(trainfrac * dataset_size)
        val_size = dataset_size - train_size
        train_subset, val_subset = random_split(graph, [train_size, val_size])
        train_indices = train_subset.indices

    if weightedsampling:     
        print("Using weighted sampling. Will use non-weighted loss.")
        train_weights = graph.weights[train_indices]
        train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=train_size, replacement=True)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, num_workers=nloaders, pin_memory=True)    
    else:
        print("Using normal sampling. Will use weighted loss.")
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=nloaders, pin_memory=True)

    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False, num_workers=nloaders, pin_memory=True)

    print(f"Train data size to model size ratio: {train_size/modelsize}")
    if train_size/modelsize < 10:
        print(f"WARNING: Training data size={train_size} is less than 10 times the modelsize={modelsize}!!!")

    bestloss = 1e9
    bestepoch = 0
    bestmodel = None
    trainlosses = []
    vallosses = []
    bestlosses = []
    lrs = []

    print("Using batch size",batch_size)
    

    if not loadmodel:
        outdir = outhandle.outdir
        print("Working dir:", outdir)   
        os.system(f"mkdir -p {outdir}/checkpoints")

        pickle.dump(modelparams,open(f"{outdir}/modelparams.pkl",'wb'))

        with nullcontext() as epochbar:
            for iepoch in range(nepochs):
                model.train()
                total_loss = 0.
                with alive_bar(len(train_loader),title=f"Epoch {iepoch} training") if trial is None else nullcontext() as batchbar:
                    for data in train_loader:
                        jet,jetp4,lep,lepp4,llp4,glo,cat,label,weight = getinputs(data,device)

                        optimizer.zero_grad()
                        outputs = model(jet,jetp4,lep,lepp4,llp4,glo,cat)
                        if weightedsampling:
                            loss = criterion(outputs[:,0], label)
                        else:
                            loss = criterion(outputs[:,0], label)*weight
                            loss = loss.mean()

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.) 
                        optimizer.step()
                        total_loss += loss.item()

                        if trial is None: batchbar()

                model.eval()
                val_loss = 0.
                with alive_bar(len(val_loader),title=f"Epoch {iepoch} validating") if trial is None else nullcontext() as batchbar:
                    for data in val_loader:
                        jet,jetp4,lep,lepp4,llp4,glo,cat,label,weight = getinputs(data,device)
                        with torch.no_grad():
                            outputs = model(jet,jetp4,lep,lepp4,llp4,glo,cat)
                            if weightedsampling:
                                loss = criterion(outputs[:,0], label)
                            else:
                                loss = criterion(outputs[:,0], label)*weight
                                loss = loss.mean()

                            val_loss += loss.item()

                        if trial is None: batchbar()

                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                if avg_val_loss < bestloss:
                    bestloss = avg_val_loss
                    bestepoch = iepoch
                    bestmodel = copy.deepcopy(model.state_dict())

                if trial is not None:
                    trial.report(avg_val_loss, iepoch)
                    # Check if the trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                scheduler.step(avg_val_loss)
                nowlr = optimizer.param_groups[-1]['lr']
                status = f"LR: {nowlr:.6f}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Best Val Loss: {bestloss:.6f}, Best epoch: {bestepoch}"
                if trial is not None:
                    status = f"Trial: {trial.number}, Epoch: {iepoch}, "+ status

                trainlosses.append(avg_train_loss)
                vallosses.append(avg_val_loss)
                bestlosses.append(bestloss)
                lrs.append(nowlr)

                if iepoch > 0 and iepoch % reportevery == 0:
                    print(status)
                    dumploss(trainlosses,bestlosses,f"{outdir}/losses_inprogress.png",lrcurve=lrs,valloss=vallosses)
                    torch.save(bestmodel,f"{outdir}/checkpoints/gnn_{iepoch}.pt")


                if iepoch - bestepoch > earlystop:
                    print(f"Loss did not improve in {earlystop} epochs. Exiting.")
                    break

                # if trial is None:
                #     epochbar.text(status)
                #     epochbar()

        print(status)

        torch.save(bestmodel,f"{outdir}/gnn.pt")
        print("Model saved:",f"{outdir}/gnn.pt")
        dumploss(trainlosses,bestlosses,f"{outdir}/losses.png",lrcurve=lrs,valloss=vallosses)

        model.load_state_dict(bestmodel)
        # torch.save(torch.jit.script(model),f"{outdir}/model.pt")  #jit does not work yet
    else:
        print("Loading existing model from",loadmodel)
        outdir = '/'.join(loadmodel.split('/')[:-1])
        print("Working dir:", outdir)   
        bestmodel = torch.load(loadmodel,weights_only=True,map_location=device)
        model.load_state_dict(bestmodel)

        # pickle.dump(modelparams,open(f"{outdir}/modelparams.pkl",'wb'))
        # if not os.path.isfile(f"{outdir}/model.pt"):
        #     torch.save(torch.jit.script(model),f"{outdir}/model.pt")

    # Eval    
    model.eval()
    allouts = []
    truth = []
    wts = []

    for ch in [0,1,2]:
        for era in [0,1]:
            val_indices = ((e % 2 == 1) & (graph.data["category"][:,1]==ch) & (graph.data["category"][:,2]==era)).nonzero(as_tuple=True)[0]
            val_subset = Subset(graph, val_indices)    
            val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False, num_workers=nloaders, pin_memory=True)
            for data in val_loader:
                jet,jetp4,lep,lepp4,llp4,glo,cat,label,weight = getinputs(data,device)
                with torch.no_grad():
                    outputs = model(jet,jetp4,lep,lepp4,llp4,glo,cat)
                    allouts.append(outputs[:,0])
                    truth.append(label)
                    wts.append(weight)
            out = torch.cat(allouts,dim=0).cpu()
            yvals = torch.cat(truth,dim=0).cpu()
            ywts = torch.cat(wts,dim=0).cpu()
            # return out,yvals,ywts,outdir

            auc = evaluate_model(yvals, input_y_pred=out, input_y_wts=ywts, plot_dir=f"{outdir}/Plots",suff=f"ch{ch}_era{era}")

    if trial is not None:
        del model,jet,jetp4,lep,lepp4,llp4,glo,cat,label,weight
        gc.collect()
        return bestloss
        # return auc

class outhandler():
    def __init__(self,outdir):
        self.outdir = outdir

    def addcustom(self,txt=""):
        self.outdir += f"_{txt}"

    def getenv(self,envvarname,default):
        #Useful for hyperparameter optimization
        if os.getenv(envvarname):
            out = os.getenv(envvarname)
            print(f"Setting {envvarname} to {out}.")            
            newval = float(out)
        else:
            newval = default
        self.outdir += f"_{envvarname}{newval}"
        return newval

def dumploss(loss,minloss,outfl,lrcurve=None,valloss=None):
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(range(1,len(loss)+1), loss, label="Loss")
    valtxt = ""
    if valloss:
        ax1.plot(range(1,len(valloss)+1), valloss, label=f"Val loss")
        valtxt = "val "
    ax1.plot(range(1,len(minloss)+1), minloss, label=f"Best {valtxt}loss\n({minloss[-1]:.6f})")
    
    plt.title('Loss Curve')
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("# epoch")
    # ax1.set_yscale("log")
    ax1.legend()
    if lrcurve:        
        ax2 = ax1.twinx()
        ax2.plot(range(1,len(lrcurve)+1), lrcurve, label="LR",color = "gray", alpha=0.5)
        ax2.set_ylabel('Learning Rate', color='gray')
        ax2.set_yscale("log")
        ax2.legend(loc=9)    
    plt.tight_layout()
    plt.savefig(outfl)
    plt.close()
