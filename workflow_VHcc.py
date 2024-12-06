import awkward as ak
import numpy as np
import uproot
import pandas as pd
import math
import warnings
import os, pickle
import lightgbm as lgb
import gc
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
import CommonSelectors
from CommonSelectors import *
import inspect

import torch
from MVA.gnnmodels import GraphAttentionClassifier
from MVA.training import process_gnn_inputs

from pocket_coffea.utils.utils import dump_ak_array
from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis
from pocket_coffea.lib.deltaR_matching import delta_phi
from pocket_coffea.lib.objects import (
    jet_correction,
    lepton_selection,
    jet_selection,
    btagging,
    CvsLsorted,
    get_dilepton,
    get_dijet
)
import awkward as ak
import numpy as np

def get_nu_4momentum(Lepton, PuppiMET):
    mW = 80.38

    # Convert pt, eta, phi, m to px, py, pz, E
    px = Lepton.pt * np.cos(Lepton.phi)
    py = Lepton.pt * np.sin(Lepton.phi)
    pz = Lepton.pt * np.sinh(Lepton.eta)
    E = np.sqrt(Lepton.mass**2 + Lepton.pt**2 * np.cosh(Lepton.eta)**2)
    
    MET_px = PuppiMET.pt * np.cos(PuppiMET.phi)
    MET_py = PuppiMET.pt * np.sin(PuppiMET.phi)
    
    MisET2 = (MET_px**2 + MET_py**2)
    mu = (mW**2) / 2 + MET_px * px + MET_py * py
    a = (mu * pz) / (E**2 - pz**2)
    a2 = a**2
    b = ((E**2) * (MisET2) - mu**2) / (E**2 - pz**2)
    
    condition = a2 - b >= 0

    # Vectorized handling of conditions
    root = np.sqrt(ak.where(condition, a2 - b, ak.zeros_like(a2)))
    pz1 = a + root
    pz2 = a - root
    pznu = ak.where(np.abs(pz1) < np.abs(pz2), pz1, pz2)
    Enu = np.sqrt(MisET2 + pznu**2)

    # Handle cases where condition is False using your fallback logic
    # Adapted to take into account the real parts of the roots if discriminant is negative
    real_part = ak.where(condition, ak.zeros_like(a), a)  # Use 'a' as the real part when condition is False
    pznu = ak.where(condition, pznu, real_part)  # Update pznu to use real_part when condition is False
    Enu = np.sqrt(MisET2 + pznu**2)  # Recalculate Enu with the updated pznu

    p4nu_rec = ak.Array([MET_px, MET_py, pznu, Enu])
    pt = np.sqrt(MET_px**2 + MET_py**2)
    phi = np.arctan2(MET_py, MET_px)
    theta = np.arctan2(pt, pznu)
    eta = -np.log(np.tan(theta / 2))
    m = np.sqrt(np.maximum(Enu**2 - (MET_px**2 + MET_py**2 + pznu**2), 0))

    return ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": m},with_name="PtEtaPhiMCandidate")

class VHccBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

        self.proc_type   = self.params["proc_type"]
        self.save_arrays = self.params["save_arrays"]
        self.run_dnn     = self.params["run_dnn"]
        self.run_gnn     = self.params["run_gnn"]
        #self.bdt_model = lgb.Booster(model_file=self.params.LightGBM_model)
        #self.bdt_low_model = lgb.Booster(model_file=self.params.LigtGBM_low)
        #self.bdt_high_model = lgb.Booster(model_file=self.params.LigtGBM_high)
        #self.dnn_model = load_model(self.params.DNN_model)
        #self.dnn_low_model = load_model(self.params.DNN_low)
        #self.dnn_high_model = load_model(self.params.DNN_high)
    
        # Define the prediction functions with @tf.function
        #self.predict_dnn = tf.function(self.model.predict, reduce_retracing=True)
        #self.predict_dnn_low = tf.function(self.model_low.predict, reduce_retracing=True)
        #self.predict_dnn_high = tf.function(self.model_high.predict, reduce_retracing=True)

        if self.run_gnn:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Processor initialized")
        
    def apply_object_preselection(self, variation):
        '''
        
        '''
        # Include the supercluster pseudorapidity variable
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        # Build masks for selection of muons, electrons, jets, fatjets
        self.events["MuonGood"] = lepton_selection(
            self.events, "Muon", self.params
        )
        self.events["ElectronGood"] = lepton_selection(
            self.events, "Electron", self.params
        )
        leptons = ak.with_name(
            ak.concatenate((self.events.MuonGood, self.events.ElectronGood), axis=1),
            name='PtEtaPhiMCandidate',
        )
        self.events["LeptonGood"] = leptons[ak.argsort(leptons.pt, ascending=False)]

        self.events["ll"] = get_dilepton(
            self.events.ElectronGood, self.events.MuonGood
        )
        
        myJetTagger = self.params.ctagging[self._year]["tagger"]
        self.myJetTagger = myJetTagger

        
        
        self.newjetdefiniton = "jet_tagger" in inspect.signature(jet_selection).parameters
        # This is a hack for now until https://github.com/PocketCoffea/PocketCoffea/pull/276/ is merged
        if self.newjetdefiniton:
            self.events["JetGood"], self.jetGoodMask = jet_selection(
                self.events, "Jet", self.params, self._year, "LeptonGood", myJetTagger
            )
        else:
            self.events["JetGood"], self.jetGoodMask = jet_selection(
                self.events, "Jet", self.params, self._year, "LeptonGood"
            )
            if self.myJetTagger == "PNet":
                B   = "btagPNetB"
                CvL = "btagPNetCvL"
                CvB = "btagPNetCvB"
            elif self.myJetTagger == "DeepFlav":
                B   = "btagDeepFlavB"
                CvL = "btagDeepFlavCvL"
                CvB = "btagDeepFlavCvB"
            elif self.myJetTagger == "RobustParT":
                B   = "btagRobustParTAK4B"
                CvL = "btagRobustParTAK4CvL"
                CvB = "btagRobustParTAK4CvB"
            else:
                raise NotImplementedError(f"This tagger is not implemented: {self.myJetTagger}")
            self.events.JetGood["btagB"] = self.events.JetGood[B]
            self.events.JetGood["btagCvL"] = self.events.JetGood[CvL]
            self.events.JetGood["btagCvB"] = self.events.JetGood[CvB]
            
        
        self.events['EventNr'] = self.events.event
        
        self.events["BJetGood"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.bJetWP)
            
    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nLeptonGood"] = ak.num(self.events.LeptonGood)

        self.events["nJet"] = ak.num(self.events.Jet)
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        self.events["nBJetGood"] = ak.num(self.events.BJetGood)
        # self.events["nfatjet"]   = ak.num(self.events.FatJetGood)


    def evaluateBDT(self, data):
        #print(self.params.Models.BDT[self.channel][self.events.metadata["year"]].model_file)
        #print()
        # Read the model file
        model = lgb.Booster(model_file=self.params.Models.BDT[self.channel][self.events.metadata["year"]].model_file)
        #bdt_score = self.bdt_model.predict(data)
        bdt_score = model.predict(data)
        # Release memory
        del model
        gc.collect()

        return bdt_score
    
    def evaluateseparateBDTs(self, data):
        data_df_low = data[data['dilep_pt'] < 150]
        data_df_high = data[data['dilep_pt'] >= 150]
        
        # Initialize empty arrays for scores
        bdt_score_low = np.array([])
        bdt_score_high = np.array([])
        
        # Read the model files
        model_low = lgb.Booster(model_file=self.params.Models.BDT[f'{self.channel}_low'][self.events.metadata["year"]].model_file)
        model_high = lgb.Booster(model_file=self.params.Models.BDT[f'{self.channel}_high'][self.events.metadata["year"]].model_file)
        
        # Predict only if data_df_low is non-empty
        if not data_df_low.empty:
            #bdt_score_low = self.bdt_low_model.predict(data_df_low)
            bdt_score_low = model_low.predict(data_df_low)
        
        # Predict only if data_df_high is non-empty
        if not data_df_high.empty:
            #bdt_score_high = self.bdt_high_model.predict(data_df_high)
            bdt_score_high = model_high.predict(data_df_high)
        
        # Concatenate the scores from low and high dataframes
        bdt_score = np.concatenate((bdt_score_low, bdt_score_high), axis=0)
        
        # Release memory
        del model_low, model_high
        gc.collect()

        return bdt_score
    
    def evaluateDNN(self, data):
        
        #print("Evaluating DNN...")
        #print(self.params.Models.DNN[self.channel][self.events.metadata["year"]].model_file)
        #print()
        # Load the model on demand
        with tf.device('/CPU:0'):  # Use CPU to avoid GPU memory issues
            model = load_model(self.params.Models.DNN[self.channel][self.events.metadata["year"]].model_file)
            dnn_score = model.predict(data, batch_size=32).ravel()
        # Release memory
        tf.keras.backend.clear_session()
        del model
        gc.collect()
        #print("DNN evaluation completed.")
        return dnn_score

    
    def evaluateseparateDNNs(self, data):
        
        data_df_low = data[data['dilep_pt'] < 150]
        data_df_high = data[data['dilep_pt'] >= 150]
        
        # Initialize empty arrays for scores
        dnn_score_low = np.array([])
        dnn_score_high = np.array([])
        
        # Read the model file
        model_low = load_model(self.params.Models.DNN[f'{self.channel}_low'][self.events.metadata["year"]].model_file)
        model_high = load_model(self.params.Models.DNN[f'{self.channel}_high'][self.events.metadata["year"]].model_file)
        
        # Predict only if data_df_low is non-empty
        if not data_df_low.empty:
            print("Predicting for low dilep_pt...")
            with tf.device('/CPU:0'):  # Use CPU to avoid GPU memory issues
                model_low = load_model(self.params.DNN_low)
                dnn_score_low = model_low.predict(data_df_low, batch_size=32).ravel()
            tf.keras.backend.clear_session()
            del model_low
            gc.collect()
            print("Prediction for low dilep_pt completed.")
        
        # Predict only if data_df_high is non-empty
        if not data_df_high.empty:
            print("Predicting for high dilep_pt...")
            with tf.device('/CPU:0'):  # Use CPU to avoid GPU memory issues
                model_high = load_model(self.params.DNN_high)
                dnn_score_high = model_high.predict(data_df_high, batch_size=32).ravel()
            tf.keras.backend.clear_session()
            del model_high
            gc.collect()
            print("Prediction for high dilep_pt completed.")
        
        
        
        dnn_score = np.concatenate((dnn_score_low, dnn_score_high), axis=0)
        
        print("Separate DNN evaluation completed.")
        
        return dnn_score
    
    def resize_tensor(self,tensor,target):
        m, n, p = tensor.shape
        
        if n > target:
            return tensor[:, :target, :]
        elif n < target:
            padding = torch.zeros((m, target - n, p), device=tensor.device, dtype=tensor.dtype)
            return torch.cat((tensor, padding), dim=1)
        else:
            return tensor

    def evaluateGNN(self,data):
        # model = torch.jit.load(self.params.Models.GNN[self.channel][self.events.metadata["year"]].model_file) #TODO This would be the most elegant way, but the current model does not work with torch.jit

        modelparams = pickle.load(open(self.params.Models.GNN["Global"].params,'rb'))
        model = GraphAttentionClassifier(**modelparams)
        model.load_state_dict(torch.load(self.params.Models.GNN["Global"].model_file,weights_only=True,map_location=self.device))
        model.eval()

        eramap = {  "2022_preEE":   0,
                    "2022_postEE":  1,
                    "2023_preBPix": 2,
                    "2023_postBPix":3,
                }

        data["era"] = eramap[self._year]
        ln = len(data)

        if self.proc_type=="ZLL":
            data["V_pt"] = data["ll_pt"]
            data["V_eta"] = data["ll_eta"]
            data["V_phi"] = data["ll_phi"]
            data["V_mass"] = data["ll_mass"]
            data["W_m"] = ak.zeros_like(data["PuppiMET_pt"])
            data["channel"] = ak.ones_like(data["PuppiMET_pt"])*2

        elif self.proc_type=="WLNu":
            data["V_pt"] = data["W_pt"]
            data["V_eta"] = data["W_eta"]
            data["V_phi"] = data["W_phi"]
            data["V_mass"] = data["W_mt"]
            data["channel"] = ak.ones_like(data["PuppiMET_pt"])

        elif self.proc_type=="ZNuNu":
            data["V_pt"] = data["Z_pt"]
            data["V_eta"] = data["Z_eta"]
            data["V_phi"] = data["Z_phi"]
            data["V_mass"] = data["Z_m"]
            data["W_m"] = ak.zeros_like(data["PuppiMET_pt"])
            data["channel"] = ak.zeros_like(data["PuppiMET_pt"])
            data["LeptonCategory"] = ak.zeros_like(data["PuppiMET_pt"])

        optionaljagged = ['LeptonGood_miniPFRelIso_all',
            'LeptonGood_pfRelIso03_all', 'LeptonGood_pt',
            'LeptonGood_eta', 'LeptonGood_phi',
            'LeptonGood_mass'
        ]
        for opt in optionaljagged:
            if opt not in data.fields:
                data[opt] = np.zeros([ln, 1]).tolist()


        varslist =  ["JetGood_btagCvL","JetGood_btagCvB",
                    "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                    "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                    "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                    "V_pt","V_eta","V_phi","V_mass",
                    "PuppiMET_pt","PuppiMET_phi","nPV","W_m",
                    "LeptonCategory","channel","era"]

        batch_size = 1024*8
        all_predictions = []

        for start_idx in range(0, ln, batch_size):
            batch_data = data[start_idx:start_idx + batch_size]

            X = batch_data[varslist]
            tensordict, _ = process_gnn_inputs(X, removenans=False)

            with torch.no_grad():
                batch_prediction = model(
                    tensordict["jet"],
                    tensordict["jetP4"],
                    tensordict["lep"],
                    tensordict["lepP4"],
                    tensordict["VP4"],
                    tensordict["global"],
                    tensordict["category"]
                )[:, 0]

            all_predictions.append(batch_prediction.cpu())

            del tensordict
            gc.collect()

        all_predictions = torch.cat(all_predictions)
        all_predictions = torch.nan_to_num(all_predictions)

        return all_predictions.numpy()

        
            
    
    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_before_presel(self, variation):
        self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)

        jetvars = ["btagCvL","btagCvB"]
        leptonvars = ["miniPFRelIso_all","pfRelIso03_all"]
        p4vars = ["pt","eta","phi","mass"]

        for var in jetvars+p4vars:
            self.events["JetGood_"+var] = self.events.JetGood[var]
        for var in leptonvars+p4vars:
            self.events["LeptonGood_"+var] = self.events.LeptonGood[var]
        for var in p4vars:
            self.events["ll_"+var] = self.events.ll[var]

        self.events["PuppiMET_pt"] = self.events.PuppiMET.pt
        self.events["PuppiMET_phi"] = self.events.PuppiMET.phi
        self.events["nPV"] = self.events.PV.npvsGood



    def define_common_variables_after_presel(self, variation):
        
        self.events["dijet"] = get_dijet(self.events.JetGood)

        if self.newjetdefiniton:
            self.events["JetsCvsL"] = CvsLsorted(self.events["JetGood"])
            self.events["dijet_csort"] = get_dijet(self.events.JetsCvsL)
        else:
            #TODO: This is temporary
            self.events["JetsCvsL"] = CvsLsorted(self.events["JetGood"], tagger = self.myJetTagger)
            self.events["dijet_csort"] = get_dijet(self.events.JetsCvsL, tagger = self.myJetTagger)

        #self.events["dijet_pt"] = self.events.dijet.pt
        
        if self.proc_type=="ZLL":

            ### General
            self.events["dijet_m"] = self.events.dijet_csort.mass
            self.events["dijet_pt"] = self.events.dijet_csort.pt
            self.events["dijet_dr"] = self.events.dijet_csort.deltaR
            self.events["dijet_deltaPhi"] = self.events.dijet_csort.deltaPhi
            self.events["dijet_deltaEta"] = self.events.dijet_csort.deltaEta
            self.events["dijet_CvsL_max"] = self.events.dijet_csort.j1CvsL
            self.events["dijet_CvsL_min"] = self.events.dijet_csort.j2CvsL
            self.events["dijet_CvsB_max"] = self.events.dijet_csort.j1CvsB
            self.events["dijet_CvsB_min"] = self.events.dijet_csort.j2CvsB
            self.events["dijet_pt_max"] = self.events.dijet_csort.j1pt
            self.events["dijet_pt_min"] = self.events.dijet_csort.j2pt

            self.events["dilep_m"] = self.events.ll.mass
            self.events["dilep_pt"] = self.events.ll.pt
            self.events["dilep_dr"] = self.events.ll.deltaR
            self.events["dilep_deltaPhi"] = self.events.ll.deltaPhi
            self.events["dilep_deltaEta"] = self.events.ll.deltaEta

            self.events["LeptonCategory"] = ak.values_astype(self.events["nMuonGood"]==2,"int32")
            
            self.events["ZH_pt_ratio"] = self.events.dijet_csort.pt/self.events.ll.pt
            self.events["ZH_deltaPhi"] = np.abs(self.events.ll.delta_phi(self.events.dijet_csort))

            # why cant't we use delta_phi function here?
            self.angle21_gen = (abs(self.events.ll.l2phi - self.events.dijet_csort.j1Phi) < np.pi)
            self.angle22_gen = (abs(self.events.ll.l2phi - self.events.dijet_csort.j2Phi) < np.pi)
            self.events["deltaPhi_l2_j1"] = ak.where(self.angle21_gen, abs(self.events.ll.l2phi - self.events.dijet_csort.j1Phi), 2*np.pi - abs(self.events.ll.l2phi - self.events.dijet_csort.j1Phi))              
            self.events["deltaPhi_l2_j2"] = ak.where(self.angle22_gen, abs(self.events.ll.l2phi - self.events.dijet_csort.j2Phi), 2*np.pi - abs(self.events.ll.l2phi - self.events.dijet_csort.j2Phi))
            self.events["deltaPhi_l2_j1"] = np.abs(delta_phi(self.events.ll.l2phi, self.events.dijet_csort.j1Phi))
            
            #events = self.events[odd_event_mask]
            # Create a record of variables to be dumped as root/parquete file:

            variables_for_MVA_eval_list = ["dilep_m","dilep_pt","dilep_dr","dilep_deltaPhi","dilep_deltaEta",
                                    "dijet_m","dijet_pt","dijet_dr","dijet_deltaPhi","dijet_deltaEta",
                                    "dijet_CvsL_max","dijet_CvsL_min","dijet_CvsB_max","dijet_CvsB_min",
                                    "dijet_pt_max","dijet_pt_min",
                                    "ZH_pt_ratio","ZH_deltaPhi","deltaPhi_l2_j1","deltaPhi_l2_j2"]

            variables_for_MVA_eval = ak.zip({v:self.events[v] for v in variables_for_MVA_eval_list})  #TODO: use odd_events instead

            gnn_vars = ["JetGood_btagCvL","JetGood_btagCvB",
                        "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                        "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                        "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                        "ll_pt","ll_eta","ll_phi","ll_mass",
                        "PuppiMET_pt","PuppiMET_phi","nPV","LeptonCategory"]

            ak_gnn = self.events[gnn_vars] #TODO: use odd_events instead
            
            df = ak.to_pandas(variables_for_MVA_eval)
            columns_to_exclude = ['dilep_m']
            df = df.drop(columns=columns_to_exclude, errors='ignore')

            self.channel = "2L"
            if not self.params.separate_models: 
                df_final = df.reindex(range(len(self.events)), fill_value=np.nan)

                bdt_predictions = self.evaluateBDT(df_final)
                bdt_predictions = np.where(df_final.isnull().any(axis=1), np.nan, bdt_predictions)
                # Convert NaN to None
                bdt_predictions = [None if np.isnan(x) else x for x in bdt_predictions]
                self.events["BDT"] = bdt_predictions
                
                if self.run_dnn:
                    self.events["DNN"] = self.evaluateDNN(df_final)
                else:
                    self.events["DNN"] = np.zeros_like(self.events["BDT"])

                if self.run_gnn:
                    self.events["GNN"] = self.evaluateGNN(ak_gnn)
                    self.events["GNN_transformed"] = np.power(self.events["GNN"],9)
                else:
                    self.events["GNN"] = np.zeros_like(self.events["BDT"])
                    self.events["GNN_transformed"] = np.zeros_like(self.events["BDT"])
            else:
                df_final = df.reindex(range(len(self.events)), fill_value=np.nan)

                bdt_predictions = self.evaluateseparateBDTs(df_final)
                bdt_predictions = np.where(df_final.isnull().any(axis=1), np.nan, bdt_predictions)
                # Convert NaN to None
                bdt_predictions = [None if np.isnan(x) else x for x in bdt_predictions]
                self.events["BDT"] = bdt_predictions
                
                if self.run_dnn:
                    self.events["DNN"] = self.evaluateseparateDNNs(df_final)
                else:
                    self.events["DNN"] = np.zeros_like(self.events["BDT"])
                
        if self.proc_type=="WLNu":
            self.events["MET_used"] = ak.zip({
                                        "pt": self.events.PuppiMET.pt,
                                        "eta": ak.zeros_like(self.events.PuppiMET.pt),
                                        "phi": self.events.PuppiMET.phi,
                                        "mass": ak.zeros_like(self.events.PuppiMET.pt),
                                        "charge": ak.zeros_like(self.events.PuppiMET.pt),
                                        },with_name="PtEtaPhiMCandidate")
            self.events["lead_lep"] = ak.firsts(self.events.LeptonGood)
            self.events["W_candidate"] = self.events.lead_lep + self.events.MET_used
            #print("W_candidate", self.events.W_candidate, self.events.W_candidate.mass, self.events.W_candidate.pt)
            self.events["W_m"] = self.events.W_candidate.mass
            self.events["W_pt"] = self.events.W_candidate.pt
            self.events["W_eta"] = ak.zeros_like(self.events.W_candidate.pt)
            self.events["W_phi"] = self.events.MET_used.phi
            self.events["W_mt"] = np.sqrt(2*self.events.lead_lep.pt*self.events.MET_used.pt*(1-np.cos(self.events.lead_lep.delta_phi(self.events.MET_used))))
            self.events["pt_miss"] = self.events.MET_used.pt
            self.events["LeptonCategory"] = ak.values_astype(self.events["nMuonGood"]==1,"int32")

            # Step 1: Calculate delta_r for each b_jet with respect to lead_lep
            delta_rs = self.events.BJetGood.delta_r(self.events.lead_lep)

            # Step 2: Find the index of the b_jet with the minimum delta_r
            min_delta_r_index = ak.argmin(delta_rs, axis=1, keepdims=True)

            # Step 3: Select the b_jet with the minimum delta_r
            self.events["b_jet"] = self.events.BJetGood[min_delta_r_index]
            
            # Create a mask to ensure at least one b_jet is present
            bjet_mask = ak.num(self.events.BJetGood) > 0
            ### FIXME: Check if we need to mask events with no b-jets
            
            self.events["dijet_m"] = self.events.dijet_csort.mass
            self.events["dijet_pt"] = self.events.dijet_csort.pt
            self.events["dijet_dr"] = self.events.dijet_csort.deltaR
            self.events["dijet_deltaPhi"] = self.events.dijet_csort.deltaPhi
            self.events["dijet_deltaEta"] = self.events.dijet_csort.deltaEta
            self.events["dijet_CvsL_max"] = self.events.dijet_csort.j1CvsL
            self.events["dijet_CvsL_min"] = self.events.dijet_csort.j2CvsL
            self.events["dijet_CvsB_max"] = self.events.dijet_csort.j1CvsB
            self.events["dijet_CvsB_min"] = self.events.dijet_csort.j2CvsB
            self.events["dijet_pt_max"] = self.events.dijet_csort.j1pt
            self.events["dijet_pt_min"] = self.events.dijet_csort.j2pt
            
            self.events["deltaPhi_jet1_MET"] = np.abs(self.events.PuppiMET.delta_phi(self.events.JetGood[:,0]))
            self.events["deltaPhi_jet2_MET"] = np.abs(self.events.PuppiMET.delta_phi(self.events.JetGood[:,1]))
        
            self.events["WH_deltaPhi"] = np.abs(self.events.W_candidate.delta_phi(self.events.dijet_csort))
            self.events["deltaPhi_l1_j1"] = np.abs(delta_phi(self.events.lead_lep.phi, self.events.dijet_csort.j1Phi))
            self.events["deltaPhi_l1_MET"] = np.abs(delta_phi(self.events.lead_lep.phi, self.events.MET_used.phi))
            self.events["deltaPhi_l1_b"] = np.abs(delta_phi(self.events.lead_lep.phi, self.events.b_jet.phi))
            self.events["deltaEta_l1_b"] = np.abs(self.events.lead_lep.eta - self.events.b_jet.eta)
            self.events["deltaR_l1_b"] = np.sqrt((self.events.lead_lep.eta - self.events.b_jet.eta)**2 + (self.events.lead_lep.phi - self.events.b_jet.phi)**2)
            self.events["b_CvsL"] = self.events.b_jet["btagCvL"]
            self.events["b_CvsB"] = self.events.b_jet["btagCvB"]
            self.events["b_Btag"] = self.events.b_jet["btagB"]
            
            self.events["neutrino_from_W"] = get_nu_4momentum(self.events.lead_lep, self.events.MET_used)
            self.events["top_candidate"] = self.events.lead_lep + self.events.b_jet + self.events.neutrino_from_W
            #print("top_candidate", self.events.top_candidate, self.events.top_candidate.mass, self.events.top_candidate.pt)

            self.events["top_mass"] = (self.events.lead_lep + self.events.b_jet + self.events.neutrino_from_W).mass
            #events = self.events[event_mask & bjet_mask]
            #self.events = self.events[bjet_mask]

            variables_for_MVA_eval_list = ["dijet_m","dijet_pt","dijet_dr","dijet_deltaPhi","dijet_deltaEta",
                                    "dijet_CvsL_max","dijet_CvsL_min","dijet_CvsB_max","dijet_CvsB_min",
                                    "dijet_pt_max","dijet_pt_min",
                                    "W_mt","W_pt","pt_miss","WH_deltaPhi",
                                    "deltaPhi_l1_j1","deltaPhi_l1_MET","deltaPhi_l1_b","deltaEta_l1_b","deltaR_l1_b",
                                    "b_CvsL","b_CvsB","b_Btag","top_mass"]

            variables_for_MVA_eval = ak.zip({v:self.events[v] for v in variables_for_MVA_eval_list})

            gnn_vars = ["JetGood_btagCvL","JetGood_btagCvB",
                        "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                        "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                        "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                        "W_pt","W_eta","W_phi","W_mt",
                        "PuppiMET_pt","PuppiMET_phi","nPV","W_m","LeptonCategory"]

            ak_gnn = self.events[gnn_vars]

            df = ak.to_pandas(variables_for_MVA_eval)

            # Remove the 'subentry' column
            df = df.reset_index(level='subentry', drop=True)
            
            # Ensure the DataFrame has a simple index
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(drop=True)
            #columns_to_exclude = []
            #df = df.drop(columns=columns_to_exclude, errors='ignore')
            self.channel = "1L"
            df_final = df.reindex(range(len(self.events)), fill_value=np.nan)

            bdt_predictions = self.evaluateBDT(df_final)
            bdt_predictions = np.where(df_final.isnull().any(axis=1), np.nan, bdt_predictions)
            # Convert NaN to None
            bdt_predictions = [None if np.isnan(x) else x for x in bdt_predictions]
            self.events["BDT"] = bdt_predictions
            
            if self.run_dnn:
                self.events["DNN"] = self.evaluateDNN(df_final)
            else:
                self.events["DNN"] = np.zeros_like(self.events["BDT"])

            if self.run_gnn:
                self.events["GNN"] = self.evaluateGNN(ak_gnn)
                self.events["GNN_transformed"] = np.power(self.events["GNN"],9)
            else:
                self.events["GNN"] = np.zeros_like(self.events["BDT"])
                self.events["GNN_transformed"] = np.zeros_like(self.events["BDT"])


        if self.proc_type=="ZNuNu":
            ### General
            self.events["MET_used"] = ak.zip({
                                        "pt": self.events.PuppiMET.pt,
                                        "eta": ak.zeros_like(self.events.PuppiMET.pt),
                                        "phi": self.events.PuppiMET.phi,
                                        "mass": ak.zeros_like(self.events.PuppiMET.pt),
                                        "charge": ak.zeros_like(self.events.PuppiMET.pt),
                                        },with_name="PtEtaPhiMCandidate")
            self.events["Z_candidate"] = self.events.MET_used
            self.events["Z_pt"] = self.events.Z_candidate.pt
            self.events["Z_eta"] = ak.zeros_like(self.events.Z_candidate.pt)
            self.events["Z_phi"] = self.events.Z_candidate.phi
            self.events["Z_m"] = ak.ones_like(self.events.Z_candidate.pt)*91.1876
            
            self.events["dijet_m"] = self.events.dijet_csort.mass
            self.events["dijet_pt"] = self.events.dijet_csort.pt
            self.events["dijet_dr"] = self.events.dijet_csort.deltaR
            self.events["dijet_deltaPhi"] = self.events.dijet_csort.deltaPhi
            self.events["dijet_deltaEta"] = self.events.dijet_csort.deltaEta
            self.events["dijet_CvsL_max"] = self.events.dijet_csort.j1CvsL
            self.events["dijet_CvsL_min"] = self.events.dijet_csort.j2CvsL
            self.events["dijet_CvsB_max"] = self.events.dijet_csort.j1CvsB
            self.events["dijet_CvsB_min"] = self.events.dijet_csort.j2CvsB
            self.events["dijet_pt_max"] = self.events.dijet_csort.j1pt
            self.events["dijet_pt_min"] = self.events.dijet_csort.j2pt
            
            self.events["ZH_pt_ratio"] = self.events.dijet_csort.pt/self.events.Z_candidate.pt
            self.events["ZH_deltaPhi"] = np.abs(self.events.Z_candidate.delta_phi(self.events.dijet_csort))
            self.events["deltaPhi_jet1_MET"] = np.abs(self.events.PuppiMET.delta_phi(self.events.JetGood[:,0]))
            self.events["deltaPhi_jet2_MET"] = np.abs(self.events.PuppiMET.delta_phi(self.events.JetGood[:,1]))

            # odd_events = self.events[odd_event_mask]

            variables_for_MVA_eval_list = ["dijet_m","dijet_pt","dijet_dr","dijet_deltaPhi","dijet_deltaEta",
                                    "dijet_CvsL_max","dijet_CvsL_min","dijet_CvsB_max","dijet_CvsB_min",
                                    "dijet_pt_max","dijet_pt_min",
                                    "ZH_pt_ratio","ZH_deltaPhi","Z_pt"]

            variables_for_MVA_eval = ak.zip({v:self.events[v] for v in variables_for_MVA_eval_list})

            gnn_vars = ["JetGood_btagCvL","JetGood_btagCvB",
                        "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                        "Z_pt","Z_eta","Z_phi","Z_m",
                        "PuppiMET_pt","PuppiMET_phi","nPV"]

            ak_gnn = self.events[gnn_vars]
            
            df = ak.to_pandas(variables_for_MVA_eval)
            #columns_to_exclude = []
            #df = df.drop(columns=columns_to_exclude, errors='ignore')
            self.channel = "0L"
            df_final = df.reindex(range(len(self.events)), fill_value=np.nan)

            bdt_predictions = self.evaluateBDT(df_final)
            bdt_predictions = np.where(df_final.isnull().any(axis=1), np.nan, bdt_predictions)
            # Convert NaN to None
            bdt_predictions = [None if np.isnan(x) else x for x in bdt_predictions]
            self.events["BDT"] = bdt_predictions
            if self.run_dnn:
                self.events["DNN"] = self.evaluateDNN(df_final)
            else:
                self.events["DNN"] = np.zeros_like(self.events["BDT"])
            if self.run_gnn:
                self.events["GNN"] = self.evaluateGNN(ak_gnn)
                self.events["GNN_transformed"] = np.power(self.events["GNN"],9)
            else:
                self.events["GNN"] = np.zeros_like(self.events["BDT"])
                self.events["GNN_transformed"] = np.zeros_like(self.events["BDT"])
            
        
