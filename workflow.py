import awkward as ak
import numpy as np
import uproot
import pandas as pd
import warnings
import os
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
import CommonSelectors
from CommonSelectors import *


from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.hist_manager import Axis
from pocket_coffea.lib.objects import (
    jet_correction,
    lepton_selection,
    jet_selection,
    btagging,
    CvsLsorted,
    get_dilepton,
    get_dijet
)

def delta_phi(a, b):
    """Compute difference in angle two phi values
    Returns a value within [-pi, pi)
    """
    return (a - b + np.pi) % (2 * np.pi) - np.pi

class VHccBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        #print("Something")

        self.proc_type   = self.params["proc_type"]
        self.save_arrays = self.params["save_arrays"]

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


        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params, "LeptonGood"
        )
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
        
        # Read the model file
        model = lgb.Booster(model_file=self.params.LightGBM_model)
        bdt_score = model.predict(data)

        return bdt_score
    
    def evaluateseparateBDTs(self, data):
        data_df_low = data[data['dilep_pt'] < 150]
        data_df_high = data[data['dilep_pt'] >= 150]
        
        # Initialize empty arrays for scores
        bdt_score_low = np.array([])
        bdt_score_high = np.array([])
        
        # Read the model files
        model_low = lgb.Booster(model_file=self.params.LigtGBM_low)
        model_high = lgb.Booster(model_file=self.params.LigtGBM_high)
        
        # Predict only if data_df_low is non-empty
        if not data_df_low.empty:
            bdt_score_low = model_low.predict(data_df_low)
        
        # Predict only if data_df_high is non-empty
        if not data_df_high.empty:
            bdt_score_high = model_high.predict(data_df_high)
        
        # Concatenate the scores from low and high dataframes
        bdt_score = np.concatenate((bdt_score_low, bdt_score_high), axis=0)

        return bdt_score
    
    def evaluateDNN(self, data):
        
        # Read the model file
        model = model = load_model(self.params.DNN_model)
        dnn_score =  model.predict(data).ravel()


        return dnn_score
    
    def evaluateseparateDNNs(self, data):
        
        data_df_low = data[data['dilep_pt'] < 150]
        data_df_high = data[data['dilep_pt'] >= 150]
        
        # Initialize empty arrays for scores
        dnn_score_low = np.array([])
        dnn_score_high = np.array([])
        
        # Read the model file
        model_low = load_model(self.params.DNN_low)
        model_high = load_model(self.params.DNN_high)
        
        # Predict only if data_df_low is non-empty
        if not data_df_low.empty:
            dnn_score_low = model_low.predict(data_df_low).ravel()
        
        # Predict only if data_df_high is non-empty
        if not data_df_high.empty:
            dnn_score_high = model_high.predict(data_df_high).ravel()
        
        
        dnn_score = np.concatenate((dnn_score_low, dnn_score_high), axis=0)

        return dnn_score
    

    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_before_presel(self, variation):
        self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)

    def define_common_variables_after_presel(self, variation):
        self.events["dijet"] = get_dijet(self.events.JetGood)

        
        self.events["JetsCvsL"] = CvsLsorted(self.events["JetGood"], self.params.ctagging.working_point[self._year])
        
        self.events["dijet_csort"] = get_dijet(self.events.JetsCvsL, tagger = True)
                
        #self.events["dijet_pt"] = self.events.dijet.pt


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


        if self.proc_type=="ZLL":


            self.events["dilep_m"] = self.events.ll.mass
            self.events["dilep_pt"] = self.events.ll.pt
            self.events["dilep_dr"] = self.events.ll.deltaR
            self.events["dilep_deltaPhi"] = self.events.ll.deltaPhi
            self.events["dilep_deltaEta"] = self.events.ll.deltaEta

            
            self.events["ZH_pt_ratio"] = self.events.dijet_csort.pt/self.events.ll.pt
            self.events["ZH_deltaPhi"] = np.abs(self.events.ll.delta_phi(self.events.dijet_csort))

            # why cant't we use delta_phi function here?
            self.angle21_gen = (abs(self.events.ll.l2phi - self.events.dijet_csort.j1Phi) < np.pi)
            self.angle22_gen = (abs(self.events.ll.l2phi - self.events.dijet_csort.j2Phi) < np.pi)
            self.events["deltaPhi_l2_j1"] = ak.where(self.angle21_gen, abs(self.events.ll.l2phi - self.events.dijet_csort.j1Phi), 2*np.pi - abs(self.events.ll.l2phi - self.events.dijet_csort.j1Phi))              
            self.events["deltaPhi_l2_j2"] = ak.where(self.angle22_gen, abs(self.events.ll.l2phi - self.events.dijet_csort.j2Phi), 2*np.pi - abs(self.events.ll.l2phi - self.events.dijet_csort.j2Phi))
            self.events["deltaPhi_l2_j1"] = np.abs(delta_phi(self.events.ll.l2phi, self.events.dijet_csort.j1Phi))

            
                
            
            # Create a record of variables to be dumped as root/parquete file:
            variables_to_process = ak.zip({
                "dilep_m": self.events["dilep_m"],
                "dilep_pt": self.events["dilep_pt"],
                "dilep_dr": self.events["dilep_dr"],
                "dilep_deltaPhi": self.events["dilep_deltaPhi"],
                "dilep_deltaEta": self.events["dilep_deltaEta"],
                
                "dijet_m": self.events["dijet_m"],
                "dijet_pt": self.events["dijet_pt"],
                "dijet_dr": self.events["dijet_dr"],
                "dijet_deltaPhi": self.events["dijet_deltaPhi"],
                "dijet_deltaEta": self.events["dijet_deltaEta"],
                "dijet_CvsL_max": self.events["dijet_CvsL_max"],
                "dijet_CvsL_min": self.events["dijet_CvsL_min"],
                "dijet_CvsB_max": self.events["dijet_CvsB_max"],
                "dijet_CvsB_min": self.events["dijet_CvsB_min"],
                "dijet_pt_max": self.events["dijet_pt_max"],
                "dijet_pt_min": self.events["dijet_pt_min"],
                
                "ZH_pt_ratio": self.events["ZH_pt_ratio"],
                "ZH_deltaPhi": self.events["ZH_deltaPhi"],
                "deltaPhi_l2_j1": self.events["deltaPhi_l2_j1"],
                "deltaPhi_l2_j2": self.events["deltaPhi_l2_j2"],
            })
            
            df = ak.to_pandas(variables_to_process)
            columns_to_exclude = ['dilep_m']
            df = df.drop(columns=columns_to_exclude, errors='ignore')
            if not self.params.separate_models: 
                self.events["BDT"] = self.evaluateBDT(df)
                self.events["DNN"] = self.evaluateDNN(df)
            else:
                self.events["BDT"] = self.evaluateseparateBDTs(df)
                self.events["DNN"] = self.evaluateseparateDNNs(df)

                
        if self.proc_type=="ZNuNu":
            self.events["deltaPhi_jet1_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,0]))
            self.events["deltaPhi_jet2_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,1]))
        

        #print("Pt sort pt:", self.events["JetGood"][self.events["nJetGood"]>=3].pt)
        #print("CvsL sort pt:", self.events["JetsCvsL"][self.events["nJetGood"]>=3].pt)

        #print("Pt sort CvsL:", self.events["JetGood"][self.events["nJetGood"]>=3].btagDeepFlavCvL)
        #print("CvsL sort CvsL:", self.events["JetsCvsL"][self.events["nJetGood"]>=3].btagDeepFlavCvL)

        elif self.proc_type=="WLNu":
            "TODO. Create arrays for WLNu channel"
            pass
        elif self.proc_type=="ZNuNu":
            "TODO. Create arrays for ZNuNu channel"
            pass
                    

        if self.save_arrays:
            mask = ((self.events.nJetGood >= 2) & (self.events.nLeptonGood>=2)) & (self.events.ll.pt > 60) & (self.events.ll.mass > 75) & (self.events.ll.mass < 115) & ((self.events.nJetGood >= 2) & (self.events.dijet_csort.mass > 75) & (self.events.dijet_csort.mass < 200)) & ((self.events.JetsCvsL.btagDeepFlavCvL[:,0]>0.2) & (self.events.JetsCvsL.btagDeepFlavCvB[:,0]>0.4))
            selection_ZLL = ak.where(ak.is_none(mask), False, mask)
                    
            SR_Data = self.events[selection_ZLL]
            variables_to_save = ak.zip({
                "dilep_m": SR_Data["dilep_m"],
                "dilep_pt": SR_Data["dilep_pt"],
                "dilep_dr": SR_Data["dilep_dr"],
                "dilep_deltaPhi": SR_Data["dilep_deltaPhi"],
                "dilep_deltaEta": SR_Data["dilep_deltaEta"],
                
                "dijet_m": SR_Data["dijet_m"],
                "dijet_pt": SR_Data["dijet_pt"],
                "dijet_dr": SR_Data["dijet_dr"],
                "dijet_deltaPhi": SR_Data["dijet_deltaPhi"],
                "dijet_deltaEta": SR_Data["dijet_deltaEta"],
                "dijet_CvsL_max": SR_Data["dijet_CvsL_max"],
                "dijet_CvsL_min": SR_Data["dijet_CvsL_min"],
                "dijet_CvsB_max": SR_Data["dijet_CvsB_max"],
                "dijet_CvsB_min": SR_Data["dijet_CvsB_min"],
                "dijet_pt_max": SR_Data["dijet_pt_max"],
                "dijet_pt_min": SR_Data["dijet_pt_min"],
                
                "ZH_pt_ratio": SR_Data["ZH_pt_ratio"],
                "ZH_deltaPhi": SR_Data["ZH_deltaPhi"],
                "deltaPhi_l2_j1": SR_Data["deltaPhi_l2_j1"],
                "deltaPhi_l2_j2": SR_Data["deltaPhi_l2_j2"],
            })
            # Here we write to root  and parquete files

            with warnings.catch_warnings():
                # Suppress FutureWarning
                warnings.filterwarnings("ignore", category=FutureWarning)

                # Check if the directory exists
                if not os.path.exists(f"Saved_root_files/{self.events.metadata['dataset']}"):
                    # If not, create it
                    os.system(f"mkdir -p Saved_root_files/{self.events.metadata['dataset']}")

                # Write the events to a ROOT file

                #with uproot.recreate(f"Saved_root_files/{self.events.metadata['dataset']}/{self.events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(self.events.metadata['entrystart'])}_{int(self.events.metadata['entrystop'])}.root") as f: 
                #    f["variables"] = ak.to_pandas(variables_to_save)


                # Write the events to a Parquet file
                ak.to_pandas(variables_to_save).to_parquet(f"Saved_root_files/{self.events.metadata['dataset']}/{self.events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(self.events.metadata['entrystart'])}_{int(self.events.metadata['entrystop'])}_vars.parquet")
