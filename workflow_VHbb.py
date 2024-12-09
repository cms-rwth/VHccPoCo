import awkward as ak
import numpy as np
import ctypes #test
import uproot
import pandas as pd
import math
import warnings
import os, pickle
import lightgbm as lgb
# import tensorflow as tf
import gc
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.callbacks import EarlyStopping
# from keras.models import load_model

# import inspect

# import torch
# from MVA.gnnmodels import GraphAttentionClassifier

import CommonSelectors
from CommonSelectors import *
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

def get_nu_4momentum(Lepton, MET):
    mW = 80.38

    # Convert pt, eta, phi, m to px, py, pz, E
    px = Lepton.pt * np.cos(Lepton.phi)
    py = Lepton.pt * np.sin(Lepton.phi)
    pz = Lepton.pt * np.sinh(Lepton.eta)
    E = np.sqrt(Lepton.mass**2 + Lepton.pt**2 * np.cosh(Lepton.eta)**2)
    
    MET_px = MET.pt * np.cos(MET.phi)
    MET_py = MET.pt * np.sin(MET.phi)
    
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

def BvsLsorted(jets, tagger):
    if tagger == "PNet":
        btag = "btagPNetB"
    elif tagger == "DeepFlav":
        btag = "btagDeepFlavB"
    elif tagger == "RobustParT":
        btag = "btagRobustParTAK4B"
    else:
        raise NotImplementedError(f"This tagger is not implemented: {tagger}")
    
    return jets[ak.argsort(jets[btag], axis=1, ascending=False)]
    
def get_dibjet(jets, tagger = 'PNet'):
    
    fields = {
        "pt": 0.,
        "eta": 0.,
        "phi": 0.,
        "mass": 0.,
    }

    jets = ak.pad_none(jets, 2)
    njet = ak.num(jets[~ak.is_none(jets, axis=1)])
    
    dijet = jets[:, 0] + jets[:, 1]

    for var in fields.keys():
        fields[var] = ak.where(
            (njet >= 2),
            getattr(dijet, var),
            fields[var]
        )

    fields["deltaR"] = ak.where( (njet >= 2), jets[:,0].delta_r(jets[:,1]), -1)
    fields["deltaPhi"] = ak.where( (njet >= 2), abs(jets[:,0].delta_phi(jets[:,1])), -1)
    fields["deltaEta"] = ak.where( (njet >= 2), abs(jets[:,0].eta - jets[:,1].eta), -1)
    fields["j1Phi"] = ak.where( (njet >= 2), jets[:,0].phi, -1)
    fields["j2Phi"] = ak.where( (njet >= 2), jets[:,1].phi, -1)
    fields["j1pt"] = ak.where( (njet >= 2), jets[:,0].pt, -1)
    fields["j2pt"] = ak.where( (njet >= 2), jets[:,1].pt, -1)
    fields["j1mass"] = ak.where( (njet >= 2), jets[:,0].mass, -1)
    fields["j2mass"] = ak.where( (njet >= 2), jets[:,1].mass, -1)
    
    if tagger == "PNet":
        BvL = "btagPNetB"
        CvL = "btagPNetCvL"
        CvB = "btagPNetCvB"
    elif tagger == "DeepFlav":
        BvL = "btagDeepFlavB"
        CvL = "btagDeepFlavCvL"
        CvB = "btagDeepFlavCvB"
    elif tagger == "RobustParT":
        BvL = "btagRobustParTAK4B"
        CvL = "btagRobustParTAK4CvL"
        CvB = "btagRobustParTAK4CvB"
    else:
        raise NotImplementedError(f"This tagger is not implemented: {tagger}")

    if tagger:
        fields["j1BvsL"] = ak.where( (njet >= 2), jets[:,0][BvL], -1)
        fields["j2BvsL"] = ak.where( (njet >= 2), jets[:,1][BvL], -1)
        fields["j1CvsL"] = ak.where( (njet >= 2), jets[:,0][CvL], -1)
        fields["j2CvsL"] = ak.where( (njet >= 2), jets[:,1][CvL], -1)
        fields["j1CvsB"] = ak.where( (njet >= 2), jets[:,0][CvB], -1)
        fields["j2CvsB"] = ak.where( (njet >= 2), jets[:,1][CvB], -1)
        
    # Lead b-jet pt: larger of the first two jets' pt
    fields["leadb_pt"] = ak.where( njet >= 2, ak.max(ak.Array([jets[:, 0].pt, jets[:, 1].pt]), axis=0), -1)

    # Sublead b-jet pt: smaller of the first two jets' pt
    fields["subleadb_pt"] = ak.where(njet >= 2, ak.min(ak.Array([jets[:, 0].pt, jets[:, 1].pt]), axis=0), -1)
    
    dibjet = ak.zip(fields, with_name="PtEtaPhiMCandidate")
    return dibjet
  
def get_additionalleptons(electrons, muons, baseNum=0):

    if muons is None and electrons is None:
        raise("Must specify either muon or electron collection in get_dilepton() function")
    elif muons is None and electrons is not None:
        leptons = ak.pad_none(ak.with_name(electrons, "PtEtaPhiMCandidate"), baseNum)
    elif electrons is None and muons is not None:
        leptons = ak.pad_none(ak.with_name(muons, "PtEtaPhiMCandidate"), baseNum)
    else:
        leptons = ak.pad_none(ak.with_name(ak.concatenate([ muons[:, 0:baseNum], electrons[:, 0:baseNum]], axis=1), "PtEtaPhiMCandidate"), baseNum)
        
    nlep = ak.num(leptons[~ak.is_none(leptons, axis=1)])
    NAL = nlep - baseNum
    return NAL

def LorentzBooster(array_pt, array_eta, array_phi, array_mass):
    
    # Flatten arrays
    flat_pt = ak.to_numpy(ak.flatten(array_pt))
    flat_eta = ak.to_numpy(ak.flatten(array_eta))
    flat_phi = ak.to_numpy(ak.flatten(array_phi))
    flat_mass = ak.to_numpy(ak.flatten(array_mass))
    
    compute_lib = ctypes.CDLL('/afs/cern.ch/work/l/lichengz/private/VHbb/VHccPoCo/scripts/interface/compute_vectors.so')
    # TODO can add it into another parameter yml file
    
    # Define the argument types for the C++ function
    compute_lib.compute_4vectors.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # pt
        ctypes.POINTER(ctypes.c_double),  # eta
        ctypes.POINTER(ctypes.c_double),  # phi
        ctypes.POINTER(ctypes.c_double),  # mass
        ctypes.POINTER(ctypes.c_double),  # px
        ctypes.POINTER(ctypes.c_double),  # py
        ctypes.POINTER(ctypes.c_double),  # pz
        ctypes.POINTER(ctypes.c_double),  # energy
        ctypes.c_int                      # size
    ]
    
    # Allocate arrays for the outputs
    size = len(flat_pt)
    px = np.zeros(size, dtype=np.float64)
    py = np.zeros(size, dtype=np.float64)
    pz = np.zeros(size, dtype=np.float64)
    energy = np.zeros(size, dtype=np.float64)
    
    # Call the C++ function
    compute_lib.compute_4vectors(
        flat_pt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        flat_eta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        flat_phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        flat_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        px.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        py.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        energy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        size
    )
    
    # Print the results
    print("px:", px)
    print("py:", py)
    print("pz:", pz)
    print("energy:", energy)
    
    return 0
  
class VHbbBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        
        self.proc_type       = self.params["proc_type"]
        self.run_bdt         = self.params["run_bdt"]
        self.run_dnn         = self.params["run_dnn"]
        self.separate_models = self.params["separate_models"]

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
        
        self.events["JetGood"], self.jetGoodMask = jet_selection(
            self.events, "Jet", self.params, self._year, "LeptonGood"
        )
        
        self.events['EventNr'] = self.events.event
        
        self.events["BJetGood"] = btagging(
            self.events["JetGood"], self.params.btagging.working_point[self._year], wp=self.params.object_preselection.bJetWP)
            
    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nElectronGood"] = ak.num(self.events.ElectronGood)
        self.events["nLeptonGood"] = ak.num(self.events.LeptonGood)

        self.events["nJet"] = ak.num(self.events.Jet)
        self.events["nJetGood"] = ak.num(self.events.JetGood)
        self.events["NaJ"] = self.events.nJetGood - 2  # additional jets
        self.events["nBJetGood"] = ak.num(self.events.BJetGood)
        # self.events["nfatjet"]   = ak.num(self.events.FatJetGood)


    def evaluateBDT(self, data):
        print(self.params.Models_Hbb.BDT[self.channel][self.events.metadata["year"]].model_file)
        print()
        # Read the model file
        model = lgb.Booster(model_file=self.params.Models_Hbb.BDT[self.channel][self.events.metadata["year"]].model_file)
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
        
    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_before_presel(self, variation):
                
        self.events["dijet"] = get_dijet(self.events.JetGood)
        self.events["JetsCvsL"] = CvsLsorted(self.events["JetGood"],
                                             tagger = self.params.object_preselection.bJet_algorithm)
        self.events["dijet_csort"] = get_dibjet(self.events.JetsCvsL, 
                                                tagger = self.params.object_preselection.bJet_algorithm)
        
        self.events["JetsBvsL"] = BvsLsorted(self.events["JetGood"], self.params.object_preselection.bJet_algorithm)
        self.events["dijet_bsort"] = get_dibjet(self.events.JetsBvsL, self.params.object_preselection.bJet_algorithm)
        
        self.events["MET_used"] = ak.zip({
                                        "pt": self.events.MET.pt,
                                        "eta": ak.zeros_like(self.events.MET.pt),
                                        "phi": self.events.MET.phi,
                                        "mass": ak.zeros_like(self.events.MET.pt),
                                        "charge": ak.zeros_like(self.events.MET.pt),
                                        },with_name="PtEtaPhiMCandidate")
        
        self.events["pt_miss"] = self.events.MET_used.pt
                
        self.events["btag_cut_L"] = self.params.btagger[self._year][self.params.object_preselection.bJet_algorithm].WP.L
        self.events["btag_cut_M"] = self.params.btagger[self._year][self.params.object_preselection.bJet_algorithm].WP.M
        self.events["btag_cut_T"] = self.params.btagger[self._year][self.params.object_preselection.bJet_algorithm].WP.T
        self.events["btag_cut_XT"] = self.params.btagger[self._year][self.params.object_preselection.bJet_algorithm].WP.XT
        self.events["btag_cut_XXT"] = self.params.btagger[self._year][self.params.object_preselection.bJet_algorithm].WP.XXT

    def define_common_variables_after_presel(self, variation):
      
        # odd_event_mask = (self.events.EventNr % 2 == 1)
        all_event_mask = (self.events.EventNr % 1 == 0)
        
        if self._isMC:
            self.events["nGenPart"] = ak.num(self.events.GenPart)
            self.events["GenPart_eta"] = self.events.GenPart.eta
            self.events["GenPart_genPartIdxMother"] = self.events.GenPart.genPartIdxMother
            self.events["GenPart_mass"] = self.events.GenPart.mass
            self.events["GenPart_pdgId"] = self.events.GenPart.pdgId
            self.events["GenPart_phi"] = self.events.GenPart.phi
            self.events["GenPart_pt"] = self.events.GenPart.pt
            self.events["GenPart_status"] = self.events.GenPart.status
            self.events["GenPart_statusFlags"] = self.events.GenPart.statusFlags
            
            # a C++ interface test
            # LorentzBooster(self.events.GenPart_pt, 
            #                self.events.GenPart_eta, 
            #                self.events.GenPart_phi, 
            #                self.events.GenPart_mass)

#             self.events["LHE_AlphaS"] = self.events.LHE.AlphaS
#             self.events["LHE_HT"] = self.events.LHE.HT
#             self.events["LHE_HTIncoming"] = self.events.LHE.HTIncoming
#             self.events["LHE_Nb"] = self.events.LHE.Nb
#             self.events["LHE_Nc"] = self.events.LHE.Nc
#             self.events["LHE_Nglu"] = self.events.LHE.Nglu
#             self.events["LHE_Njets"] = self.events.LHE.Njets
#             self.events["LHE_NpLO"] = self.events.LHE.NpLO
#             self.events["LHE_NpNLO"] = self.events.LHE.NpNLO
#             self.events["LHE_Nuds"] = self.events.LHE.Nuds
#             self.events["LHE_Vpt"] = self.events.LHE.Vpt

#             self.events["LHEPart_eta"] = self.events.LHEPart.eta
#             self.events["LHEPart_incomingpz"] = self.events.LHEPart.incomingpz
#             self.events["LHEPart_mass"] = self.events.LHEPart.mass
#             self.events["LHEPart_pdgId"] = self.events.LHEPart.pdgId
#             self.events["LHEPart_phi"] = self.events.LHEPart.phi
#             self.events["LHEPart_pt"] = self.events.LHEPart.pt
#             self.events["LHEPart_spin"] = self.events.LHEPart.spin
#             self.events["LHEPart_status"] = self.events.LHEPart.status
#             self.events["nLHEPart"] = ak.num(self.events.LHEPart)
                
        if self.proc_type=="ZLL":

            ### General
            self.events["NaL"] = get_additionalleptons(
                self.events.ElectronGood, self.events.MuonGood, 2
            ) # number of additional leptons
            
            self.events["dijet_m"] = self.events.dijet_csort.mass
            self.events["dijet_pt"] = self.events.dijet_csort.pt
            self.events["dijet_dr"] = self.events.dijet_csort.deltaR
            self.events["dijet_deltaPhi"] = self.events.dijet_csort.deltaPhi
            self.events["dijet_deltaEta"] = self.events.dijet_csort.deltaEta
            self.events["dijet_BvsL_max"] = self.events.dijet_csort.j1BvsL
            self.events["dijet_BvsL_min"] = self.events.dijet_csort.j2BvsL
            self.events["dijet_CvsL_max"] = self.events.dijet_csort.j1CvsL
            self.events["dijet_CvsL_min"] = self.events.dijet_csort.j2CvsL
            self.events["dijet_CvsB_max"] = self.events.dijet_csort.j1CvsB
            self.events["dijet_CvsB_min"] = self.events.dijet_csort.j2CvsB
            self.events["dijet_pt_max"] = self.events.dijet_csort.j1pt
            self.events["dijet_pt_min"] = self.events.dijet_csort.j2pt
            
            self.events["dibjet_m"] = self.events.dijet_bsort.mass
            self.events["dibjet_pt"] = self.events.dijet_bsort.pt
            self.events["dibjet_eta"] = self.events.dijet_bsort.eta
            self.events["dibjet_phi"] = self.events.dijet_bsort.phi
            self.events["dibjet_dr"] = self.events.dijet_bsort.deltaR
            self.events["dibjet_deltaPhi"] = self.events.dijet_bsort.deltaPhi
            self.events["dibjet_deltaEta"] = self.events.dijet_bsort.deltaEta
            self.events["dibjet_BvsL_max"] = self.events.dijet_bsort.j1BvsL
            self.events["dibjet_BvsL_min"] = self.events.dijet_bsort.j2BvsL
            self.events["dibjet_CvsL_max"] = self.events.dijet_bsort.j1CvsL
            self.events["dibjet_CvsL_min"] = self.events.dijet_bsort.j2CvsL
            self.events["dibjet_CvsB_max"] = self.events.dijet_bsort.j1CvsB
            self.events["dibjet_CvsB_min"] = self.events.dijet_bsort.j2CvsB
            self.events["dibjet_pt_max"] = self.events.dijet_bsort.j1pt
            self.events["dibjet_pt_min"] = self.events.dijet_bsort.j2pt
            self.events["dibjet_mass_max"] = self.events.dijet_bsort.j1mass
            self.events["dibjet_mass_min"] = self.events.dijet_bsort.j2mass

            self.events["dilep_m"] = self.events.ll.mass
            self.events["dilep_pt"] = self.events.ll.pt
            self.events["dilep_eta"] = self.events.ll.eta
            self.events["dilep_phi"] = self.events.ll.phi
            self.events["dilep_dr"] = self.events.ll.deltaR
            self.events["dilep_deltaPhi"] = self.events.ll.deltaPhi
            self.events["dilep_deltaEta"] = self.events.ll.deltaEta
            
            # self.events["ZH_pt_ratio"] = self.events.dijet_pt/self.events.dilep_pt
            # self.events["ZH_deltaPhi"] = np.abs(self.events.ll.delta_phi(self.events.dijet_csort))
            
            self.events["ZHbb_pt_ratio"] = self.events.dibjet_pt/self.events.dilep_pt
            self.events["VHbb_pt_ratio"] = self.events.ZHbb_pt_ratio
            
            self.events["ZHbb_deltaPhi"] = np.abs(self.events.ll.delta_phi(self.events.dijet_bsort))
            self.events["VHbb_deltaPhi"] = self.events.ZHbb_deltaPhi
            
            self.events["ZHbb_deltaR"] = np.abs(self.events.ll.delta_r(self.events.dijet_bsort))
            self.events["VHbb_deltaR"] = self.events.ZHbb_deltaR
            
            if self.run_bdt:
                odd_events = self.events[all_event_mask]
                # Create a record of variables to be dumped as root/parquete file:
                variables_to_process = ak.zip({
                    "events_dilep_m": odd_events["dilep_m"],
                    "events_dilep_pt": odd_events["dilep_pt"],
                    "events_dilep_dr": odd_events["dilep_dr"],
                    "events_dilep_deltaPhi": odd_events["dilep_deltaPhi"],
                    "events_dilep_deltaEta": odd_events["dilep_deltaEta"],
                    # "events_dibjet_m": odd_events["dibjet_m"],  # Commented out as per the features list
                    "events_dibjet_pt": odd_events["dibjet_pt"],
                    "events_dibjet_dr": odd_events["dibjet_dr"],
                    "events_dibjet_deltaPhi": odd_events["dibjet_deltaPhi"],
                    "events_dibjet_deltaEta": odd_events["dibjet_deltaEta"],
                    "events_dibjet_pt_max": odd_events["dibjet_pt_max"],
                    "events_dibjet_pt_min": odd_events["dibjet_pt_min"],
                    "events_dibjet_mass_max": odd_events["dibjet_mass_max"],
                    "events_dibjet_mass_min": odd_events["dibjet_mass_min"],
                    "events_dibjet_BvsL_max": odd_events["dibjet_BvsL_max"],
                    "events_dibjet_BvsL_min": odd_events["dibjet_BvsL_min"],
                    "events_dibjet_CvsB_max": odd_events["dibjet_CvsB_max"],
                    "events_dibjet_CvsB_min": odd_events["dibjet_CvsB_min"],
                    "events_VHbb_pt_ratio": odd_events["VHbb_pt_ratio"],
                    "events_VHbb_deltaPhi": odd_events["VHbb_deltaPhi"],
                    "events_VHbb_deltaR": odd_events["VHbb_deltaR"]
                })

                df = ak.to_pandas(variables_to_process)
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
            
            self.events["NaL"] = get_additionalleptons(
                self.events.ElectronGood, self.events.MuonGood, 1
            ) # number of additional leptons
            
            self.events["lead_lep"] = ak.firsts(self.events.LeptonGood)
            self.events["W_candidate"] = self.events.lead_lep + self.events.MET_used
            self.events["W_m"] = self.events.W_candidate.mass
            self.events["W_pt"] = self.events.W_candidate.pt
            self.events["W_mt"] = np.sqrt(2*self.events.lead_lep.pt*self.events.MET_used.pt*(1-np.cos(self.events.lead_lep.delta_phi(self.events.MET_used))))
            
#             # Step 1: Calculate delta_r for each b_jet with respect to lead_lep
#             delta_rs = self.events.BJetGood.delta_r(self.events.lead_lep)

#             # Step 2: Find the index of the b_jet with the minimum delta_r
#             min_delta_r_index = ak.argmin(delta_rs, axis=1, keepdims=True)

#             # Step 3: Select the b_jet with the minimum delta_r
#             self.events["b_jet"] = self.events.BJetGood[min_delta_r_index]
            
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
            
            self.events["dibjet_m"] = self.events.dijet_bsort.mass
            self.events["dibjet_pt"] = self.events.dijet_bsort.pt
            self.events["dibjet_eta"] = self.events.dijet_bsort.eta
            self.events["dibjet_phi"] = self.events.dijet_bsort.phi
            self.events["dibjet_dr"] = self.events.dijet_bsort.deltaR
            self.events["dibjet_deltaPhi"] = self.events.dijet_bsort.deltaPhi
            self.events["dibjet_deltaEta"] = self.events.dijet_bsort.deltaEta
            self.events["dibjet_BvsL_max"] = self.events.dijet_bsort.j1BvsL
            self.events["dibjet_BvsL_min"] = self.events.dijet_bsort.j2BvsL
            self.events["dibjet_CvsL_max"] = self.events.dijet_bsort.j1CvsL
            self.events["dibjet_CvsL_min"] = self.events.dijet_bsort.j2CvsL
            self.events["dibjet_CvsB_max"] = self.events.dijet_bsort.j1CvsB
            self.events["dibjet_CvsB_min"] = self.events.dijet_bsort.j2CvsB
            self.events["dibjet_pt_max"] = self.events.dijet_bsort.j1pt
            self.events["dibjet_pt_min"] = self.events.dijet_bsort.j2pt
            self.events["dibjet_mass_max"] = self.events.dijet_bsort.j1mass
            self.events["dibjet_mass_min"] = self.events.dijet_bsort.j2mass
            
            self.events["lep_pt"] = self.events.lead_lep.pt
            self.events["lep_eta"] = self.events.lead_lep.eta
            self.events["lep_phi"] = self.events.lead_lep.phi
            self.events["lep_m"] = self.events.lead_lep.mass
            
            self.events["lead_b"] = ak.firsts(self.events.JetsBvsL)
            self.events["deltaR_Leadb_Lep"] = self.events.lead_b.delta_r(self.events.lead_lep)
            self.events["deltaPhi_Leadb_Lep"] = np.abs(delta_phi(self.events.lead_lep.phi, self.events.lead_b.phi))
            self.events["deltaEta_Leadb_Lep"] = np.abs(self.events.lead_lep.eta - self.events.lead_b.eta)
            
            # self.events["deltaR_l1_b"] = np.sqrt((self.events.lead_lep.eta - self.events.b_jet.eta)**2 + (self.events.lead_lep.phi - self.events.b_jet.phi)**2)
            # self.events["deltaPhi_jet1_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,0]))
            # self.events["deltaPhi_jet2_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,1]))
            
            self.events["WHbb_pt_ratio"] = self.events.dibjet_pt/self.events.W_pt
            self.events["VHbb_pt_ratio"] = self.events.WHbb_pt_ratio
            
            self.events["WHbb_deltaPhi"] = np.abs(self.events.W_candidate.delta_phi(self.events.dijet_bsort))
            self.events["VHbb_deltaPhi"] = self.events.WHbb_deltaPhi
            
            self.events["WHbb_deltaEta"] = np.abs(self.events.W_candidate.eta - self.events.dijet_bsort.eta)
            self.events["VHbb_deltaEta"] = self.events.WHbb_deltaEta
            
            self.events["WHbb_deltaR"] = np.abs(self.events.W_candidate.delta_r(self.events.dijet_bsort))
            self.events["VHbb_deltaR"] = self.events.WHbb_deltaR
            
            self.events["WH_deltaPhi"] = np.abs(self.events.W_candidate.delta_phi(self.events.dijet_csort))
            
            self.events["deltaPhi_l1_j1"] = np.abs(delta_phi(self.events.lead_lep.phi, self.events.dijet_bsort.j1Phi))
            self.events["deltaPhi_l1_MET"] = np.abs(delta_phi(self.events.lead_lep.phi, self.events.MET_used.phi))
            
            # self.events["b_CvsL"] = self.events.b_jet.btagDeepFlavCvL
            # self.events["b_CvsB"] = self.events.b_jet.btagDeepFlavCvB
            # self.events["b_Btag"] = self.events.b_jet.btagDeepFlavB
            
            self.events["neutrino_from_W"] = get_nu_4momentum(self.events.lead_lep, self.events.MET_used)
            self.events["top_candidate"] = self.events.lead_lep + self.events.lead_b + self.events.neutrino_from_W
            self.events["top_mass"] = (self.events.lead_lep + self.events.lead_b + self.events.neutrino_from_W).mass
            
            if self.run_bdt:
                odd_events = self.events[all_event_mask]
                # Create a record of variables to be dumped as root/parquete file:
                variables_to_process = ak.zip({
                    "W_mt": self.events["W_mt"],
                    "W_pt": self.events["W_pt"],
                    "pt_miss": self.events["pt_miss"],
                    "WH_deltaPhi": self.events["WH_deltaPhi"],
                    "deltaPhi_l1_j1": self.events["deltaPhi_l1_j1"],
                    "deltaPhi_l1_MET": self.events["deltaPhi_l1_MET"],
                    "top_mass": self.events["top_mass"]
                })
            
                df = ak.to_pandas(variables_to_process)
                columns_to_exclude = ['dibjet_m']
                df = df.drop(columns=columns_to_exclude, errors='ignore')
                self.channel = "1L"
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

        if self.proc_type=="ZNuNu":
            ### General
            self.events["Z_candidate"] = self.events.MET_used
            self.events["Z_pt"] = self.events.Z_candidate.pt
            self.events["Z_eta"] = self.events.Z_candidate.eta
            self.events["Z_phi"] = self.events.Z_candidate.phi
            
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
            
            self.events["dibjet_m"] = self.events.dijet_bsort.mass
            self.events["dibjet_pt"] = self.events.dijet_bsort.pt
            self.events["dibjet_eta"] = self.events.dijet_bsort.eta
            self.events["dibjet_phi"] = self.events.dijet_bsort.phi
            self.events["dibjet_dr"] = self.events.dijet_bsort.deltaR
            self.events["dibjet_deltaPhi"] = self.events.dijet_bsort.deltaPhi
            self.events["dibjet_deltaEta"] = self.events.dijet_bsort.deltaEta
            self.events["dibjet_BvsL_max"] = self.events.dijet_bsort.j1BvsL
            self.events["dibjet_BvsL_min"] = self.events.dijet_bsort.j2BvsL
            self.events["dibjet_CvsL_max"] = self.events.dijet_bsort.j1CvsL
            self.events["dibjet_CvsL_min"] = self.events.dijet_bsort.j2CvsL
            self.events["dibjet_CvsB_max"] = self.events.dijet_bsort.j1CvsB
            self.events["dibjet_CvsB_min"] = self.events.dijet_bsort.j2CvsB
            self.events["dibjet_pt_max"] = self.events.dijet_bsort.j1pt
            self.events["dibjet_pt_min"] = self.events.dijet_bsort.j2pt
            self.events["dibjet_mass_max"] = self.events.dijet_bsort.j1mass
            self.events["dibjet_mass_min"] = self.events.dijet_bsort.j2mass
            
            self.events["ZHbb_pt_ratio"] = self.events.dijet_bsort.pt/self.events.Z_candidate.pt
            self.events["VHbb_pt_ratio"] = self.events.ZHbb_pt_ratio
            
            self.events["ZH_deltaPhi"] = np.abs(self.events.Z_candidate.delta_phi(self.events.dijet_bsort))
            self.events["VHbb_deltaPhi"] = self.events.ZH_deltaPhi
            
            self.events["deltaPhi_jet1_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetsBvsL[:,0]))
            self.events["deltaPhi_jet2_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetsBvsL[:,1]))
            
            if self.run_bdt:
                odd_events = self.events[all_event_mask]
                # Create a record of variables to be dumped as root/parquete file:
                variables_to_process = ak.zip({
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
                    "Z_pt": self.events["Z_pt"]
                })
            
                df = ak.to_pandas(variables_to_process)
                # columns_to_exclude = ['dilep_m']
                df = df.drop(columns=columns_to_exclude, errors='ignore')
                self.channel = "0L"
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

