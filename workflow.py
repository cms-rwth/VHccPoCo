import awkward as ak
import numpy as np
import uproot
import pandas as pd
import warnings
import os

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


    def evaluateBDT(self):
        # Template for BDT evaluation method

        # Some useful links:
        ## xgboost wrapper in coffea:
        ## https://coffeateam.github.io/coffea/api/coffea.ml_tools.xgboost_wrapper.html#coffea.ml_tools.xgboost_wrapper
        ## Its usage example:
        ## https://github.com/CoffeaTeam/coffea/blob/d167d169821f4ffca94db65cf82d74daafb890e4/tests/test_ml_tools.py#L166-L183


        # Read the model file, depending on the channel
        if self.proc_type=='ZLL':
            model_file = self.params.xgboost[self._year].model_file
        elif self.proc_type=='WLNu':
            model_file = self.params.xgboost[self._year].model_file
        elif self.proc_type=='ZNuNu':
            model_file = self.params.xgboost[self._year].model_file
        else:
            raise('There is not BDT training model for this channel:', self.proc_type)
        # print("XGBoost Model file: ", model_file)

        # Create input data

        # Evaluate the BDT score

        bdt_score = np.zeros(len(self.events), dtype=np.float64)

        return bdt_score

    # Function that defines common variables employed in analyses and save them as attributes of `events`
    def define_common_variables_before_presel(self, variation):
        self.events["JetGood_Ht"] = ak.sum(abs(self.events.JetGood.pt), axis=1)

    def define_common_variables_after_presel(self, variation):
        self.events["dijet"] = get_dijet(self.events.JetGood)

        #self.events["dijet_pt"] = self.events.dijet.pt


        ### General        
        self.events["dijet_m"] = self.events.dijet.mass
        self.events["dijet_pt"] = self.events.dijet.pt
        self.events["dijet_dr"] = self.events.dijet.deltaR
        self.events["dijet_deltaPhi"] = self.events.dijet.deltaPhi
        self.events["dijet_deltaEta"] = self.events.dijet.deltaEta

        if self.proc_type=="ZLL":

            self.events["dilep_m"] = self.events.ll.mass
            self.events["dilep_pt"] = self.events.ll.pt
            self.events["dilep_dr"] = self.events.ll.deltaR
            self.events["dilep_deltaPhi"] = self.events.ll.deltaPhi
            self.events["dilep_deltaEta"] = self.events.ll.deltaEta

            self.events["ZH_pt_ratio"] = self.events.dijet.pt/self.events.ll.pt
            self.events["ZH_deltaPhi"] = np.abs(self.events.ll.delta_phi(self.events.dijet))

            # why cant't we use delta_phi function here?
            self.angle21_gen = (abs(self.events.ll.l2phi - self.events.dijet.j1Phi) < np.pi)
            self.angle22_gen = (abs(self.events.ll.l2phi - self.events.dijet.j2Phi) < np.pi)
            self.events["deltaPhi_l2_j1"] = ak.where(self.angle21_gen, abs(self.events.ll.l2phi - self.events.dijet.j1Phi), 2*np.pi - abs(self.events.ll.l2phi - self.events.dijet.j1Phi))
            self.events["deltaPhi_l2_j2"] = ak.where(self.angle22_gen, abs(self.events.ll.l2phi - self.events.dijet.j2Phi), 2*np.pi - abs(self.events.ll.l2phi - self.events.dijet.j2Phi))
            self.events["deltaPhi_l2_j1"] = np.abs(delta_phi(self.events.ll.l2phi, self.events.dijet.j1Phi))


        if self.proc_type=="ZNuNu":
            self.events["deltaPhi_jet1_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,0]))
            self.events["deltaPhi_jet2_MET"] = np.abs(self.events.MET.delta_phi(self.events.JetGood[:,1]))

        self.events["JetsCvsL"] = CvsLsorted(self.events["JetGood"], self.params.ctagging.working_point[self._year])

        #print("Pt sort pt:", self.events["JetGood"][self.events["nJetGood"]>=3].pt)
        #print("CvsL sort pt:", self.events["JetsCvsL"][self.events["nJetGood"]>=3].pt)

        #print("Pt sort CvsL:", self.events["JetGood"][self.events["nJetGood"]>=3].btagDeepFlavCvL)
        #print("CvsL sort CvsL:", self.events["JetsCvsL"][self.events["nJetGood"]>=3].btagDeepFlavCvL)

        self.events["dijet_csort"] = get_dijet(self.events.JetsCvsL)


        self.events["BDT"] = self.evaluateBDT()

            
        if self.save_arrays:
            if self.proc_type=="ZLL":
                # Create a record of variables to be dumped as root/parquete file:
                variables_to_save = ak.zip({
                    "dilep_m": self.events["dilep_m"],
                    "dilep_pt": self.events["dilep_pt"],
                    "dilep_dr": self.events["dilep_deltaR"],
                    "dilep_deltaPhi": self.events["dilep_deltaPhi"],
                    "dilep_deltaEta": self.events["dilep_deltaEta"],

                    "dijet_m": self.events["dijet_m"],
                    "dijet_pt": self.events["dijet_pt"],
                    "dijet_dr": self.events["dijet_deltaR"],
                    "dijet_deltaPhi": self.events["dijet_deltaPhi"],
                    "dijet_deltaEta": self.events["dijet_deltaEta"],

                    "ZH_pt_ratio": self.events["ZH_pt_ratio"],
                    "ZH_deltaPhi": self.events["ZH_deltaPhi"],
                    "deltaPhi_l2_j1": self.events["deltaPhi_l2_j1"],
                    "deltaPhi_l2_j2": self.events["deltaPhi_l2_j2"],
                })

            elif self.proc_type=="WLNu":
                "TODO. Create arrays for WLNu channel"
                pass
            elif self.proc_type=="ZNuNu":
                "TODO. Create arrays for ZNuNu channel"
                pass
                    

        if self.save_arrays:
            # Here we write to root  and parquete files

            with warnings.catch_warnings():
                # Suppress FutureWarning
                warnings.filterwarnings("ignore", category=FutureWarning)

                # Check if the directory exists
                if not os.path.exists(f"Saved_root_files/{self.events.metadata['dataset']}"):
                    # If not, create it
                    os.system(f"mkdir -p Saved_root_files/{self.events.metadata['dataset']}")

                # Write the events to a ROOT file
                with uproot.recreate(f"Saved_root_files/{self.events.metadata['dataset']}/{self.events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(self.events.metadata['entrystart'])}_{int(self.events.metadata['entrystop'])}.root") as f:
                    f["variables"] = ak.to_pandas(variables_to_save)

                # Write the events to a Parquet file
                ak.to_pandas(ak.to_pandas(variables_to_save)).to_parquet(f"Saved_root_files/{self.events.metadata['dataset']}/{self.events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(self.events.metadata['entrystart'])}_{int(self.events.metadata['entrystop'])}_vars.parquet")
