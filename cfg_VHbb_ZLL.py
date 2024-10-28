from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.lib.cut_functions import get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common.common import common_weights
# import workflow_VHbb
# from workflow_VHbb import VHbbBaseProcessor
import vjet_weights
from vjet_weights import *
import workflow_VHbb
from workflow_VHbb import VHbbBaseProcessor

import CommonSelectors
from CommonSelectors import *

import cloudpickle
cloudpickle.register_pickle_by_value(workflow_VHbb)
cloudpickle.register_pickle_by_value(CommonSelectors)
cloudpickle.register_pickle_by_value(vjet_weights)

import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/ctagging.yaml",
                                                  f"{localdir}/params/btagger.yaml",
                                                  f"{localdir}/params/xgboost.yaml",
                                                  update=True)
files_2016 = [
    f"{localdir}/datasets/Run2UL2016_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2016_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2016_DATA.json",
]
files_2017 = [
    f"{localdir}/datasets/Run2UL2017_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2017_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2017_DATA.json",
]
files_2018 = [
    f"{localdir}/datasets/Run2UL2018_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2018_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2018_DATA.json",
]
files_Run3 = [
    f"{localdir}/datasets/Run3_MC_VJets.json",
    f"{localdir}/datasets/Run3_MC_TOP.json",
    f"{localdir}/datasets/Run3_MC_OtherBkg.json",
    f"{localdir}/datasets/Run3_DATA.json",
    f"{localdir}/datasets/Run3_MC_Sig.json",
]

parameters["proc_type"] = "ZLL"
parameters["save_arrays"] = True
parameters["separate_models"] = False
parameters['run_dnn'] = False

cfg = Configurator(
    parameters = parameters,
    weights_classes = common_weights + [custom_weight_vjet],
    datasets = {
        #"jsons": files_2016 + files_2017 + files_2018,
        "jsons": files_Run3,
        
        "filter" : {
            "samples": [
                "DATA_DoubleMuon",
                "DATA_EGamma",
                "DATA_MuonEG",
                "ZH_Hto2B_Zto2L",
                "ZH_Hto2C_Zto2L",
                "DYJetsToLL_FxFx",
                "TTTo2L2Nu",
                "TTToHadrons",
                "TTToSemiLeptonic"
            ],
            "samples_exclude" : [],
            #"year": ['2022_preEE','2022_postEE','2023_preBPix','2023_postBPix']

            "year": ['2022_preEE']
        },

        "subsamples": {
          # 'DYJetsToLL_FxFx': {
          #      'DiJet_incl': [passthrough],
          #      'DiJet_bx': [DiJet_bx],
          #      'DiJet_cx': [DiJet_cx],
          #      'DiJet_ll': [DiJet_ll],
          #  }
        }

    },

    workflow = VHbbBaseProcessor,

    #skim = [get_HLTsel(primaryDatasets=["SingleMuon","SingleEle"]),
    skim = [get_HLTsel(primaryDatasets=["DoubleMuon","DoubleEle"]),
            get_nObj_min(4, 18., "Jet"),
            get_nPVgood(1), eventFlags, goldenJson],

    preselections = [ZLLHBB_2J, 
                     dijet_mass_50to250, 
                     dijet_pt_cut_50, 
                     missing_invpt_cut_60, 
                     dijet_eta_cut_1,
                     nAddLep_cut_0,
                     bJ_pT_cut_30_30,
                     bJ_mass_cut_5to30_5to30,
                    ],
  
    categories = {
      "baseline_ZLLHBB_2J": [passthrough],
      
      "SR_2L2B": [VH_dPhi_cut_2p5, VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                  dilep_mass_75to105, dijet_mass_90to150, 
                  nAddJet_cut_1, 
                  btag_j1_tight, btag_j2_medium],
      
      "CR_LF": [VH_dPhi_cut_2p5, VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_75to105, 
                nAddJet_cut_2, 
                btag_j1_medium_inv, btag_j2_loose_inv],
      
      "CR_B": [VH_dPhi_cut_2p5, VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
               dilep_mass_75to105, dijet_mass_90to150, 
               nAddJet_cut_1,
               btag_j1_tight, btag_j2_medium_inv],
      
      "CR_BB": [VH_dPhi_cut_2p5, VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_85to97, dijet_invmass_90to150, 
                nAddJet_cut_1, 
                btag_j1_tight, btag_j2_medium],
      
      "CR_TT": [VH_dPhi_cut_2p5, VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_invmass_75to120, dijet_mass_90to150, 
                nAddJet_cut_2, 
                btag_j1_tight, btag_j2_medium],
    },

    weights = {
        "common": {
            "inclusive": ["signOf_genWeight","lumi","XS",
                          "pileup", #Not in 2022/2023
                          "sf_mu_id","sf_mu_iso",
                          "sf_ele_reco","sf_ele_id",
                          #"sf_ctag", "sf_ctag_calib"
                          ],
            "bycategory" : {
                #"baseline_2L2J_ctag" : ["sf_ctag"],
                #"baseline_2L2J_ctag_calib": ["sf_ctag","sf_ctag_calib"]
            }
        },
        "bysample": {
            # "DYJetsToLL_FxFx": {"inclusive": ["weight_vjet"] },
            #"DYJetsToLL_MiNNLO_ZptWei": {"inclusive": ["genWeight"] }
            
        },
    },
    
    variations = {
        "weights": {
            "common": {
                "inclusive": [
                    "pileup",
                    "sf_mu_id", "sf_mu_iso",
                    "sf_ele_reco", "sf_ele_id",
                    #"sf_ctag",
                ]
            },
            "bysample": { }
        },
        #"shape": {
        #    "common":{
        #        #"inclusive": [ "JES_Total_AK4PFchs", "JER_AK4PFchs" ] # For Run2UL
        #        "inclusive": [ "JES_Total_AK4PFPuppi", "JER_AK4PFPuppi" ] # For Run3
        #    }
        #}
    },

    variables = {
      
      "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),
      "dijet_m" : HistConf( [Axis(field="dijet_m", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),

      "dibjet_m" : HistConf( [Axis(field="dibjet_m", bins=100, start=0, stop=600, label=r"$M_{bb}$ [GeV]")] ),
      "dibjet_BvsL_j1" : HistConf( [Axis(field="dibjet_BvsL_max", bins=24, start=0, stop=1, label=r"$BvsL_{bj1}$ [GeV]")] ),
      "dibjet_BvsL_j2" : HistConf( [Axis(field="dibjet_BvsL_min", bins=24, start=0, stop=1, label=r"$BvsL_{bj2}$ [GeV]")] ),
      
      "dilep_m" : HistConf( [Axis(coll="ll", field="mass", bins=100, start=0, stop=200, label=r"$M_{\ell\ell}$ [GeV]")] ),
      "dilep_m_zoom" : HistConf( [Axis(coll="ll", field="mass", bins=40, start=70, stop=110, label=r"$M_{\ell\ell}$ [GeV]")] ),
      "dilep_pt" : HistConf( [Axis(coll="ll", field="pt", bins=90, start=0, stop=450, label=r"$p_T{\ell\ell}$ [GeV]")] ),
      
      "dibjet_dr" : HistConf( [Axis(field="dibjet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{bb}$")] ),
      
#         **lepton_hists(coll="LeptonGood", pos=0),
#         **lepton_hists(coll="LeptonGood", pos=1),
#         **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
#         **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
#         **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
#         **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
#         **jet_hists(coll="JetGood", pos=0),
#         **jet_hists(coll="JetGood", pos=1),

#         **jet_hists(coll="JetsCvsL", pos=0),
# 	    **jet_hists(coll="JetsCvsL", pos=1),

#         "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),
        
#         "dilep_m" : HistConf( [Axis(coll="ll", field="mass", bins=100, start=0, stop=200, label=r"$M_{\ell\ell}$ [GeV]")] ),
#         "dilep_m_zoom" : HistConf( [Axis(coll="ll", field="mass", bins=40, start=70, stop=110, label=r"$M_{\ell\ell}$ [GeV]")] ),
#         "dilep_pt" : HistConf( [Axis(coll="ll", field="pt", bins=100, start=0, stop=400, label=r"$p_T{\ell\ell}$ [GeV]")] ),
#         "dilep_dr" : HistConf( [Axis(coll="ll", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{\ell\ell}$")] ),
#         "dilep_deltaPhi": HistConf( [Axis(field="dilep_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{\ell\ell}$")] ),
#         "dilep_deltaEta": HistConf( [Axis(field="dilep_deltaEta", bins=50, start=0, stop=3.0, label=r"$\Delta \eta_{\ell\ell}$")] ),

#         "dilep_dijet_ratio": HistConf( [Axis(field="ZH_pt_ratio", bins=100, start=0, stop=2, label=r"$\frac{p_T(jj)}{p_T(\ell\ell)}$")] ),
#         "dilep_dijet_dphi": HistConf( [Axis(field="ZH_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi (\ell\ell, jj)$")] ),
#         "dilep_l2j1": HistConf( [Axis(field="deltaPhi_l2_j1", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi (\ell_2, j_1)$")] ),
#         "dilep_l2j2": HistConf( [Axis(field="deltaPhi_l2_j2", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi (\ell_2, j_2)$")] ),

#         "dijet_m" : HistConf( [Axis(field="dijet_m", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
#         "dijet_pt" : HistConf( [Axis(field="dijet_pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
#         "dijet_dr" : HistConf( [Axis(field="dijet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
#         "dijet_deltaPhi": HistConf( [Axis(field="dijet_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{jj}$")] ),
#         "dijet_deltaEta": HistConf( [Axis(field="dijet_deltaEta", bins=50, start=0, stop=4, label=r"$\Delta \eta_{jj}$")] ),
#         "dijet_pt_j1" : HistConf( [Axis(field="dijet_pt_max", bins=100, start=0, stop=400, label=r"$p_T{j1}$ [GeV]")] ),
#         "dijet_pt_j2" : HistConf( [Axis(field="dijet_pt_min", bins=100, start=0, stop=400, label=r"$p_T{j2}$ [GeV]")] ),
#         "dijet_CvsL_j1" : HistConf( [Axis(field="dijet_CvsL_max", bins=24, start=0, stop=1, label=r"$CvsL_{j1}$ [GeV]")] ),
#         "dijet_CvsL_j2" : HistConf( [Axis(field="dijet_CvsL_min", bins=24, start=0, stop=1, label=r"$CvsL_{j2}$ [GeV]")] ),
#         "dijet_CvsB_j1" : HistConf( [Axis(field="dijet_CvsB_max", bins=24, start=0, stop=1, label=r"$CvsB_{j1}$ [GeV]")] ),
#         "dijet_CvsB_j2" : HistConf( [Axis(field="dijet_CvsB_min", bins=24, start=0, stop=1, label=r"$CvsB_{j2}$ [GeV]")] ),
        
#         "dijet_csort_m" : HistConf( [Axis(coll="dijet_csort", field="mass", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
#         "dijet_csort_pt" : HistConf( [Axis(coll="dijet_csort", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
#         "dijet_csort_dr" : HistConf( [Axis(coll="dijet_csort", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),


#         "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=700, label=r"Jet HT [GeV]")] ),
#         "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),
#         "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=50, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

#         "BDT": HistConf( [Axis(field="BDT", bins=24, start=0, stop=1, label="BDT")],
#                          only_categories = ['SR_mumu_2J_cJ','SR_ee_2J_cJ','SR_ll_2J_cJ','SR_ll_2J_cJ_low','SR_ll_2J_cJ_high']),
#         "DNN": HistConf( [Axis(field="DNN", bins=24, start=0, stop=1, label="DNN")],
#                          only_categories = ['SR_mumu_2J_cJ','SR_ee_2J_cJ','SR_ll_2J_cJ','SR_ll_2J_cJ_low','SR_ll_2J_cJ_high']),
        
        
#         # 2D histograms:
#         "Njet_Ht": HistConf([ Axis(coll="events", field="nJetGood",bins=[0,2,3,4,8],
#                                    type="variable",   label="N. Jets (good)"),
#                               Axis(coll="events", field="JetGood_Ht",
#                                    bins=[0,80,150,200,300,450,700],
#                                    type="variable",
#                                    label="Jets $H_T$ [GeV]")]),
        
#         "dphi_jj_dr_jj": HistConf([ Axis(field="dijet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$"),
#                                     Axis(field="dijet_deltaPhi", bins=50, start=-1, stop=3.5, label=r"$\Delta \phi_{jj}$")]),
    }
)
