from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
import workflow
from workflow import VHccBaseProcessor

import CommonSelectors
from CommonSelectors import *

import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(CommonSelectors)

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
    f"{localdir}/datasets/Run3_MC_OtherBkg.json",
    f"{localdir}/datasets/Run3_DATA.json",
    f"{localdir}/datasets/Run3_MC_Sig.json",
]

parameters["proc_type"] = "ZLL"
parameters["save_arrays"] = False
parameters["LightGBM_model"] = f"{localdir}/Models/ZH_Hto2C_Zto2L_2022_postEE/model_DY.txt"
parameters["DNN_model"] = f"{localdir}/Models/ZH_Hto2C_Zto2L_2022_postEE/dnn_model.h5"
parameters["separate_models"] = True
parameters["LigtGBM_low"] = f"{localdir}/Models/ZH_Hto2C_Zto2L_2022_postEE/_low/model_DY.txt"
parameters["LigtGBM_high"] = f"{localdir}/Models/ZH_Hto2C_Zto2L_2022_postEE/_high/model_DY.txt"
parameters["DNN_low"] = f"{localdir}/Models/ZH_Hto2C_Zto2L_2022_postEE/_low/dnn_model_DY.h5"
parameters["DNN_high"] = f"{localdir}/Models/ZH_Hto2C_Zto2L_2022_postEE/_high/dnn_model_DY.h5"

cfg = Configurator(
    parameters = parameters,
    datasets = {
        #"jsons": files_2016 + files_2017 + files_2018,
        "jsons": files_Run3,
        
        "filter" : {
            "samples": [
                #"DATA_DoubleMuon",
                #"DATA_DoubleEG", # in 2016/2017
                "DATA_EGamma",   # in 2018/2022/2023
                ##"DATA_SingleMuon",
                ##"DATA_SingleElectron",
	        #"WW", "WZ", "ZZ",
                "DYJetsToLL_FxFx",
                #"DYJetsToLL_MLM",
                #"TTToSemiLeptonic",
                #"DYJetsToLL_MiNNLO",
                #"DYJetsToLL_MiNNLO_ZptWei",
                "TTTo2L2Nu",
                "ZH_Hto2C_Zto2L"
            ],
            "samples_exclude" : [],
            #"year": ['2017']
            #"year": ['2016_PreVFP', '2016_PostVFP','2017','2018']
            #"year": ['2022_preEE','2022_postEE','2023_preBPix','2023_postBPix']

            "year": ['2022_preEE','2022_postEE']
            #"year": ['2023_preBPix']
        },

        #"subsamples": {
        #    'DYJetsToLL_MLM': {
        #        'DiJet_incl': [passthrough],
        #        'DiJet_bx': [DiJet_bx],
        #        'DiJet_cx': [DiJet_cx],
        #        'DiJet_ll': [DiJet_ll],
        #    },
        #    'DYJetsToLL_FxFx': {
        #        'DiJet_incl': [passthrough],
        #        'DiJet_bx': [DiJet_bx],
        #        'DiJet_cx': [DiJet_cx],
        #        'DiJet_ll': [DiJet_ll],
        #    }
        #}

    },

    workflow = VHccBaseProcessor,

    #skim = [get_HLTsel(primaryDatasets=["SingleMuon","SingleEle"]),
    skim = [get_HLTsel(primaryDatasets=["DoubleMuon","DoubleEle"]),
            get_nObj_min(4, 18., "Jet")],

    preselections = [ll_2j],
    categories = {
        #"baseline_2L2J": [passthrough],
        "baseline_2L2J_no_ctag": [passthrough],
        #"baseline_2L2J_ctag": [passthrough],
        #"baseline_2L2J_ctag_calib": [passthrough],
        "presel_mumu_2J": [mumu_2j],
        "presel_ee_2J": [ee_2j],
        
        "SR_mumu_2J_cJ": [Zmumu_2j, ctag_j1, dijet_mass_cut],
        "SR_ee_2J_cJ": [Zee_2j, ctag_j1, dijet_mass_cut],
        "SR_ll_2J_cJ": [Zll_2j, ctag_j1, dijet_mass_cut],

        "SR_ll_2J_cJ_low":  [Zll_2j, dijet_mass_cut, dilep_pt60to150],
        "SR_ll_2J_cJ_high": [Zll_2j, dijet_mass_cut, dilep_pt150to2000],

        

        "CR_ll_2J_LF": [Zll_2j, antictag_j1, dijet_mass_cut],
        "CR_ll_2J_HF": [Zll_2j, btag_j1, dijet_mass_cut],
        "CR_ll_2J_CC": [Zll_2j, ctag_j1, dijet_invmass_cut],
        "CR_ll_4J_TT": [ll_antiZ_4j, btag_j1, dijet_mass_cut]
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
        #"bysample": { "DYJetsToLL_MiNNLO_ZptWei": {"inclusive": ["genWeight"] } }
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
        **lepton_hists(coll="LeptonGood", pos=0),
        **lepton_hists(coll="LeptonGood", pos=1),
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),

        **jet_hists(coll="JetsCvsL", pos=0),
	**jet_hists(coll="JetsCvsL", pos=1),

        "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),
        
        "dilep_m" : HistConf( [Axis(coll="ll", field="mass", bins=100, start=0, stop=200, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_m_zoom" : HistConf( [Axis(coll="ll", field="mass", bins=40, start=70, stop=110, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_pt" : HistConf( [Axis(coll="ll", field="pt", bins=100, start=0, stop=400, label=r"$p_T{\ell\ell}$ [GeV]")] ),
        "dilep_dr" : HistConf( [Axis(coll="ll", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{\ell\ell}$")] ),
        "dilep_deltaPhi": HistConf( [Axis(field="dilep_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{\ell\ell}$")] ),
        "dilep_deltaEta": HistConf( [Axis(field="dilep_deltaEta", bins=50, start=0, stop=3.0, label=r"$\Delta \eta_{\ell\ell}$")] ),

        "dilep_dijet_ratio": HistConf( [Axis(field="ZH_pt_ratio", bins=100, start=0, stop=2, label=r"$\frac{p_T(jj)}{p_T(\ell\ell)}$")] ),
        "dilep_dijet_dphi": HistConf( [Axis(field="ZH_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi (\ell\ell, jj)$")] ),
        "dilep_l2j1": HistConf( [Axis(field="deltaPhi_l2_j1", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi (\ell_2, j_1)$")] ),
        "dilep_l2j2": HistConf( [Axis(field="deltaPhi_l2_j2", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi (\ell_2, j_2)$")] ),

        "dijet_m" : HistConf( [Axis(field="dijet_m", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_pt" : HistConf( [Axis(field="dijet_pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_dr" : HistConf( [Axis(field="dijet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_deltaPhi": HistConf( [Axis(field="dijet_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{jj}$")] ),
        "dijet_deltaEta": HistConf( [Axis(field="dijet_deltaEta", bins=50, start=0, stop=4, label=r"$\Delta \eta_{jj}$")] ),
        "dijet_pt_j1" : HistConf( [Axis(field="dijet_pt_max", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_pt_j2" : HistConf( [Axis(field="dijet_pt_min", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_CvsL_j1" : HistConf( [Axis(field="dijet_CvsL_max", bins=24, start=0, stop=1, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_CvsL_j2" : HistConf( [Axis(field="dijet_CvsL_min", bins=24, start=0, stop=1, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_CvsB_j1" : HistConf( [Axis(field="dijet_CvsB_max", bins=24, start=0, stop=1, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_CvsB_j2" : HistConf( [Axis(field="dijet_CvsB_min", bins=24, start=0, stop=1, label=r"$p_T{jj}$ [GeV]")] ),
        
        "dijet_csort_m" : HistConf( [Axis(coll="dijet_csort", field="mass", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_csort_pt" : HistConf( [Axis(coll="dijet_csort", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_csort_dr" : HistConf( [Axis(coll="dijet_csort", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),


        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=700, label=r"Jet HT [GeV]")] ),
        "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),
        "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=50, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

        "BDT": HistConf( [Axis(field="BDT", bins=24, start=0, stop=1, label="BDT")],
                         only_categories = ['SR_mumu_2J_cJ','SR_ee_2J_cJ','SR_ll_2J_cJ','SR_ll_2J_cJ_low','SR_ll_2J_cJ_high']),
        "DNN": HistConf( [Axis(field="DNN", bins=24, start=0, stop=1, label="DNN")],
                         only_categories = ['SR_mumu_2J_cJ','SR_ee_2J_cJ','SR_ll_2J_cJ','SR_ll_2J_cJ_low','SR_ll_2J_cJ_high']),
        
        
        # 2D histograms:
        "Njet_Ht": HistConf([ Axis(coll="events", field="nJetGood",bins=[0,2,3,4,8],
                                   type="variable",   label="N. Jets (good)"),
                              Axis(coll="events", field="JetGood_Ht",
                                   bins=[0,80,150,200,300,450,700],
                                   type="variable",
                                   label="Jets $H_T$ [GeV]")]),
        
        "dphi_jj_dr_jj": HistConf([ Axis(field="dijet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$"),
                                    Axis(field="dijet_deltaPhi", bins=50, start=-1, stop=3.5, label=r"$\Delta \phi_{jj}$")]),
    }
)
