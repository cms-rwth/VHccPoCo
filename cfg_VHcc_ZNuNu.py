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
    f"{localdir}/datasets/Run2UL2017_Signal.json"
]
files_2018 = [
    f"{localdir}/datasets/Run2UL2018_MC_VJets.json",
    f"{localdir}/datasets/Run2UL2018_MC_OtherBkg.json",
    f"{localdir}/datasets/Run2UL2018_DATA.json",
]

parameters["proc_type"] = "ZNuNu"
parameters["save_arrays"] = False
parameters["LightGBM_model"] = f"{localdir}/Models/ZH_HToCC_ZToNuNu_2017/model_QCD.txt"
parameters["DNN_model"] = f"{localdir}/Models/ZH_HToCC_ZToNuNu_2017/dnn_model_QCD.h5"
parameters["separate_models"] = False

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": files_2016 + files_2017 + files_2018,

        "filter" : {
            "samples": [
                "DATA_MET",
                "WW", 
                #"WZ", 
                "ZZ",
                "QCD", 
                "QCD_Mu", "QCD_EM",
                #"DYJetsToLL_FxFx",
                #"ZJetsToNuNu_FxFx",
                "WJetsToLNu_FxFx",
                "WJetsToQQ", "ZJetsToQQ",
                "TTToSemiLeptonic", "TTTo2L2Nu",
                "TTToHadrons",
                "ZH_HToCC_ZToNuNu"
            ],
            "samples_exclude" : [],
            "year": ['2017'],
            #"year": ['2016_PreVFP', '2016_PostVFP', '2017', '2018']
        },
        "subsamples": {
            'DYJetsToLL_MLM': {
                'DiJet_incl': [passthrough],
                'DiJet_bx': [DiJet_bx],
                'DiJet_cx': [DiJet_cx],
                'DiJet_ll': [DiJet_ll],
            },
            'DYJetsToLL_FxFx': {
                'DiJet_incl': [passthrough],
                'DiJet_bx': [DiJet_bx],
                'DiJet_cx': [DiJet_cx],
                'DiJet_ll': [DiJet_ll],
            },
            'WJetsToLNu_FxFx': {
                'DiJet_incl': [passthrough],
                'DiJet_bx': [DiJet_bx],
		'DiJet_cx': [DiJet_cx],
                'DiJet_ll': [DiJet_ll],
            }
        }
    },

    workflow = VHccBaseProcessor,

    skim = [get_HLTsel(primaryDatasets=["MET"]),
            get_nObj_min(2, 32., "Jet")],

    preselections = [met_2jets_0lep],

    categories = {
        "presel_Met_2J_no_ctag": [passthrough],
        "presel_Met_2J_ctag": [passthrough],
        "presel_Met_2J_ctag_calib": [passthrough],
        "baseline_Met_2J_ptcut":  [dijet_pt_cut, jet_met_dphi_cut],
        
        "SR_ZNuNu_2J_cJ":  [dijet_pt_cut, jet_met_dphi_cut, ctag_j1, dijet_mass_cut],

        "CR_ZNuNu_2J_LF": [dijet_pt_cut, jet_met_dphi_cut, antictag_j1, dijet_mass_cut],
	    "CR_ZNuNu_2J_HF": [dijet_pt_cut, jet_met_dphi_cut, btag_j1, dijet_mass_cut],
        "CR_ZNuNu_2J_CC": [dijet_pt_cut, jet_met_dphi_cut, ctag_j1, dijet_invmass_cut],
        "CR_ZNuNu_4J_TT": [dijet_pt_cut, jet_met_dphi_cut, btag_j1, dijet_mass_cut]

    },

    weights = {
        "common": {
            "inclusive": ["signOf_genWeight","lumi","XS",
                          "pileup",
                          "sf_mu_id","sf_mu_iso",
                          "sf_ele_reco","sf_ele_id",
                          ],
            "bycategory" : {
                "presel_Met_2J_ctag" : ["sf_ctag"],
                "presel_Met_2J_ctag_calib" : ["sf_ctag", "sf_ctag_calib"],
            }
        },
        "bysample": {
        }
    },

    variations = {
        "weights": {
            "common": {
                "inclusive": [
                    "pileup",
                    "sf_mu_id", "sf_mu_iso",
                    "sf_ele_reco", "sf_ele_id",
                ],
                "bycategory" : {
                }
            },
            "bysample": {
            }
        },
    },


    variables = {
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),

        **jet_hists(coll="JetsCvsL", pos=0),
        **jet_hists(coll="JetsCvsL", pos=1),

        "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),


        "dijet_nom_m" : HistConf( [Axis(coll="dijet", field="mass", bins=100, start=0, stop=700, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_nom_dr" : HistConf( [Axis(coll="dijet", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_nom_pt" : HistConf( [Axis(coll="dijet", field="pt", bins=100, start=0, stop=500, label=r"$p_T{jj}$ [GeV]")] ),

        "dijet_m" : HistConf( [Axis(field="dijet_m", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_pt" : HistConf( [Axis(field="dijet_pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_dr" : HistConf( [Axis(field="dijet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_deltaPhi": HistConf( [Axis(field="dijet_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{jj}$")] ),
        "dijet_deltaEta": HistConf( [Axis(field="dijet_deltaEta", bins=50, start=0, stop=4, label=r"$\Delta \eta_{jj}$")] ),
        "dijet_pt_j1" : HistConf( [Axis(field="dijet_pt_max", bins=100, start=0, stop=400, label=r"$p_T{j1}$ [GeV]")] ),
        "dijet_pt_j2" : HistConf( [Axis(field="dijet_pt_min", bins=100, start=0, stop=400, label=r"$p_T{j2}$ [GeV]")] ),
        "dijet_CvsL_j1" : HistConf( [Axis(field="dijet_CvsL_max", bins=24, start=0, stop=1, label=r"$CvsL_{j1}$ [GeV]")] ),
        "dijet_CvsL_j2" : HistConf( [Axis(field="dijet_CvsL_min", bins=24, start=0, stop=1, label=r"$CvsL_{j2}$ [GeV]")] ),
        "dijet_CvsB_j1" : HistConf( [Axis(field="dijet_CvsB_max", bins=24, start=0, stop=1, label=r"$CvsB_{j1}$ [GeV]")] ),
        "dijet_CvsB_j2" : HistConf( [Axis(field="dijet_CvsB_min", bins=24, start=0, stop=1, label=r"$CvsB_{j2}$ [GeV]")] ),
        
        "Z_pt": HistConf( [Axis(field="Z_pt", bins=100, start=0, stop=400, label=r"$p_T{Z}$ [GeV]")] ),
        "dilep_dijet_ratio": HistConf( [Axis(field="ZH_pt_ratio", bins=100, start=0, stop=2, label=r"$\frac{p_T(jj)}{p_T(\ell\ell)}$")] ),
        "dilep_dijet_dphi": HistConf( [Axis(field="ZH_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi (\ell\ell, jj)$")] ),
        
        
        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=900, label=r"Jet HT [GeV]")] ),
        "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=100, stop=600, label=r"MET $p_T$ [GeV]")] ),
        "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=64, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

        "met_deltaPhi_j1": HistConf( [Axis(field="deltaPhi_jet1_MET", bins=64, start=0, stop=math.pi, label=r"$\Delta\phi$(MET, jet 1)")] ),
        "met_deltaPhi_j2": HistConf( [Axis(field="deltaPhi_jet2_MET", bins=64, start=0, stop=math.pi, label=r"$\Delta\phi$(MET, jet 2)")] ),

        "BDT": HistConf( [Axis(field="BDT", bins=24, start=0, stop=1, label="BDT")],
                         only_categories = ['SR_ZNuNu_2J_cJ','baseline_Met_2J_ptcut']),
        "DNN": HistConf( [Axis(field="DNN", bins=24, start=0, stop=1, label="DNN")],
                         only_categories = ['SR_ZNuNu_2J_cJ','baseline_Met_2J_ptcut']),
        
        
        # 2D plots
	"Njet_Ht": HistConf([ Axis(coll="events", field="nJetGood",bins=[0,2,3,4,8],
                                   type="variable",   label="N. Jets (good)"),
                              Axis(coll="events", field="JetGood_Ht",
                                   bins=[0,80,150,200,300,450,700],
                                   type="variable",
                                   label="Jets $H_T$ [GeV]")]),

    }
)
