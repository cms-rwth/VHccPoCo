from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.lib.cut_functions import get_nPVgood, goldenJson, eventFlags, get_JetVetoMap
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.lib.columns_manager import ColOut
import click
import workflow_VHcc
from workflow_VHcc import VHccBaseProcessor
import CommonSelectors
from CommonSelectors import *
import vjet_weights
from vjet_weights import *
import MVA
from MVA.gnnmodels import GraphAttentionClassifier
from MVA.training import process_gnn_inputs

import cloudpickle
cloudpickle.register_pickle_by_value(workflow_VHcc)
cloudpickle.register_pickle_by_value(CommonSelectors)
cloudpickle.register_pickle_by_value(vjet_weights)
cloudpickle.register_pickle_by_value(MVA)

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
                                                  f"{localdir}/params/trainings.yaml",
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
    f"{localdir}/datasets/Run2UL2017_Signal.json",
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

parameters["proc_type"] = "WLNu"
parameters["save_arrays"] = False
parameters['run_dnn'] = False
parameters['run_gnn'] = True
parameters["save_gnn_arrays"] = False
ctx = click.get_current_context()
outputdir = ctx.params.get('outputdir')

cfg = Configurator(
    parameters = parameters,
    weights_classes = common_weights + [custom_weight_vjet],
    datasets = {
        #"jsons": files_2016 + files_2017 + files_2018,
        #"jsons": files_2017,
        "jsons": files_Run3,

        "filter" : {
            "samples": [
                "DATA_SingleMuon",
                #"DATA_SingleElectron", # For 2017
                "DATA_EGamma",
                "WW",
                "WZ",
                "ZZ",
                #"QCD",
                "WJetsToLNu_FxFx",
                #"WJetsToLNu_NJPT_FxFx",
                #"WJetsToLNu_MLM",
                #"WJetsToQQ_MLM",
                #"DYJetsToLL_FxFx",
                "TTToSemiLeptonic",
                #"TTTo2L2Nu",
                "SingleTop",
                #"TTToHadrons",
                "WH_Hto2C_WtoLNu",
                "WH_Hto2B_WtoLNu",
            ],
            "samples_exclude" : [],
            #"year": ['2017']
            #"year": ['2016_PreVFP', '2016_PostVFP', '2017', '2018']
            #"year": ['2022_preEE','2022_postEE']
            "year": ['2022_preEE']
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
        },

    },

    workflow = VHccBaseProcessor,
    workflow_options = {"dump_columns_as_arrays_per_chunk": f"{outputdir}/Saved_columnar_arrays_WLNu"},

    skim = [get_HLTsel(primaryDatasets=["SingleMuon","SingleEle"]),
            get_nObj_min(3, 20., "Jet"), # in default jet collection there are leptons. So we ask for 1lep+2jets=3Jet objects
            get_JetVetoMap(),
            get_nPVgood(1), eventFlags, goldenJson],

    preselections = [lep_met_2jets],
    categories = {
        "baseline_1L2j": [passthrough],
        #"baseline_1L2J_no_ctag": [passthrough],
        #"baseline_1L2J_ctag": [passthrough],
        #"baseline_1L2J_ctag_calib": [passthrough],
        "presel_Wln_2J": [wlnu_plus_2j()],

        "SR_Wln_2J_cJ": [wlnu_plus_2j(), ctag_j1, dijet_mass_cut],
        "SR_Wmn_2J_cJ": [wlnu_plus_2j('mu'), ctag_j1, dijet_mass_cut],
        "SR_Wen_2J_cJ": [wlnu_plus_2j('el'), ctag_j1, dijet_mass_cut],

        "CR_Wmn_2J_LF": [wlnu_plus_2j('mu'), antictag_j1, dijet_mass_cut],
        "CR_Wmn_2J_HF": [wlnu_plus_2j('mu'), btag_j1, dijet_mass_cut],
        "CR_Wmn_2J_CC": [wlnu_plus_2j('mu'), ctag_j1, dijet_invmass_cut],
        "CR_Wmn_4J_TT": [wlnu_plus_2j('mu'), four_jets, btag_j1, dijet_mass_cut],

        "CR_Wen_2J_LF": [wlnu_plus_2j('el'), antictag_j1, dijet_mass_cut],
        "CR_Wen_2J_HF": [wlnu_plus_2j('el'), btag_j1, dijet_mass_cut],
        "CR_Wen_2J_CC": [wlnu_plus_2j('el'), ctag_j1, dijet_invmass_cut],
        "CR_Wen_4J_TT": [wlnu_plus_2j('el'), four_jets, btag_j1, dijet_mass_cut]

    },

    columns = {
        "common": {
            "bycategory": {
                "SR_Wln_2J_cJ": [
                    ColOut("events", ["EventNr", "dijet_m", "dijet_pt", "dijet_dr", "dijet_deltaPhi", "dijet_deltaEta",
                                      "dijet_CvsL_max", "dijet_CvsL_min", "dijet_CvsB_max", "dijet_CvsB_min",
                                      "dijet_pt_max", "dijet_pt_min", "W_mt", "W_pt", "pt_miss",
                                      "WH_deltaPhi", "deltaPhi_l1_j1", "deltaPhi_l1_MET", "deltaPhi_l1_b", "deltaEta_l1_b", "deltaR_l1_b",
                                      "b_CvsL", "b_CvsB", "b_Btag", "top_mass",
                                      "JetGood_btagCvL","JetGood_btagCvB",
                                      "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                                      "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                                      "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                                      "W_pt","W_eta","W_phi","W_mt",
                                      "PuppiMET_pt","PuppiMET_phi","nPV","W_m","LeptonCategory"] +
                           ["GNN","GNN_transformed"] if parameters['run_gnn'] else [], flatten=False),
                ],
                "presel_Wln_2J": [
                    ColOut("events", ["EventNr", "dijet_m", "dijet_pt", "dijet_dr", "dijet_deltaPhi", "dijet_deltaEta",
                                      "dijet_CvsL_max", "dijet_CvsL_min", "dijet_CvsB_max", "dijet_CvsB_min",
                                      "dijet_pt_max", "dijet_pt_min", "W_mt", "W_pt", "pt_miss",
                                      "WH_deltaPhi", "deltaPhi_l1_j1", "deltaPhi_l1_MET", "deltaPhi_l1_b", "deltaEta_l1_b", "deltaR_l1_b",
                                      "b_CvsL", "b_CvsB", "b_Btag", "top_mass",
                                      "JetGood_btagCvL","JetGood_btagCvB",
                                      "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                                      "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                                      "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                                      "W_pt","W_eta","W_phi","W_mt",
                                      "PuppiMET_pt","PuppiMET_phi","nPV","W_m","LeptonCategory"], flatten=False),
                ]
            }
        },
    } if parameters["save_arrays"] else {
        "common": {
            "bycategory": {
                    "SR_Wln_2J_cJ": [
                        ColOut("events", ["GNN","GNN_transformed"], flatten=False),
                    ]
                }
        },
    } if parameters['save_gnn_arrays'] else {},

    weights = {
        "common": {
            "inclusive": ["signOf_genWeight","lumi","XS",
                          "pileup",
                          "sf_mu_id","sf_mu_iso",
                          "sf_ele_reco","sf_ele_id",
                          #"sf_ctag", "sf_ctag_calib"
                          ],
            #"bycategory" : {
            #    "baseline_1L2J_ctag" : ["sf_ctag"],
            #    "baseline_1L2J_ctag_calib" : ["sf_ctag", "sf_ctag_calib"],
            #}
        },
        "bysample": {
            #"DYJetsToLL_FxFx": {"inclusive": ["weight_vjet"] },
            "WJetsToLNu_FxFx": {"inclusive": ["weight_vjet"] },
        }
    },

    variations = {
        "weights": {
            "common": {
                "inclusive": [
                    "pileup",
                    "sf_mu_id", "sf_mu_iso",
                    "sf_ele_reco", "sf_ele_id",
                    #"sf_ctag"
                ],
                "bycategory" : {
                }
            },
            "bysample": {}
        },
    },


    variables = {
        **lepton_hists(coll="LeptonGood", pos=0),
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),

        **jet_hists(coll="JetsCvsL", pos=0),
        **jet_hists(coll="JetsCvsL", pos=1),

        "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),

        "dijet_nom_m" : HistConf( [Axis(coll="dijet", field="mass", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_nom_m_zoom" : HistConf( [Axis(coll="dijet", field="mass", bins=50, start=50, stop=200, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_nom_dr" : HistConf( [Axis(coll="dijet", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_nom_pt" : HistConf( [Axis(coll="dijet", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),

        "dijet_csort_m" : HistConf( [Axis(coll="dijet_csort", field="mass", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_csort_dr" : HistConf( [Axis(coll="dijet_csort", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),
        "dijet_csort_pt" : HistConf( [Axis(coll="dijet_csort", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),

        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=700, label=r"Jet HT [GeV]")] ),
        "met_pt": HistConf( [Axis(coll="PuppiMET", field="pt", bins=50, start=0, stop=200, label=r"PuppiMET $p_T$ [GeV]")] ),
        "met_phi": HistConf( [Axis(coll="PuppiMET", field="phi", bins=64, start=-math.pi, stop=math.pi, label=r"PuppiMET $phi$")] ),

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

        "W_mt" : HistConf( [Axis(field="W_mt", bins=100, start=-10, stop=200, label=r"$Mt_{l\nu}$ [GeV]")] ),
        "W_m": HistConf( [Axis(field="W_m", bins=100, start=0, stop=200, label=r"$M_{l\nu}$ [GeV]")] ),
        "W_pt" : HistConf( [Axis(field="W_pt", bins=100, start=0, stop=200, label=r"$p_{T_{l\nu}}$ [GeV]")] ),
        "pt_miss" : HistConf( [Axis(field="pt_miss", bins=100, start=0, stop=200, label=r"$p_T^{miss}$ [GeV]")] ),
        "Wc_dijet_dphi": HistConf( [Axis(field="WH_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\frac{p_T(jj)}{p_T(l\nu)}$")] ),
        "deltaPhi_l1_j1": HistConf( [Axis(field="deltaPhi_l1_j1", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{l1,j1}$")] ),
        "deltaPhi_l1_MET": HistConf( [Axis(field="deltaPhi_l1_MET", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{l1,MET}$")] ),
        "deltaPhi_l1_b": HistConf( [Axis(field="deltaPhi_l1_b", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{l1,b}$")] ),
        "deltaEta_l1_b": HistConf( [Axis(field="deltaEta_l1_b", bins=50, start=0, stop=5, label=r"$\Delta \eta_{l1,b}$")] ),
        "deltaR_l1_b": HistConf( [Axis(field="deltaR_l1_b", bins=50, start=0, stop=5, label=r"$\Delta R_{l1,b}$")] ),
        "b_CvsL": HistConf( [Axis(field="b_CvsL", bins=24, start=0, stop=1, label=r"$CvsL_{b}$")] ),
        "b_CvsB": HistConf( [Axis(field="b_CvsB", bins=24, start=0, stop=1, label=r"$CvsB_{b}$")] ),
        "b_Btag": HistConf( [Axis(field="b_Btag", bins=24, start=0, stop=1, label=r"$Btag_{b}$")] ),
        "top_mass": HistConf( [Axis(field="top_mass", bins=100, start=0, stop=400, label=r"$M_{top}$ [GeV]")] ),

        "BDT": HistConf( [Axis(field="BDT", bins=10000, start=0, stop=1, label="BDT")],
                         only_categories = ['SR_Wmn_2J_cJ','SR_Wen_2J_cJ']),
        "DNN": HistConf( [Axis(field="DNN", bins=10000, start=0, stop=1, label="DNN")],
                         only_categories = ['SR_Wmn_2J_cJ','SR_Wen_2J_cJ']),
        "GNN": HistConf( [Axis(field="GNN", bins=10000, start=0, stop=1, label="GNN")],
                         only_categories = ['SR_Wln_2J_cJ','SR_Wmn_2J_cJ','SR_Wen_2J_cJ']),

        "BDT_coarse": HistConf( [Axis(field="BDT", bins=24, start=0, stop=1, label="BDT")],
                                only_categories = ['SR_Wln_2J_cJ','SR_Wmn_2J_cJ','SR_Wen_2J_cJ','presel_Wln_2J']),
        "DNN_coarse": HistConf( [Axis(field="DNN", bins=24, start=0, stop=1, label="DNN")],
                                only_categories = ['SR_Wln_2J_cJ','SR_Wmn_2J_cJ','SR_Wen_2J_cJ','presel_Wln_2J']),
        "GNN_coarse": HistConf( [Axis(field="GNN", bins=24, start=0, stop=1, label="GNN")],
                                only_categories = ['SR_Wln_2J_cJ','SR_Wmn_2J_cJ','SR_Wen_2J_cJ','presel_Wln_2J']),
      
        "GNN_transformed": HistConf( [Axis(field="GNN_transformed", bins=10000, start=0, stop=1, label="GNN")],
                         only_categories = ['SR_Wln_2J_cJ','SR_Wmn_2J_cJ','SR_Wen_2J_cJ']),        
        "GNN_transformed_coarse": HistConf( [Axis(field="GNN_transformed", bins=24, start=0, stop=1, label="GNN")],
                         only_categories = ['SR_Wln_2J_cJ','SR_Wmn_2J_cJ','SR_Wen_2J_cJ']),


        # 2D plots
        "Njet_Ht": HistConf([ Axis(coll="events", field="nJetGood",bins=[0,2,3,4,8],
                                   type="variable",   label="N. Jets (good)"),
                              Axis(coll="events", field="JetGood_Ht",
                                   bins=[0,80,150,200,300,450,700],
                                   type="variable", label="Jets $H_T$ [GeV]")]),

    }
)
