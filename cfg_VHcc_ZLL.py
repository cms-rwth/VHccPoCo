from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.lib.cut_functions import get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common.common import common_weights
import workflow_VHcc
from pocket_coffea.lib.columns_manager import ColOut
import click
from workflow_VHcc import VHccBaseProcessor
import vjet_weights 
from vjet_weights import *
import CommonSelectors
from CommonSelectors import *
import MVA
from MVA.gnnmodels import GraphAttentionClassifier

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
    f"{localdir}/datasets/Run3_MC_Sig.json"
]

parameters["proc_type"] = "ZLL"
parameters["save_arrays"] = False
parameters["separate_models"] = False
parameters['run_dnn'] = True
parameters['run_gnn'] = True
ctx = click.get_current_context()
outputdir = ctx.params.get('outputdir')

cfg = Configurator(
    parameters = parameters,
    weights_classes = common_weights + [custom_weight_vjet],
    datasets = {
        #"jsons": files_2016 + files_2017 + files_2018,
        "jsons": files_Run3,
        
        "filter" : {
            "samples": [
                "DATA_DoubleMuon",
                #"DATA_DoubleEG", # in 2016/2017
                "DATA_EGamma",   # in 2018/2022/2023
                ##"DATA_SingleMuon",
                ##"DATA_SingleElectron",
	            "WW", "WZ", "ZZ",
                "DYJetsToLL_FxFx",
                #"DYJetsToLL_MLM",
                #"WJetsToLNu_FxFx",
                #"TTToSemiLeptonic",
                #"DYJetsToLL_MiNNLO",
                #"DYJetsToLL_MiNNLO_ZptWei",
                "TTTo2L2Nu",
                "ZH_Hto2C_Zto2L",
                "ZH_Hto2B_Zto2L"
            ],
            "samples_exclude" : [],
            #"year": ['2017']
            #"year": ['2016_PreVFP', '2016_PostVFP','2017','2018']
            #"year": ['2022_preEE','2022_postEE','2023_preBPix','2023_postBPix']

            "year": ['2022_postEE']
            #"year": ['2023_preBPix']
        },

        "subsamples": {
        #    'DYJetsToLL_MLM': {
        #        'DiJet_incl': [passthrough],
        #        'DiJet_bx': [DiJet_bx],
        #        'DiJet_cx': [DiJet_cx],
        #        'DiJet_ll': [DiJet_ll],
        #    },
            'DYJetsToLL_FxFx': {
                'DiJet_incl': [passthrough],
                'DiJet_bx': [DiJet_bx],
                'DiJet_cx': [DiJet_cx],
                'DiJet_ll': [DiJet_ll],
            }
        }

    },

    workflow = VHccBaseProcessor,
    
    workflow_options = {"dump_columns_as_arrays_per_chunk": f"{outputdir}/Saved_columnar_arrays_ZLL"},

    #skim = [get_HLTsel(primaryDatasets=["SingleMuon","SingleEle"]),
    skim = [get_HLTsel(primaryDatasets=["DoubleMuon","DoubleEle"]),
            get_nObj_min(4, 18., "Jet"),
            get_nPVgood(1), eventFlags, goldenJson],

    preselections = [ll_2j()],
    categories = {
        #"baseline_2L2J": [passthrough],
        "baseline_2L2J_no_ctag": [passthrough],
        #"baseline_2L2J_ctag": [passthrough],
        #"baseline_2L2J_ctag_calib": [passthrough],
        "presel_mm_2J": [ll_2j('mu')],
        "presel_ee_2J": [ll_2j('el')],
        
        "SR_mm_2J_cJ": [Zll_2j('mu'), ctag_j1, dijet_mass_cut],
        "SR_ee_2J_cJ": [Zll_2j('el'), ctag_j1, dijet_mass_cut],
        "SR_ll_2J_cJ": [Zll_2j(), ctag_j1, dijet_mass_cut],
        
        "SR_ll_2J_cJ_loPT": [Zll_2j(), ctag_j1, dijet_mass_cut, dilep_pt(60,150)],
        "SR_ll_2J_cJ_hiPT": [Zll_2j(), ctag_j1, dijet_mass_cut, dilep_pt(150,2000)],
        
        "CR_ee_2J_LF": [Zll_2j('el'), antictag_j1, dijet_mass_cut],
        "CR_ee_2J_HF": [Zll_2j('el'), btag_j1, dijet_mass_cut],
        "CR_ee_2J_CC": [Zll_2j('el'), ctag_j1, dijet_invmass_cut],
        "CR_ee_4J_TT": [ll_antiZ_4j('el'), btag_j1, dijet_mass_cut],
        
        "CR_mm_2J_LF": [Zll_2j('mu'), antictag_j1, dijet_mass_cut],
        "CR_mm_2J_HF": [Zll_2j('mu'), btag_j1, dijet_mass_cut],
        "CR_mm_2J_CC": [Zll_2j('mu'), ctag_j1, dijet_invmass_cut],
        "CR_mm_4J_TT": [ll_antiZ_4j('mu'), btag_j1, dijet_mass_cut],

        #"CR_ll_2J_LF": [Zll_2j('both'), antictag_j1, dijet_mass_cut],
        #"CR_ll_2J_HF": [Zll_2j('both'), btag_j1, dijet_mass_cut],
        #"CR_ll_2J_CC": [Zll_2j('both'), ctag_j1, dijet_invmass_cut],
        #"CR_ll_4J_TT": [ll_antiZ_4j('both'), btag_j1, dijet_mass_cut],
    },
    
    columns = {
        "common": {
            "bycategory": {
                    "SR_ll_2J_cJ": [
                        ColOut("events", ["EventNr", "dilep_m","dilep_pt","dilep_dr","dilep_deltaPhi","dilep_deltaEta",
                                    "dijet_m","dijet_pt","dijet_dr","dijet_deltaPhi","dijet_deltaEta",
                                    "dijet_CvsL_max","dijet_CvsL_min","dijet_CvsB_max","dijet_CvsB_min",
                                    "dijet_pt_max","dijet_pt_min",
                                    "ZH_pt_ratio","ZH_deltaPhi","deltaPhi_l2_j1","deltaPhi_l2_j2",
                                    "JetGood_btagCvL","JetGood_btagCvB",
                                    "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                                    "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                                    "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                                    "ll_pt","ll_eta","ll_phi","ll_mass",
                                    "MET_pt","MET_phi","nPV","LeptonCategory"], flatten=False),
                    ],
                    "baseline_2L2J_no_ctag": [
                        ColOut("events", ["EventNr", "dilep_m","dilep_pt","dilep_dr","dilep_deltaPhi","dilep_deltaEta",
                                    "dijet_m","dijet_pt","dijet_dr","dijet_deltaPhi","dijet_deltaEta",
                                    "dijet_CvsL_max","dijet_CvsL_min","dijet_CvsB_max","dijet_CvsB_min",
                                    "dijet_pt_max","dijet_pt_min",
                                    "ZH_pt_ratio","ZH_deltaPhi","deltaPhi_l2_j1","deltaPhi_l2_j2",
                                    "JetGood_btagCvL","JetGood_btagCvB",
                                    "JetGood_pt","JetGood_eta","JetGood_phi","JetGood_mass",
                                    "LeptonGood_miniPFRelIso_all","LeptonGood_pfRelIso03_all",
                                    "LeptonGood_pt","LeptonGood_eta","LeptonGood_phi","LeptonGood_mass",
                                    "ll_pt","ll_eta","ll_phi","ll_mass",
                                    "MET_pt","MET_phi","nPV","LeptonCategory"], flatten=False),
                    ]
                }
        },
        
    } if parameters["save_arrays"] else {},

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
            "DYJetsToLL_FxFx": {"inclusive": ["weight_vjet"] },
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
        "dijet_pt_j1" : HistConf( [Axis(field="dijet_pt_max", bins=100, start=0, stop=400, label=r"$p_T{j1}$ [GeV]")] ),
        "dijet_pt_j2" : HistConf( [Axis(field="dijet_pt_min", bins=100, start=0, stop=400, label=r"$p_T{j2}$ [GeV]")] ),
        "dijet_CvsL_j1" : HistConf( [Axis(field="dijet_CvsL_max", bins=24, start=0, stop=1, label=r"$CvsL_{j1}$ [GeV]")] ),
        "dijet_CvsL_j2" : HistConf( [Axis(field="dijet_CvsL_min", bins=24, start=0, stop=1, label=r"$CvsL_{j2}$ [GeV]")] ),
        "dijet_CvsB_j1" : HistConf( [Axis(field="dijet_CvsB_max", bins=24, start=0, stop=1, label=r"$CvsB_{j1}$ [GeV]")] ),
        "dijet_CvsB_j2" : HistConf( [Axis(field="dijet_CvsB_min", bins=24, start=0, stop=1, label=r"$CvsB_{j2}$ [GeV]")] ),
        
        "dijet_csort_m" : HistConf( [Axis(coll="dijet_csort", field="mass", bins=100, start=0, stop=600, label=r"$M_{jj}$ [GeV]")] ),
        "dijet_csort_pt" : HistConf( [Axis(coll="dijet_csort", field="pt", bins=100, start=0, stop=400, label=r"$p_T{jj}$ [GeV]")] ),
        "dijet_csort_dr" : HistConf( [Axis(coll="dijet_csort", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$")] ),


        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=700, label=r"Jet HT [GeV]")] ),
        "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),
        "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=50, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

        "BDT": HistConf( [Axis(field="BDT", bins=24, start=0, stop=1, label="BDT")],
                         only_categories = ['SR_mm_2J_cJ','SR_ee_2J_cJ','SR_ll_2J_cJ','SR_ll_2J_cJ_loPT','SR_ll_2J_cJ_hiPT']),
        "DNN": HistConf( [Axis(field="DNN", bins=24, start=0, stop=1, label="DNN")],
                         only_categories = ['SR_mm_2J_cJ','SR_ee_2J_cJ','SR_ll_2J_cJ','SR_ll_2J_cJ_loPT','SR_ll_2J_cJ_hiPT']),

        "GNN": HistConf( [Axis(field="GNN", bins=24, start=0, stop=1, label="GNN")],
                         only_categories = ['SR_mm_2J_cJ','SR_ee_2J_cJ','SR_ll_2J_cJ']),
        
        
        # 2D histograms:
        "Njet_Ht": HistConf([ Axis(coll="events", field="nJetGood",bins=[0,2,3,4,8],
                                   type="variable", label="N. Jets (good)"),
                              Axis(coll="events", field="JetGood_Ht",
                                   bins=[0,80,150,200,300,450,700],
                                   type="variable", label="Jets $H_T$ [GeV]")]),
        
        "dphi_jj_dr_jj": HistConf([ Axis(field="dijet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{jj}$"),
                                    Axis(field="dijet_deltaPhi", bins=50, start=-1, stop=3.5, label=r"$\Delta \phi_{jj}$")]),
    }
)
