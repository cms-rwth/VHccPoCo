from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.lib.cut_functions import get_nPVgood, goldenJson, eventFlags
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common.common import common_weights
import vjet_weights
from vjet_weights import *

import click

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
                                                  f"{localdir}/params/trainings_Hbb.yaml",
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
parameters["save_arrays"] = True
parameters["separate_models"] = False
parameters["run_bdt"] = True
parameters['run_dnn'] = False
ctx = click.get_current_context()
outputdir = ctx.params.get('outputdir')

outVariables = [
    "nJet", "EventNr",
    "dilep_m", "dilep_pt", "dilep_dr", "dilep_deltaPhi", "dilep_deltaEta",
    "dibjet_m", "dibjet_pt", "dibjet_dr", "dibjet_deltaPhi", "dibjet_deltaEta",
    "dibjet_pt_max", "dibjet_pt_min",
    "dibjet_mass_max", "dibjet_mass_min",
    "dibjet_BvsL_max", "dibjet_BvsL_min", "dibjet_CvsL_max", "dibjet_CvsL_min", "dibjet_CvsB_max", "dibjet_CvsB_min",
    "VHbb_pt_ratio", "VHbb_deltaPhi", "VHbb_deltaR", "BDT"
]

cfg = Configurator(
    parameters = parameters,
    weights_classes = common_weights + [custom_weight_vjet],
    datasets = {
        #"jsons": files_2016 + files_2017 + files_2018,
        "jsons": files_Run3,
        
        "filter" : {
            "samples": [
                # "DATA_DoubleMuon",
                # "DATA_EGamma",
                # "WW",
                # "WZ", 
                # "ZZ",
                "ZH_Hto2C_Zto2L",
                # "ggZH_Hto2C_Zto2L",
                "ZH_Hto2B_Zto2L",
                # "ggZH_Hto2B_Zto2L",
                # "DYJetsToLL_FxFx",
                # "DYJetsToLL_PT_FxFx",
                # "DYJetsToLL_NJ_FxFx",
                # "SingleTop",
                # "TTTo2L2Nu",
                # "TTToHadrons",
                # "TTToSemiLeptonic"
            ],
            "samples_exclude" : [],
            #"year": ['2022_preEE','2022_postEE','2023_preBPix','2023_postBPix']

            "year": ['2022_preEE','2022_postEE']
        },

        "subsamples": {
            
          # 'DYJetsToLL_FxFx': {
          #      'DiJet_incl': [passthrough],
          #      'DiJet_bx': [DiJet_bx],
          #      'DiJet_cx': [DiJet_cx],
          #      'DiJet_ll': [DiJet_ll],
          #  },
            
           # 'DYJetsToLL_NJ_FxFx': {
           #     'DiJet_NJ_incl': [passthrough],
           #     'DiJet_NJ_bx': [DiJet_bx],
           #     'DiJet_NJ_cx': [DiJet_cx],
           #     'DiJet_NJ_ll': [DiJet_ll],
           # },
            
#           'DYJetsToLL_PT_FxFx': {
#                'DiJet_pT_incl': [passthrough],
#                'DiJet_pT_bx': [DiJet_bx],
#                'DiJet_pT_cx': [DiJet_cx],
#                'DiJet_pT_ll': [DiJet_ll],
#           }
        }

    },

    workflow = VHbbBaseProcessor,
    workflow_options = {"dump_columns_as_arrays_per_chunk": 
                        "root://eosuser.cern.ch//eos/user/l/lichengz/Column_output_vhbb_zll_STOP_1203/"} if parameters["save_arrays"] else {},
    
    # save_skimmed_files = "root://eosuser.cern.ch://eos/user/l/lichengz/skimmed_samples/Run3ZLL/",
    skim = [get_HLTsel(primaryDatasets=["DoubleMuon","DoubleEle"]),
            get_nObj_min(4, 18., "Jet"),
            get_nPVgood(1), eventFlags, goldenJson],

    preselections = [ZLLHBB_2J(), 
                     dijet_mass(50, 250, False), 
                     dibjet_pt_cut(True, 50, 0, 0), 
                     missing_pt_cut(True, 60),
                     dibjet_eta_cut(1.0),
                     dibjet_pt_cut(False, 0, 30, 30),
                     bJ_mass_cut_5to30_5to30
                    ],
  
    categories = {
        
        "baseline_ZLLHBB_2J_ll": [passthrough],
        "baseline_ZLLHBB_2J_ee": [ZLLHBB_2J('el')],
        "baseline_ZLLHBB_2J_mm": [ZLLHBB_2J('mu')],

        "SR_2L2B_ll": [VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                  dilep_mass_window(False, 75, 105), dijet_mass(90, 150, False),
                  nAddJetCut(1, False), nAddLep_cut_0,
                  bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        
        "SR_2L2B_ee": [ZLLHBB_2J('el'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                  dilep_mass_window(False, 75, 105), dijet_mass(90, 150, False),
                  nAddJetCut(1, False), nAddLep_cut_0,
                  bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        
        "SR_2L2B_mm": [ZLLHBB_2J('mu'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                  dilep_mass_window(False, 75, 105), dijet_mass(90, 150, False),
                  nAddJetCut(1, False), nAddLep_cut_0,
                  bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],

        "CR_LF_ll": [VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_window(False, 75, 105), 
                nAddJetCut(2, False), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'M', True), bjet_tagger('DeepFlav', 1, 'L', True)],
        
        "CR_LF_ee": [ZLLHBB_2J('el'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_window(False, 75, 105), 
                nAddJetCut(2, False), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'M', True), bjet_tagger('DeepFlav', 1, 'L', True)],
        
        "CR_LF_mm": [ZLLHBB_2J('mu'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_window(False, 75, 105), 
                nAddJetCut(2, False), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'M', True), bjet_tagger('DeepFlav', 1, 'L', True)],

        "CR_B_ll": [VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
               dilep_mass_window(False, 75, 105), dijet_mass(90, 150, False), 
               nAddJetCut(1, False), nAddLep_cut_0,
               bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', True)],
        
        "CR_B_ee": [ZLLHBB_2J('el'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
               dilep_mass_window(False, 75, 105), dijet_mass(90, 150, False), 
               nAddJetCut(1, False), nAddLep_cut_0,
               bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', True)],
        
        "CR_B_mm": [ZLLHBB_2J('mu'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
               dilep_mass_window(False, 75, 105), dijet_mass(90, 150, False), 
               nAddJetCut(1, False), nAddLep_cut_0,
               bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', True)],

        "CR_BB_ll": [VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_window(False, 85, 97), dijet_mass(90, 150, True), 
                nAddJetCut(1, False), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        
        "CR_BB_ee": [ZLLHBB_2J('el'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_window(False, 85, 97), dijet_mass(90, 150, True), 
                nAddJetCut(1, False), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        
        "CR_BB_mm": [ZLLHBB_2J('mu'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_window(False, 85, 97), dijet_mass(90, 150, True), 
                nAddJetCut(1, False), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],

        "CR_TT_ll": [VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_window(True, 75, 120), dijet_mass(90, 150, False), 
                nAddJetCut(2, False), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        
        "CR_TT_ee": [ZLLHBB_2J('el'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_window(True, 75, 120), dijet_mass(90, 150, False), 
                nAddJetCut(2, False), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        
        "CR_TT_mm": [ZLLHBB_2J('mu'), VH_dPhi_cut(2.5), VH_dR_cut_3p6, HV_pTRatio_cut_0p5to2,
                dilep_mass_window(True, 75, 120), dijet_mass(90, 150, False), 
                nAddJetCut(2, False), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
    },
  
    
    columns = {
        "common": {
            "bycategory": {
                    "baseline_ZLLHBB_2J_ll": [ ColOut("events", outVariables, flatten=False), ],
                    "SR_2L2B_ll": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_LF_ll": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_B_ll": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_BB_ll": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_TT_ll": [ ColOut("events", outVariables, flatten=False), ],
                
                    "baseline_ZLLHBB_2J_ee": [ ColOut("events", outVariables, flatten=False), ],
                    "SR_2L2B_ee": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_LF_ee": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_B_ee": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_BB_ee": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_TT_ee": [ ColOut("events", outVariables, flatten=False), ],
                
                    "baseline_ZLLHBB_2J_mm": [ ColOut("events", outVariables, flatten=False), ],
                    "SR_2L2B_mm": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_LF_mm": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_B_mm": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_BB_mm": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_TT_mm": [ ColOut("events", outVariables, flatten=False), ],
                }
         },
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
            # "DYJetsToLL_PT_FxFx": {"inclusive": ["weight_vjet"] },
            # "DYJetsToLL_NJ_FxFx": {"inclusive": ["weight_vjet"] },
            #"DYJetsToLL_MiNNLO_ZptWei": {"inclusive": ["genWeight"] }
        },
    },
    
    variations = {
        "weights": {
            "common": {
                "inclusive": [
                    "pileup",
                    "sf_mu_id", "sf_mu_iso",
                    # "sf_ele_reco", "sf_ele_id",
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
      
        # **lepton_hists(coll="LeptonGood", pos=0),
        # **lepton_hists(coll="LeptonGood", pos=1),
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        # **jet_hists(coll="JetGood", pos=0),
        # **jet_hists(coll="JetGood", pos=1),

        "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),
        
        "dilep_m" : HistConf( [Axis(coll="ll", field="mass", bins=100, start=0, stop=200, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_m_zoom" : HistConf( [Axis(coll="ll", field="mass", bins=40, start=70, stop=110, label=r"$M_{\ell\ell}$ [GeV]")] ),
        "dilep_pt" : HistConf( [Axis(coll="ll", field="pt", bins=90, start=0, stop=450, label=r"$p_T{\ell\ell}$ [GeV]")] ),
        "dilep_dr" : HistConf( [Axis(coll="ll", field="deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{\ell\ell}$")] ),
        "dilep_deltaPhi": HistConf( [Axis(field="dilep_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{\ell\ell}$")] ),
        "dilep_deltaEta": HistConf( [Axis(field="dilep_deltaEta", bins=50, start=0, stop=3.0, label=r"$\Delta \eta_{\ell\ell}$")] ),
        
        "dibjet_m" : HistConf( [Axis(field="dibjet_m", bins=100, start=0, stop=600, label=r"$M_{bb}$ [GeV]")] ),
        "dibjet_m_zoom" : HistConf( [Axis(field="dibjet_m", bins=25, start=0, stop=600, label=r"$M_{bb}$ [GeV]")] ),
        "dibjet_pt" : HistConf( [Axis(field="dibjet_pt", bins=100, start=0, stop=400, label=r"$p_T{bb}$ [GeV]")] ),
        "dibjet_dr" : HistConf( [Axis(field="dibjet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{bb}$")] ),
        "dibjet_deltaPhi": HistConf( [Axis(field="dibjet_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{bb}$")] ),
        "dibjet_deltaEta": HistConf( [Axis(field="dibjet_deltaEta", bins=50, start=0, stop=4, label=r"$\Delta \eta_{bb}$")] ),
        "dibjet_pt_max" : HistConf( [Axis(field="dibjet_pt_max", bins=100, start=0, stop=400, label=r"$p_T{b1}$ [GeV]")] ),
        "dibjet_pt_min" : HistConf( [Axis(field="dibjet_pt_min", bins=100, start=0, stop=400, label=r"$p_T{b2}$ [GeV]")] ),
        "dibjet_mass_max" : HistConf( [Axis(field="dibjet_mass_max", bins=100, start=0, stop=600, label=r"$M_{b1}$ [GeV]")] ),
        "dibjet_mass_min" : HistConf( [Axis(field="dibjet_mass_min", bins=100, start=0, stop=600, label=r"$M_{b2}$ [GeV]")] ),
        "dibjet_BvsL_max" : HistConf( [Axis(field="dibjet_BvsL_max", bins=24, start=0, stop=1, label=r"$BvsL_{bj1}$ [GeV]")] ),
        "dibjet_BvsL_min" : HistConf( [Axis(field="dibjet_BvsL_min", bins=24, start=0, stop=1, label=r"$BvsL_{bj2}$ [GeV]")] ),

        "VHbb_pt_ratio" : HistConf( [Axis(field="VHbb_pt_ratio", bins=24, start=0, stop=2, label=r"$p_T{H}/p_T{Z}")] ),
        "VHbb_deltaPhi" : HistConf( [Axis(field="VHbb_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{ZH}$")] ),
        "VHbb_deltaR" : HistConf( [Axis(field="VHbb_deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{VH}$")] ),
        "BDT_coarse": HistConf( [Axis(field="BDT", bins=24, start=0, stop=1, label="BDT")],
                         only_categories = ['SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
                                            'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
                                            'SR_2L2B_mm','CR_BB_mm','CR_TT_mm'
                                           ]),
        
        "BDT": HistConf( [Axis(field="BDT", bins=1000, start=0, stop=1, label="BDT")],
                         only_categories = ['SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
                                            'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
                                            'SR_2L2B_mm','CR_BB_mm','CR_TT_mm'
                                           ]),
        
        "DNN_coarse": HistConf( [Axis(field="DNN", bins=24, start=0, stop=1, label="DNN")],
                         only_categories = ['SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
                                            'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
                                            'SR_2L2B_mm','CR_BB_mm','CR_TT_mm'
                                           ]),
        
        "DNN": HistConf( [Axis(field="DNN", bins=1000, start=0, stop=1, label="DNN")],
                         only_categories = ['SR_2L2B_ll','CR_BB_ll','CR_TT_ll',
                                            'SR_2L2B_ee','CR_BB_ee','CR_TT_ee',
                                            'SR_2L2B_mm','CR_BB_mm','CR_TT_mm'
                                           ]),
                        
        
#         "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=700, label=r"Jet HT [GeV]")] ),
#         "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),
#         "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=50, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

#         
#                          
#         
        
        
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
