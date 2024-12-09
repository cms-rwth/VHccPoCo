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

parameters["proc_type"] = "WLNu"
parameters["save_arrays"] = False
parameters["separate_models"] = False
parameters["run_bdt"] = False
parameters['run_dnn'] = False
ctx = click.get_current_context()
outputdir = ctx.params.get('outputdir')

outVariables = [
    "nJet", 
    "dibjet_m", "dibjet_pt", "dibjet_dr",
    "dibjet_deltaPhi", "dibjet_deltaEta", "dibjet_pt_max", "dibjet_pt_min",
    "dibjet_mass_max", "dibjet_mass_min", "dibjet_BvsL_max", "dibjet_BvsL_min",
    "JetGood_Ht", "lep_pt", "lep_eta", "lep_phi", "pt_miss",
    "deltaR_Leadb_Lep", "deltaPhi_Leadb_Lep", "deltaEta_Leadb_Lep",
    "VHbb_pt_ratio", "VHbb_deltaPhi", "VHbb_deltaR",
    "deltaPhi_l1_j1", "deltaPhi_l1_MET", "top_mass"
]


cfg = Configurator(
    parameters = parameters,
    weights_classes = common_weights + [custom_weight_vjet],
    datasets = {
        #"jsons": files_2016 + files_2017 + files_2018,
        "jsons": files_Run3,
        
        "filter" : {
            "samples": [
                # "DATA_SingleMuon",
                # "DATA_EGamma",
                # "WminusH_Hto2B_WtoLNu",
                # "WplusH_Hto2B_WtoLNu",
                # "WW",
                # "WZ",
                # "ZZ",
                # "DYJetsToLL_FxFx",
                # "WminusH_HTo2C_WToLNu",
                # "WplusH_Hto2C_WtoLNu",
                "WJetsToLNu_MLM",
                # "TTTo2L2Nu",
                # "SingleTop",
                # "TTToSemiLeptonic",
            ],
            "samples_exclude" : [],
            #"year": ['2022_preEE','2022_postEE','2023_preBPix','2023_postBPix']

            "year": ['2022_preEE','2022_postEE']
        },

        "subsamples": {
          'WJetsToLNu_MLM': {
                'DiJet_incl': [passthrough],
                'DiJet_bx': [DiJet_bx],
                'DiJet_cx': [DiJet_cx],
                'DiJet_ll': [DiJet_ll],
            }
        }

    },

    workflow = VHbbBaseProcessor,
    workflow_options = {"dump_columns_as_arrays_per_chunk": 
                        "root://eosuser.cern.ch//eos/user/l/lichengz/Saved_output_arrays_WLNu_VJet_1204/"} if parameters["save_arrays"] else {},

    skim = [get_HLTsel(primaryDatasets=["SingleMuon","SingleEle"]),
            get_nObj_min(3, 20., "Jet"), 
            # in default jet collection there are leptons. So we ask for 1lep+2jets=3Jet objects
            get_nPVgood(1), eventFlags, goldenJson],

    preselections = [WLNuHBB_2J(), 
                     missing_pt_cut(False, 30),
                     dijet_mass(50, 250, False), 
                     bJ_mass_cut_5to30_5to30,
                     dibjet_pt_cut(False, 0, 30, 30),
                     dibjet_pt_cut(True, 100, 0, 0), 
                     dibjet_eta_cut(1.8),
                    ],
  
    categories = {
        "baseline_WLNuHBB_2J_ln": [passthrough],
        "baseline_WLNuHBB_2J_en": [WLNuHBB_2J('el')],
        "baseline_WLNuHBB_2J_mn": [WLNuHBB_2J('mu')],
        

        "SR_LNu2B_ln": [VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                   dijet_mass(90, 150, False),
                   nAddJetCut(1, False), LepMetDPhi(2), nAddLep_cut_0,
                   bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        "SR_LNu2B_en": [WLNuHBB_2J('el'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                   dijet_mass(90, 150, False),
                   nAddJetCut(1, False), LepMetDPhi(2), nAddLep_cut_0,
                   bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        "SR_LNu2B_mn": [WLNuHBB_2J('mu'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                   dijet_mass(90, 150, False),
                   nAddJetCut(1, False), LepMetDPhi(2), nAddLep_cut_0,
                   bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],

        "CR_LF_ln": [VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                nAddJetCut(2, False), LepMetDPhi(2), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'M', True), bjet_tagger('DeepFlav', 1, 'L', True)],
        "CR_LF_en": [WLNuHBB_2J('el'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                nAddJetCut(2, False), LepMetDPhi(2), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'M', True), bjet_tagger('DeepFlav', 1, 'L', True)],
        "CR_LF_mn": [WLNuHBB_2J('mu'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                nAddJetCut(2, False), LepMetDPhi(2), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'M', True), bjet_tagger('DeepFlav', 1, 'L', True)],

        "CR_B_ln": [VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
               dijet_mass(90, 150, False), 
               nAddJetCut(1, False), LepMetDPhi(2), nAddLep_cut_0,
               bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'L', True)],
        "CR_B_en": [WLNuHBB_2J('mu'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
               dijet_mass(90, 150, False), 
               nAddJetCut(1, False), LepMetDPhi(2), nAddLep_cut_0,
               bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'L', True)],
        "CR_B_mn": [WLNuHBB_2J('mu'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
               dijet_mass(90, 150, False), 
               nAddJetCut(1, False), LepMetDPhi(2), nAddLep_cut_0,
               bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'L', True)],

        "CR_BB_ln": [VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                dijet_mass(90, 150, True), 
                nAddJetCut(1, False), LepMetDPhi(2), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        "CR_BB_en": [WLNuHBB_2J('el'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                dijet_mass(90, 150, True), 
                nAddJetCut(1, False), LepMetDPhi(2), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        "CR_BB_mn": [WLNuHBB_2J('mu'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                dijet_mass(90, 150, True), 
                nAddJetCut(1, False), LepMetDPhi(2), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],

        "CR_TT_ln": [VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                dijet_mass(90, 150, False), 
                nAddJetCut(2, True), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        "CR_TT_en": [WLNuHBB_2J('el'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                dijet_mass(90, 150, False), 
                nAddJetCut(2, True), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
        "CR_TT_mn": [WLNuHBB_2J('mu'), VH_dPhi_cut(1.5), VH_dEta_cut_2, HV_pTRatio_cut_0p5to2,
                dijet_mass(90, 150, False), 
                nAddJetCut(2, True), nAddLep_cut_0,
                bjet_tagger('DeepFlav', 0, 'T', False), bjet_tagger('DeepFlav', 1, 'M', False)],
    },
  
    columns = {
        "common": {
            "bycategory": {
                    "baseline_WLNuHBB_2J_ln": [ ColOut("events", outVariables, flatten=False), ],
                    "SR_LNu2B_ln": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_LF_ln": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_B_ln": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_BB_ln": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_TT_ln": [ ColOut("events", outVariables, flatten=False), ],
                
                    "baseline_WLNuHBB_2J_en": [ ColOut("events", outVariables, flatten=False), ],
                    "SR_LNu2B_en": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_LF_en": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_B_en": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_BB_en": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_TT_en": [ ColOut("events", outVariables, flatten=False), ],
                
                    "baseline_WLNuHBB_2J_mn": [ ColOut("events", outVariables, flatten=False), ],
                    "SR_LNu2B_mn": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_LF_mn": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_B_mn": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_BB_mn": [ ColOut("events", outVariables, flatten=False), ],
                    "CR_TT_mn": [ ColOut("events", outVariables, flatten=False), ],
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
            "WJetsToLNu_MLM": {"inclusive": ["weight_vjet"] },
            
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
      
        # **lepton_hists(coll="LeptonGood", pos=0),
        # **lepton_hists(coll="LeptonGood", pos=1),
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=5, start=0, stop=5),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=5, start=0, stop=5),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        # **jet_hists(coll="JetGood", pos=0),
        # **jet_hists(coll="JetGood", pos=1),

        "nJet": HistConf( [Axis(field="nJet", bins=15, start=0, stop=15, label=r"nJet direct from NanoAOD")] ),
        
        "dibjet_m" : HistConf( [Axis(field="dibjet_m", bins=100, start=0, stop=600, label=r"$M_{bb}$ [GeV]")] ),
        "dibjet_m_zoom" : HistConf( [Axis(field="dibjet_m", bins=25, start=0, stop=600, label=r"$M_{bb}$ [GeV]")] ),
        
        "dibjet_pt" : HistConf( [Axis(field="dibjet_pt", bins=100, start=0, stop=400, label=r"$p_T{bb}$ [GeV]")] ),
        "dibjet_pt_zoom" : HistConf( [Axis(field="dibjet_pt", bins=25, start=0, stop=400, label=r"$p_T{bb}$ [GeV]")] ),
        
        "dibjet_dr" : HistConf( [Axis(field="dibjet_dr", bins=50, start=0, stop=5, label=r"$\Delta R_{bb}$")] ),
        "dibjet_dr_zoom" : HistConf( [Axis(field="dibjet_dr", bins=25, start=0, stop=5, label=r"$\Delta R_{bb}$")] ),

        "dibjet_deltaPhi": HistConf( [Axis(field="dibjet_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{bb}$")] ),
        "dibjet_deltaPhi_zoom": HistConf( [Axis(field="dibjet_deltaPhi", bins=25, start=0, stop=math.pi, label=r"$\Delta \phi_{bb}$")] ),

        "dibjet_deltaEta": HistConf( [Axis(field="dibjet_deltaEta", bins=50, start=0, stop=4, label=r"$\Delta \eta_{bb}$")] ),
        "dibjet_deltaEta_zoom": HistConf( [Axis(field="dibjet_deltaEta", bins=25, start=0, stop=4, label=r"$\Delta \eta_{bb}$")] ),

        "dibjet_pt_j1" : HistConf( [Axis(field="dibjet_pt_max", bins=100, start=0, stop=400, label=r"$p_T{b1}$ [GeV]")] ),
        "dibjet_pt_j1_zoom" : HistConf( [Axis(field="dibjet_pt_max", bins=25, start=0, stop=400, label=r"$p_T{b1}$ [GeV]")] ),

        "dibjet_pt_j2" : HistConf( [Axis(field="dibjet_pt_min", bins=100, start=0, stop=400, label=r"$p_T{b2}$ [GeV]")] ),
        "dibjet_pt_j2_zoom" : HistConf( [Axis(field="dibjet_pt_min", bins=25, start=0, stop=400, label=r"$p_T{b2}$ [GeV]")] ),

        "dibjet_mass_j1" : HistConf( [Axis(field="dibjet_mass_max", bins=100, start=0, stop=600, label=r"$M_{b1}$ [GeV]")] ),
        "dibjet_mass_j1_zoom" : HistConf( [Axis(field="dibjet_mass_max", bins=25, start=0, stop=600, label=r"$M_{b1}$ [GeV]")] ),

        "dibjet_mass_j2" : HistConf( [Axis(field="dibjet_mass_min", bins=100, start=0, stop=600, label=r"$M_{b2}$ [GeV]")] ),
        "dibjet_mass_j2_zoom" : HistConf( [Axis(field="dibjet_mass_min", bins=25, start=0, stop=600, label=r"$M_{b2}$ [GeV]")] ),

        
        "dibjet_BvsL_j1" : HistConf( [Axis(field="dibjet_BvsL_max", bins=24, start=0, stop=1, label=r"$BvsL_{bj1}$ [GeV]")] ),
        "dibjet_BvsL_j1_zoom" : HistConf( [Axis(field="dibjet_BvsL_max", bins=12, start=0, stop=1, label=r"$BvsL_{bj1}$ [GeV]")] ),

        "dibjet_BvsL_j2" : HistConf( [Axis(field="dibjet_BvsL_min", bins=24, start=0, stop=1, label=r"$BvsL_{bj2}$ [GeV]")] ),
        "dibjet_BvsL_j2_zoom" : HistConf( [Axis(field="dibjet_BvsL_min", bins=12, start=0, stop=1, label=r"$BvsL_{bj2}$ [GeV]")] ),

        "HT":  HistConf( [Axis(field="JetGood_Ht", bins=100, start=0, stop=700, label=r"Jet HT [GeV]")] ),
        "HT_zoom": HistConf( [Axis(field="JetGood_Ht", bins=50, start=0, stop=700, label=r"Jet HT [GeV]")] ),

        "met_pt": HistConf( [Axis(coll="MET", field="pt", bins=50, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),
        "met_pt_zoom": HistConf( [Axis(coll="MET", field="pt", bins=25, start=0, stop=200, label=r"MET $p_T$ [GeV]")] ),

        "met_phi": HistConf( [Axis(coll="MET", field="phi", bins=50, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),
        "met_phi_zoom": HistConf( [Axis(coll="MET", field="phi", bins=25, start=-math.pi, stop=math.pi, label=r"MET $phi$")] ),

        "lep_pt": HistConf( [Axis(field="lep_pt", bins=100, start=0, stop=400, label=r"$p_T{Lep}$ [GeV]")] ),
        "lep_pt_zoom": HistConf( [Axis(field="lep_pt", bins=50, start=0, stop=400, label=r"$p_T{Lep}$ [GeV]")] ),

        "lep_eta": HistConf( [Axis(field="lep_eta", bins=50, start=-2.5, stop=2.5, label=r"$\eta_{Lep}$")] ),
        "lep_eta_zoom": HistConf( [Axis(field="lep_eta", bins=25, start=-2.5, stop=2.5, label=r"$\eta_{Lep}$")] ),

        "lep_phi": HistConf( [Axis(field="lep_phi", bins=50, start=-math.pi, stop=math.pi, label=r"$\phi_{Lep}$")] ),
        "lep_phi_zoom": HistConf( [Axis(field="lep_phi", bins=25, start=-math.pi, stop=math.pi, label=r"$\phi_{Lep}$")] ),

        "deltaR_Leadb_Lep": HistConf( [Axis(field="deltaR_Leadb_Lep", bins=50, start=0, stop=5, label=r"$\Delta R_{b1，Lep}$")] ),
        "deltaR_Leadb_Lep_zoom": HistConf( [Axis(field="deltaR_Leadb_Lep", bins=25, start=0, stop=5, label=r"$\Delta R_{b1，Lep}$")] ),

        "deltaPhi_Leadb_Lep": HistConf( [Axis(field="deltaPhi_Leadb_Lep", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{b1，Lep}$")] ),
        "deltaPhi_Leadb_Lep_zoom": HistConf( [Axis(field="deltaPhi_Leadb_Lep", bins=25, start=0, stop=math.pi, label=r"$\Delta \phi_{b1，Lep}$")] ),

        "deltaEta_Leadb_Lep": HistConf( [Axis(field="deltaEta_Leadb_Lep", bins=50, start=0, stop=4, label=r"$\Delta \eta_{b1，Lep}$")] ),
        "deltaEta_Leadb_Lep_zoom": HistConf( [Axis(field="deltaEta_Leadb_Lep", bins=25, start=0, stop=4, label=r"$\Delta \eta_{b1，Lep}$")] ),

        "VHbb_pt_ratio": HistConf( [Axis(field="VHbb_pt_ratio", bins=50, start=0, stop=2, label=r"$pT_{H}/pT_{W}$")] ),
        "VHbb_pt_ratio_zoom": HistConf( [Axis(field="VHbb_pt_ratio", bins=25, start=0, stop=2, label=r"$pT_{H}/pT_{W}$")] ),

        "VHbb_deltaPhi": HistConf( [Axis(field="VHbb_deltaPhi", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{H，W}$")] ),
        "VHbb_deltaPhi_zoom": HistConf( [Axis(field="VHbb_deltaPhi", bins=25, start=0, stop=math.pi, label=r"$\Delta \phi_{H，W}$")] ),

        "VHbb_deltaR": HistConf( [Axis(field="VHbb_deltaR", bins=50, start=0, stop=5, label=r"$\Delta R_{H，W}$")] ),
        "VHbb_deltaR_zoom": HistConf( [Axis(field="VHbb_deltaR", bins=25, start=0, stop=5, label=r"$\Delta R_{H，W}$")] ),

        "deltaPhi_l1_j1": HistConf( [Axis(field="deltaPhi_l1_j1", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{L1，b1}$")] ),
        "deltaPhi_l1_j1_zoom": HistConf( [Axis(field="deltaPhi_l1_j1", bins=25, start=0, stop=math.pi, label=r"$\Delta \phi_{L1，b1}$")] ),

        "deltaPhi_l1_MET": HistConf( [Axis(field="deltaPhi_l1_MET", bins=50, start=0, stop=math.pi, label=r"$\Delta \phi_{L1，MET}$")] ),
        "deltaPhi_l1_MET_zoom": HistConf( [Axis(field="deltaPhi_l1_MET", bins=25, start=0, stop=math.pi, label=r"$\Delta \phi_{L1，MET}$")] ),

        "top_mass": HistConf( [Axis(field="top_mass", bins=50, start=0, stop=1000, label=r"$M_{top}$ [GeV]")] ),
        "top_mass_zoom": HistConf( [Axis(field="top_mass", bins=25, start=0, stop=1000, label=r"$M_{top}$ [GeV]")] ),


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
