import awkward as ak
from pocket_coffea.lib.cut_definition import Cut

def diLepton(events, params, year, sample, **kwargs):

    # Masks for same-flavor (SF) and opposite-sign (OS)
    EE = ((events.nMuonGood == 0) & (events.nElectronGood == 2))
    MuMu = ((events.nMuonGood == 2) & (events.nElectronGood == 0))

    OS = events.ll.charge == 0

    mask = (
        (MuMu | EE) & OS
        & (ak.firsts(events.LeptonGood.pt) > params["pt_leading_lep"])
        & (events.ll.mass > params["mll"]["low"])
        & (events.ll.mass < params["mll"]["high"])
        & (events.ll.pt > params["pt_dilep"])
    )
    
    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

def TwoMuons(events, **kwargs):
    mask = (events.nMuonGood >= 2)
    return ak.where(ak.is_none(mask), False, mask)

def TwoElectrons(events, **kwargs):
    mask = (events.nElectronGood >= 2)
    return ak.where(ak.is_none(mask), False, mask)

def NJets(events, params, **kwargs):
    mask = (events.nJetGood >= params['nj'])
    return ak.where(ak.is_none(mask), False, mask)

def TwoLepTwoJets(events, params, **kwargs):
    if params["lep_flav"] not in ['mu','el','both']:
        print("This lepton flavor is not supported:", params["lep_flav"])
        raise Exception("The lepton flavor is not supported")
    
    mask = ( (events.nJetGood >= 2)
             & ( ( (params["lep_flav"]=="mu") & (events.nMuonGood>=2) ) |
                 ( (params["lep_flav"]=="el") & (events.nElectronGood>=2) )  |
                 ( (params["lep_flav"]=="both") & (events.nLeptonGood>=2) )
                )
             & (events.ll.pt > params["pt_dilep"])
             & (events.ll.mass > params["mll"]["low"])
             & (events.ll.mass < params["mll"]["high"])
            )
    return ak.where(ak.is_none(mask), False, mask)

def AntiZFourJets(events, params, **kwargs):
    # This mask is used for TTbar CR
    if params["lep_flav"] not in ['mu','el','both']:
        print("This lepton flavor is not supported:", params["lep_flav"])
        raise Exception("The lepton flavor is not supported")
    mask = ( (events.nJetGood >= 4)
             & ( ( (params["lep_flav"]=="mu") & (events.nMuonGood>=2) ) |
                 ( (params["lep_flav"]=="el") & (events.nElectronGood>=2) )  |
                 ( (params["lep_flav"]=="both") & (events.nLeptonGood>=2) )
                )
             & (events.ll.pt > params["pt_dilep"])
             & ( (events.ll.mass < params["mll"]["low"]) | (events.ll.mass > params["mll"]["high"]) )
             & (events.ll.mass > 50) 
            )
    return ak.where(ak.is_none(mask), False, mask)

def OneLeptonPlusMet(events, params, **kwargs):
    #mask = (events.nLeptonGood == 1 )
    mask = ( (events.nLeptonGood == 1 )
             & (ak.firsts(events.LeptonGood.pt) > params["pt_lep"])
             & (events.PuppiMET.pt > params["pt_met"])
            )
    return ak.where(ak.is_none(mask), False, mask)

def LepMetTwoJets(events, params, **kwargs):
    mask = ( (events.nLeptonGood == 1 )
             & (ak.firsts(events.LeptonGood.pt) > params["pt_lep"])
             & (events.PuppiMET.pt > params["pt_met"])
             & (events.nJetGood >= 2)
            )
    return ak.where(ak.is_none(mask), False, mask)

def MetTwoJetsNoLep(events, params, **kwargs):    
    mask = ( (events.nLeptonGood == 0 )
             & (events.PuppiMET.pt > params["pt_met"])
             & (events.nJetGood >= 2)
             & (ak.firsts(events.JetGood.pt) > params["pt_jet1"])
             & (ak.pad_none(events.JetGood.pt, 2, axis=1)[:,1] > params["pt_jet2"])
            )
    return ak.where(ak.is_none(mask), False, mask)

def WLNuTwoJets(events, params, **kwargs):
    if params["lep_flav"] not in ['mu','el','both']:
        print("This lepton flavor is not supported:", params["lep_flav"])
        raise Exception("The lepton flavor is not supported")

    fields = {
        "pt": events.PuppiMET.pt,
        "eta": ak.zeros_like(events.PuppiMET.pt),
        "phi": events.PuppiMET.phi,
        "mass": ak.zeros_like(events.PuppiMET.pt),
        "charge": ak.zeros_like(events.PuppiMET.pt),
    }

    METs = ak.zip(fields, with_name="PtEtaPhiMCandidate")
    LepPlusMet = METs + ak.firsts(events.LeptonGood)
    mask = ( (events.nJetGood >= 2)
             & ( ( (params["lep_flav"]=="mu") & (events.nMuonGood==1) ) |
                 ( (params["lep_flav"]=="el") & (events.nElectronGood==1) )  |
                 (params["lep_flav"]=="both") 
                )
             & (LepPlusMet.pt > params["pt_w"])
        )
    return ak.where(ak.is_none(mask), False, mask)

def jettag(events, params, **kwargs):
    if params['tagger'] == "PNet":
        CvL = "btagPNetCvL"
        CvB = "btagPNetCvB"
    elif params['tagger'] == "DeepFlav":
        CvL = "btagDeepFlavCvL"
        CvB = "btagDeepFlavCvB"
    elif params['tagger'] == "RobustParT":
        CvL = "btagRobustParTAK4CvL"
        CvB = "btagRobustParTAK4CvB"
    else:
        raise NotImplementedError(f"This tagger is not implemented: {params['tagger']}")

    #print(events.JetsCvsL.[ctag][:, 0]>0.2)

    if params["ctag"]:
        mask_CvL = (events.JetsCvsL[CvL][:,0]>params["cut_CvL"])
    else:
        mask_CvL = (events.JetsCvsL[CvL][:,0]<params["cut_CvL"])

    if params["btag"]:
        mask_CvB = (events.JetsCvsL[CvB][:,0]<params["cut_CvB"])
    else:
        mask_CvB = (events.JetsCvsL[CvB][:,0]>params["cut_CvB"])

    mask = mask_CvL & mask_CvB
    
    return ak.where(ak.is_none(mask), False, mask)

def bjettag(events, params, **kwargs):
    if params['tagger'] == "PNet":
        BvL = "btagPNetB"
    elif params['tagger'] == "DeepFlav":
        BvL = "btagDeepFlavB"
    elif params['tagger'] == "RobustParT":
        BvL = "btagRobustParTAK4B"
    else:
        raise NotImplementedError(f"This tagger is not implemented: {params['tagger']}")
        
    tightness = params["btag_cut"]
    _btag_cut = events[f"btag_cut_{tightness}"]
    
    if params["inv"]:
      mask_BvL = (events.JetsBvsL[BvL][:, params["nJ"]]<_btag_cut)
    else:
      mask_BvL = (events.JetsBvsL[BvL][:, params["nJ"]]>_btag_cut)
      
    mask = mask_BvL
    
    return ak.where(ak.is_none(mask), False, mask)
  
def METpTCut(events, params, **kwargs):
    if params["invert"]:
        mask = (events.pt_miss < params["pt_met"])
    else:
        mask = (events.pt_miss > params["pt_met"])
      
    return ak.where(ak.is_none(mask), False, mask)

def DiJetPtCut(events, params, **kwargs):
    mask = (  (events.nJetGood >= 2)
              & (events.dijet.pt > params["pt_dijet"])
              #& (events.dijet_csort.pt > params["pt_dijet"])
            )
    return ak.where(ak.is_none(mask), False, mask)

def DiBJetPtCut(events, params, **kwargs):
    if params["dijet"]:
        mask = (  (events.nJetGood >= 2)
                & (events.dijet_bsort.pt > params["pt_dijet"])
              )
    else:
        mask = (  (events.nJetGood >= 2)
                & (events.dijet_bsort.j1pt > params["pt_b1"])
                & (events.dijet_bsort.j2pt > params["pt_b2"])
              )
            
    return ak.where(ak.is_none(mask), False, mask)

def DiJetMassCut(events, params, **kwargs):

    if params["invert"]:
        mask = (  (events.nJetGood >= 2)          
                  & ( (events.dijet_csort.mass < params["mjj"]["low"])
                      | (events.dijet_csort.mass > params["mjj"]["high"]) )
                )        
    else:
        mask = (  (events.nJetGood >= 2)          
                  & (events.dijet_csort.mass > params["mjj"]["low"])
                  & (events.dijet_csort.mass < params["mjj"]["high"])
                )
    return ak.where(ak.is_none(mask), False, mask)

def DiBJetMassCut(events, params, **kwargs):

    if params["invert"]:
        mask = (  (events.nJetGood >= 2)          
                  & ( (events.dijet_bsort.mass < params["mjj"]["low"])
                      | (events.dijet_bsort.mass > params["mjj"]["high"]) )
                )        
    else:
        mask = (  (events.nJetGood >= 2)          
                  & (events.dijet_bsort.mass > params["mjj"]["low"])
                  & (events.dijet_bsort.mass < params["mjj"]["high"])
                )
    return ak.where(ak.is_none(mask), False, mask)

def BJetMassCut(events, params, **kwargs):
    mask = (  (events.nJetGood >= 2)          
            & (events.dijet_bsort.j1mass >= params["mass_b1_min"])
            & (events.dijet_bsort.j1mass <= params["mass_b1_max"])
            & (events.dijet_bsort.j2mass >= params["mass_b2_min"])
            & (events.dijet_bsort.j2mass <= params["mass_b2_max"])
    )
    return ak.where(ak.is_none(mask), False, mask)

def HbbPtCut(events, params, **kwargs):
    
    mask = ( (events.nJetGood >= 2)
           & (events.dijet_bsort.leadb_pt > params["b1_pt"])
           & (events.dijet_bsort.subleadb_pt > params["b2_pt"])
    )
    return ak.where(ak.is_none(mask), False, mask)

def DiBJetDeltaEtaCut(events, params, **kwargs):
    mask = ( (events.nJetGood >= 2) & (events.dijet_bsort.deltaEta < params["bb_deta"]) )
    return ak.where(ak.is_none(mask), False, mask)

def DeltaPhiVHCut(events, params, **kwargs):
    mask = ( events.VHbb_deltaPhi > params["VHdPhi"] )
    return ak.where(ak.is_none(mask), False, mask)

def DeltaRVHCut(events, params, **kwargs):
    mask = ( events.VHbb_deltaR < params["VHdR"] )
    return ak.where(ak.is_none(mask), False, mask)

def DeltaEtaVHCut(events, params, **kwargs):
    mask = ( events.VHbb_deltaEta < params["VHdEta"] )
    return ak.where(ak.is_none(mask), False, mask)

def HVpTRatioCut(events, params, **kwargs):
    mask = ( (events.VHbb_pt_ratio >= params["HVpTRatio_min"]) 
         & (events.VHbb_pt_ratio <= params["HVpTRatio_max"])
         )
    return ak.where(ak.is_none(mask), False, mask)

def LepMetDeltaPhi(events, params, **kwargs):
    mask = ( events.deltaPhi_l1_MET <= params["lepmetdphi"] )
    return ak.where(ak.is_none(mask), False, mask)

def AddLepCut(events, params, **kwargs):
    mask = ( events.NaL == params["add_lep"] )
    return ak.where(ak.is_none(mask), False, mask)

def AddJetCut(events, params, **kwargs):
    if params["equal"]: mask = ( events.NaJ == params["add_jet"] )
    else: mask = ( events.NaJ <= params["add_jet"] )
    return ak.where(ak.is_none(mask), False, mask)

def DiLeptonMassCut(events, params, **kwargs):
    if params["invert"]:
        mask = (  (events.ll.mass < params["mll"]["low"]) | (events.ll.mass > params["mll"]["high"]) )
        return ak.where(ak.is_none(mask), False, mask)
    else:
        mask = (  (events.ll.mass >= params["mll"]["low"]) & (events.ll.mass <= params["mll"]["high"]) )
        return ak.where(ak.is_none(mask), False, mask)
    
def DiLeptonPtCut(events, params, **kwargs):
    mask = (  (events.ll.pt > params["ptll"]["low"]) & (events.ll.pt < params["ptll"]["high"]) )
    return ak.where(ak.is_none(mask), False, mask)


def DeltaPhiJetMetCut(events, params, **kwargs):
    mask = ( (events['deltaPhi_jet1_MET'] > params["jet_met_dphi_cut"])
             & (events['deltaPhi_jet2_MET'] > params["jet_met_dphi_cut"])
            )
    return ak.where(ak.is_none(mask), False, mask)


def TrueJetFlavors(events, params, **kwargs):
    gen_jets = events.GenJet
    
    cGenJetTot = ak.sum((gen_jets.hadronFlavour == 4) & (gen_jets.pt > 20) & (abs(gen_jets.eta) < 2.4), axis=1)
    bGenJetTot = ak.sum((gen_jets.hadronFlavour == 5) & (gen_jets.pt > 20) & (abs(gen_jets.eta) < 2.4), axis=1)

    tag_cc = (cGenJetTot >= 2)
    tag_bb = (bGenJetTot >= 2)
    tag_bc = (bGenJetTot == 1) & (cGenJetTot == 1)
    tag_cl = (cGenJetTot == 1) & (bGenJetTot == 0)
    tag_bl = (bGenJetTot == 1) & (cGenJetTot == 0)
    tag_ll = (cGenJetTot == 0) & (bGenJetTot == 0)

    mask = (  ((params['jj_flav']=='cc') & tag_cc) |
              ((params['jj_flav']=='bb') & tag_bb) |
              ((params['jj_flav']=='bc') & tag_bc) |
              ((params['jj_flav']=='cl') & tag_cl) |
              ((params['jj_flav']=='bl') & tag_bl) |
              ((params['jj_flav']=='bx') & (tag_bb | tag_bc | tag_bl)) |
              ((params['jj_flav']=='cx') & (tag_cc | tag_cl)) |
              ((params['jj_flav']=='ll') & tag_ll)
            )
    
    #mask = ( (params['jj_flav']=='cc') & tag_cc)
    """
    sampleFlavSplit = 1 * tag_cc  +  2 * tag_bb  +  3 * tag_bc  +  4 * tag_cl  +  5 * tag_bl  +  6 * tag_ll 

    mask = (  (params['jj_flav']=='cc' & (sampleFlavSplit==1)) |
              (params['jj_flav']=='bb' & (sampleFlavSplit==2)) |
              (params['jj_flav']=='bc' & (sampleFlavSplit==3)) |
              (params['jj_flav']=='cl' & (sampleFlavSplit==4)) |
              (params['jj_flav']=='bl' & (sampleFlavSplit==5)) |
              (params['jj_flav']=='udsg' & (sampleFlavSplit==6)) 
            )
    """
    
    return ak.where(ak.is_none(mask), False, mask)

# General cuts


DiJet_bx = Cut(
    name="DiJet_bx",
    function=TrueJetFlavors,
    params={"jj_flav": "bx"}
)
DiJet_cx = Cut(
    name="DiJet_bcx",
    function=TrueJetFlavors,
    params={"jj_flav": "cx"}
)
DiJet_ll = Cut(
    name="DiJet_ll",
    function=TrueJetFlavors,
    params={"jj_flav": "ll"}
)

one_jet = Cut(
    name="one_jet",
    function=NJets,
    params={'nj': 1}
)
two_jets = Cut(
    name="two_jets",
    function=NJets,
    params={'nj': 2}
)
four_jets = Cut(
    name="four_jets",
    function=NJets,
    params={'nj': 4}
)

ctag_j1 = Cut(
    name="ctag_j1",
    function=jettag,
    params={
        "tagger": 'RobustParT',
        "ctag": True,
        "btag": False,
        "cut_CvL": 0.2,
        "cut_CvB": 0.4
    }
)
antictag_j1 = Cut(
    name="antictag_j1",
    function=jettag,
    params={
        "tagger": 'RobustParT',
        "ctag": False,
        "btag": False,
        "cut_CvL": 0.2,
        "cut_CvB": 0.4
    }
)
btag_j1 = Cut(
    name="btag_j1",
    function=jettag,
    params={
        "tagger": 'RobustParT',
        "ctag": True,
        "btag": True,
        "cut_CvL": 0.2,
        "cut_CvB": 0.4
    }
)

def bjet_tagger(tagger='DeepFlav', nJ = 0, btag_cut = 'M', invert = False):
    return Cut(
    name = "bjet_tagger",
    function = bjettag,
    params = {
      "tagger": tagger,
      "nJ": nJ,
      "btag_cut": btag_cut,
      "inv": invert
    }
    )

dijet_pt_cut = Cut(
    name="dijet_pt_cut",
    function=DiJetPtCut,
    params={
	"pt_dijet": 120,
    },
)

dijet_mass_cut = Cut(
    name="dijet_mass_cut",
    function=DiJetMassCut,
    params={
        "invert": False,
        "mjj": {'low': 70, 'high': 250}
    },
)

dijet_invmass_cut = Cut(
    name="dijet_invmass_cut",
    function=DiJetMassCut,
    params={
        "invert": True,
	"mjj": {'low': 70, 'high': 250}
    },
)

jet_met_dphi_cut = Cut(
    name='jet_met_dphi_cut',
    function=DeltaPhiJetMetCut,
    params={
	"jet_met_dphi_cut": 0.6,
    },
)
# Cuts for 0-Lep channel

met_2jets_0lep = Cut(
    name="met_2jets_0lep",
    function=MetTwoJetsNoLep,
    params={
        "pt_met": 170,
        "pt_jet1": 60,
        "pt_jet2": 35,
    },
)

def ZNNHBB_2J():
    return Cut(
        name = "ZNNHBB_2J",
        function=MetTwoJetsNoLep,
        params={
            "pt_met": 180,
            "pt_jet1": 0,
            "pt_jet2": 0,
        },
    )
def bJ_pt_cut(b1_pt = 60, b2_pt = 35):
    return Cut(
        name="bJ_pt_cut",
        function=HbbPtCut,
        params={
            "b1_pt": b1_pt,
            "b2_pt": b2_pt,
        },
    )

import numpy as np

def min_dPhi_bJ_MET(events, params, **kwargs):
    mask = ( np.minimum(events.deltaPhi_jet1_MET, events.deltaPhi_jet2_MET) <= params["max_min_dPhi"] )
    return ak.where(ak.is_none(mask), False, mask)
    
min_dPhi_bJ_MET_1p57 = Cut(
    name = "min_dPhi_bJ_MET_1p57",
    function=min_dPhi_bJ_MET,
    params={
        "max_min_dPhi": 1.57,
    },
)

# Cuts for 1-Lep channel

onelep_plus_met = Cut(
    name="onelep_plus_met",
    function=OneLeptonPlusMet,
    params={
        "pt_lep": 33,
        "pt_met": 10,
    },
)

lep_met_2jets = Cut(
    name="lep_met_2jets",
    function=LepMetTwoJets,
    params={
        "pt_lep": 33,
        "pt_met": 10,
    },
)

def wlnu_plus_2j(lep_flav='both'):
    return Cut(
        name="w_plus_2j_lepflav_"+lep_flav,
        function=WLNuTwoJets,
        params={
            "lep_flav": lep_flav,
            "pt_w": 100
        },
    )

def WLNuHBB_2J(lep_flav='both'):
    return Cut(
    name = "WLNuTwoJets_"+lep_flav,
    function=WLNuTwoJets,
    params={
            "lep_flav": lep_flav,
            "pt_w": 0
        },
    )

def LepMetDPhi(dphi = 2):
    return Cut(
        name = 'LepMetDPhi',
        function=LepMetDeltaPhi,
        params={
            "lepmetdphi": dphi
        }, 
    )

# Cuts for 2-Lep channel

dilepton = Cut(
    name="dilepton",
    function=diLepton,
    params={
        "pt_leading_lep": 27,
        "mll": {'low': 81, 'high': 101},
        "pt_dilep": 15
    },
)

mumu_channel = Cut(
    name = 'mumu',
    function=TwoMuons,
    params=None
)
ee_channel = Cut(
    name = 'ee',
    function=TwoElectrons,
    params=None
)


def ll_2j(lep_flav='both'):
    return Cut(
        name = 'll_2j_lepflav_'+lep_flav,
        function=TwoLepTwoJets,
        params={"lep_flav": lep_flav,
                "pt_dilep": 60,
                "mll": {'low': 50, 'high': 400}
                }
    )

def Zll_2j(lep_flav='both'):
    return  Cut(
        name = 'Zll_2j_lepflav_'+lep_flav,
        function=TwoLepTwoJets,
        params={"lep_flav": lep_flav,
                "pt_dilep": 60,
                "mll": {'low': 75, 'high': 115}
                }
    )

def ZLLHBB_2J(lep_flav='both'):
    return Cut(
    name = 'ZLLHBB_2J',
    function=TwoLepTwoJets,
    params={"lep_flav": lep_flav,
            "pt_dilep": 0,
            "mll": {'low': 10, 'high': 450}
            }
    )

def VH_dPhi_cut(VHdPhi = 2.5):
    return Cut(
        name = 'VH_dPhi_cut',
        function=DeltaPhiVHCut,
        params={
          "VHdPhi": VHdPhi,
        }
    )

VH_dPhi_cut_2p5 = Cut(
  name = 'VH_dPhi_cut_2p5',
  function=DeltaPhiVHCut,
    params={
      "VHdPhi": 2.5,
           }
)

VH_dR_cut_3p6 = Cut(
  name = 'VH_dR_cut_3p6',
  function=DeltaRVHCut,
    params={
      "VHdR": 3.6,
           }
)

VH_dEta_cut_2 = Cut(
  name = 'VH_dEta_cut_2',
  function=DeltaEtaVHCut,
    params={
      "VHdEta": 2,
           }
)


HV_pTRatio_cut_0p5to2 = Cut(
  name = 'HV_pTRatio_cut_0p5to2',
  function=HVpTRatioCut,
    params={
      "HVpTRatio_min": 0.5,
      "HVpTRatio_max": 2,
           }
)

def dijet_mass(mass_min=0, mass_max=2000, invert=False):
    return Cut(
        name=f"dijet_mass_cut",
        function=DiBJetMassCut,
        params={
            "invert": invert,
            "mjj": {'low': mass_min, 'high': mass_max}
        }
    )

def dibjet_pt_cut(dijet=True, pt_dijet=50, pt_b1=0, pt_b2=0):
    return Cut(
      name="dijet_pt_cut_50",
      function=DiBJetPtCut,
      params={
        "dijet": True,
        "pt_dijet": 50,
        "pt_b1": 0,
        "pt_b2": 0,
      }
    )

bJ_mass_cut_5to30_5to30 = Cut(
    name="bJ_mass_cut_5to30_5to30",
    function=BJetMassCut,
    params={
      "mass_b1_min": 5,
      "mass_b2_min": 5,
      "mass_b1_max": 30,
      "mass_b2_max": 30,
    },
)

def dibjet_eta_cut(bb_deta = 1.0):
    return Cut(
        name="dibjet_eta_cut",
        function=DiBJetDeltaEtaCut,
        params={
            "bb_deta": bb_deta,
        },
    )
                                   
nAddLep_cut_0 = Cut(
    name="nAddLep_cut_0",
    function=AddLepCut,
    params={
      "add_lep": 0,
    },
) 

def nAddJetCut(add_jet=1, equal=False):
    return Cut(
        name="nAddJet_cut",
        function=AddJetCut,
        params={
          "equal": equal,
          "add_jet": add_jet,
        },
    ) 

def missing_pt_cut(invert=False, pt_met=60):
    return Cut(
        name="missing_pt_cut",
        function=METpTCut,
        params={
          "invert": invert,
          "pt_met": pt_met,
        },
    )

def dilep_mass_window(invert=False, mll_low=75, mll_high=400):
    return Cut(
        name = 'dilep_mass_window',
        function=DiLeptonMassCut,
          params={
            "invert": invert,
            "mll": {'low': mll_low, 'high': mll_high}
          }
    )

def dilep_pt(pt_min=60, pt_max=2000):
    return Cut(
        name="dilep_pt_cut_Pt%iTo%i"%(pt_min,pt_max),
        function=DiLeptonPtCut,
        params={
	    "ptll": {'low': pt_min, 'high': pt_max}
        },
    )

# Cuts for ttbar control region
def ll_antiZ_4j(lep_flav='both'):
    return  Cut(
        name = 'll_antiZ_4j_lepflav_'+lep_flav,
        function=AntiZFourJets,
        params={"lep_flav": lep_flav,
                "pt_dilep": 60,
                "mll": {'low': 75, 'high': 120}
                }
    )


