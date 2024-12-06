import pandas as pd
import numpy as np
import os
from coffea.util import load
from glob import glob

import matplotlib.pyplot as plt
import boost_histogram as bh

def asimovSig(s, b, b_var):
    """
    Calculate the asimov significance
    """
    term1 = (s+b)*np.log(((s+b)*(b+b_var))/(b*b+(s+b)*b_var))
    term2 = (b*b/b_var)*np.log(1+(b_var*s)/(b*(b+b_var)))
    sig = np.sqrt(2*(term1-term2))
    sig = np.where(b!=0, sig, 0.0)
    return(sig)

def mergeBins(bins, fixedBins, n):
    """
    Merge bins with a coarseness of n, keeping fixedBins fixed

    Arguments:
        bins: original bins
        fixedBins: number of bins to not merge at the beginning
        n: number of bins to merge
    Returns:
        newBins: merged bins
    """
    flippedBins = np.flip(bins)
    newBins = []
    for x in range(fixedBins+1):
        newBins.append(flippedBins[x])
    for x in range(fixedBins+1, len(bins), n):
        newBins.append(flippedBins[x])
    if(newBins[-1]!=0):
        newBins.append(0)
    return(np.flip(newBins))

def mergeBinsEnd(bins, fixedBins, n):
    """
    Merge the first n bins with a coarseness of n, keeping fixedBins fixed

    Arguments:
        bins: original bins
        fixedBins: number of bins to not merge at the beginning
        n: number of bins to merge
    Returns:
        newBins: merged bins
    """
    flippedBins = np.flip(bins)
    newBins = []
    for x in range(fixedBins+1):
        newBins.append(flippedBins[x])
    for x in range(fixedBins+1, min(fixedBins+1+2*n,len(bins)), n):
        newBins.append(flippedBins[x])
    for x in range(fixedBins+1+2*n, len(bins), 1):
        newBins.append(flippedBins[x])
    if(newBins[-1]!=0):
        newBins.append(0)
    return(np.flip(newBins))
    

def makeBins(sigWeights, bkgWeights, sigMVAScores, bkgMVAScores, title,
             minBinSize=0.01, maxBinSize=10, binSizeSearchIncrement=0.01,
             sigCut=0.01, uncertCut=0.3, doPlot=False):
    
    """
        Arguments:
            sigWeights: array with the weights for all the signal events
            bkgWeights: array with the weights for all the background events
            sigMVAScores: array with the MVA scores for the signal events
            bkgMVAScores: array with the MVA scores for the background evnets
            title: name of region
            minBinSize: starting target size for the number of signal events 
                        in each bin, default is 0.01
            maxBinSize: maximum number of signal events in a bin prior to 
                        merging, default is 10
            binSizeSearchIncrement: how much to increment the bin size in each
                                    iteration, default is 0.01
            sigCut: amount of significance which can be lost with each merge,
                    default is 0.01
            uncertCut: maximum stat uncertainty per initial bin, default is 0.3
        Returns:
            sigSumFinal: final asimov significance with the binning
            bins: calculated bins
    """

    # determine the finest possible flat binning that keeps the background uncertainties low
    targetBinSize = minBinSize
    finalUncert = 10
    while(finalUncert>uncertCut and targetBinSize<maxBinSize):
        # sort the signal events by BDT score
        sortOrder = np.argsort(sigMVAScores)
        sortedWeights = sigWeights[sortOrder]
        sortedScores = sigMVAScores[sortOrder]

        # get the cdf of the signal events
        cdf = np.cumsum(sortedWeights)
        # determine where to make the bins with the target bin size
        binVals = np.arange(cdf[-1],0,-targetBinSize)
        binVals = np.flip(binVals)
        # determine where these bins occur in the cdf
        indices = np.searchsorted(cdf, binVals, side="left")
        if(indices[0]!=0):
            indices = np.concatenate([[0], indices])

        # determine the actual bin edges based on the granularity of the signal sample
        bins = []
        for index in indices:
            bins.append(sortedScores[index])
        bins[-1] = 1.0
        bins[0]=0.0
        # remove any duplicates in the bin edges
        bins = set(bins)
        bins = np.sort(np.array([binVal for binVal in bins]))
        
        # make a boost histogram using these bins
        bkgHist = bh.Histogram(bh.axis.Variable(bins),storage=bh.storage.Weight())
        bkgHist.fill(bkgMVAScores, weight=bkgWeights)

        # get the background uncertainty on the last bin
        finalUncert = np.max(np.sqrt(bkgHist.variances())/bkgHist.values())
        if(np.min(bkgHist.values())<=0):
            finalUncert = 100

        targetBinSize+=binSizeSearchIncrement

    # Settle on the last binSize
    targetBinSize-=binSizeSearchIncrement

    # display selected bin size
    print(title, "flat bin size:", targetBinSize)

    # make a signal histogram with the variable binning
    sigHist = bh.Histogram(bh.axis.Variable(bins),storage=bh.storage.Weight())
    sigHist.fill(sigMVAScores, weight=sigWeights)

    if(doPlot):
        # Make a plot of the flat binning
        plt.figure()
        plt.stairs(sigHist.values(),edges=range(len(sigHist.values())+1), label="Signal",linewidth=1.5, color="r")
        plt.stairs(bkgHist.values(),edges=range(len(bkgHist.values())+1), label="Background",linewidth=1.5, color="b")
        plt.errorbar(np.linspace(0.5, len(sigHist.values())-0.5,len(sigHist.values())), sigHist.values(), yerr=np.sqrt(sigHist.variances()), ls="none")
        plt.errorbar(np.linspace(0.5, len(bkgHist.values())-0.5,len(bkgHist.values())), bkgHist.values(), yerr=np.sqrt(bkgHist.variances()), ls="none")
        plt.yscale("log")
        plt.title(title+" sig bkg")
        plt.legend()
        plt.savefig("flatbinning.png")


    targetBin = 0
    requiredBins = 1
    while(targetBin<bins.size-1):
        # construct signal and background histograms with the current binning

        bkgHist = bh.Histogram(bh.axis.Variable(bins),storage=bh.storage.Weight())
        bkgHist.fill(bkgMVAScores, weight=bkgWeights)
        sigHist = bh.Histogram(bh.axis.Variable(bins),storage=bh.storage.Weight())
        sigHist.fill(sigMVAScores, weight=sigWeights)

        # determine the binwise asimov significance from these histograms
        s = sigHist.values()
        b = bkgHist.values()
        b_var = bkgHist.variances()
        sig = asimovSig(s, b, b_var)

        # add the significance of each bin in quadrature
        sigSum = [sig[-1]]
        for x in range(1, len(sig)):
            sigSum.append(np.sqrt(sigSum[-1]**2+sig[-1-x]**2))

        # Put total significance in sigSumNew
        sigSum = np.flip(sigSum)
        sigSumFinal = sigSum[0]
 
        sigSumNew = sigSumFinal

        selectedBins = np.copy(bins) # current bins we want
        
        # determine the number of initial bins to merge (always need at least as many as previous iteration)
        initialBins = requiredBins if targetBin+requiredBins<selectedBins.size-1 else selectedBins.size-1-targetBin
        mergeEnd = targetBin+initialBins # set the bin from which to end merging
        binsMerged = initialBins-1 # number of bins we have already merged
        # keep merging until we hit the end or there is too much significance drop
        while(mergeEnd<=selectedBins.size-1 and (sigSumNew>(1-sigCut)*sigSumFinal or binsMerged<requiredBins)):
            # Merge remaining bins with the current coarseness
            
            #newBins = np.copy(np.flip(np.concatenate([np.flip(selectedBins)[0:targetBin+1],np.flip(selectedBins)[mergeEnd+1:]])))#
            newBins = mergeBins(selectedBins, targetBin, binsMerged+1)
            
            # determine new histograms
            bkgHist = bh.Histogram(bh.axis.Variable(newBins),storage=bh.storage.Weight())
            bkgHist.fill(bkgMVAScores, weight=bkgWeights)
            sigHist = bh.Histogram(bh.axis.Variable(newBins),storage=bh.storage.Weight())
            sigHist.fill(sigMVAScores, weight=sigWeights)
            
            # get asimov significance
            s = sigHist.values()
            b = bkgHist.values()
            b_var = bkgHist.variances()
            
            sig = asimovSig(s, b, b_var)
            sigSum = [sig[-1]]
            for x in range(1, len(sig)):
                sigSum.append(np.sqrt(sigSum[-1]**2+sig[-1-x]**2))
            sigSum = np.flip(sigSum)
            sigSumNew = sigSum[0]
            
            if(sigSumNew>sigSumFinal): # new significance is higher
                sigSumFinal = np.copy(sigSumNew)
            
            if(sigSumNew>(1-sigCut)*sigSumFinal or binsMerged<requiredBins): # merge this bin
                # update bins with just the last bin merged to this coarseness
                #print(np.flip(selectedBins)[0:targetBin+1], np.flip(selectedBins)[mergeEnd:])
                #bins = np.copy(np.flip(np.concatenate([np.flip(selectedBins)[0:targetBin+1],np.flip(selectedBins)[mergeEnd+1:]])))#np.copy(newBins)
                bins = mergeBinsEnd(selectedBins, targetBin, binsMerged+1)
                binsMerged+=1
    
                
            mergeEnd+=1
        # update required bins and target bin
        requiredBins = binsMerged
        targetBin+=1
    
    # make final bkg and sig histograms
    bkgHist = bh.Histogram(bh.axis.Variable(bins),storage=bh.storage.Weight())
    bkgHist.fill(bkgMVAScores, weight=bkgWeights)
    sigHist = bh.Histogram(bh.axis.Variable(bins),storage=bh.storage.Weight())
    sigHist.fill(sigMVAScores, weight=sigWeights)
    
   
    # get asimov significance
    s = sigHist.values()
    b = bkgHist.values()
    b_var = bkgHist.variances()
    
    sig = asimovSig(s, b, b_var)
    sigSum = [sig[-1]]
    #print(sigSum)
    for x in range(1, len(sig)):
        sigSum.append(np.sqrt(sigSum[-1]**2+sig[-1-x]**2))
    sigSum = np.flip(sigSum)
    sigSumFinal = sigSum[0]
    
    # plot finall binning
    if(doPlot):
        plt.figure()
        plt.stairs(sigHist.values(),edges=range(len(sigHist.values())+1), label="Signal",linewidth=1.5, color="r")
        plt.stairs(bkgHist.values(),edges=range(len(bkgHist.values())+1), label="Background",linewidth=1.5, color="b")
        plt.errorbar(np.linspace(0.5, len(sigHist.values())-0.5,len(sigHist.values())), sigHist.values(), yerr=np.sqrt(sigHist.variances()), ls="none")
        plt.errorbar(np.linspace(0.5, len(bkgHist.values())-0.5,len(bkgHist.values())), bkgHist.values(), yerr=np.sqrt(bkgHist.variances()), ls="none")
        plt.yscale("log")
        plt.title(title+" sig bkg")
        plt.legend()
        plt.savefig("final1.png")
    
    
        # plot binwise asimo signifiance
        plt.figure()
        plt.plot(sig, label="Asimov")
        
        plt.title(title+" significance")
        plt.legend()
        plt.savefig("final2.png")
        
        
        # plot binwise background uncertainty
        plt.figure()
        plt.stairs(np.sqrt(b_var)/b,edges=range(len(sig)+1), label="uncert",linewidth=1.5, color="r")
        plt.plot([0, len(sig)],[uncertCut,uncertCut], linewidth=1, color="orange")
        
        plt.title(title+" bkg uncert")
        
        plt.legend()
        plt.savefig("final3.png")
    
    return(sigSumFinal, bins)


def findFiles(directory):
    """
    Recursively find parquet files.    
    
    Arguments:
        directory: directory to search inside of
    Returns:
        files: list of files
    """
    print(directory)
    files = glob(f'{directory}/**/*.parquet',recursive=True)
    files = [f for f in files if "DiJet_ll" not in f]

    return files 

def getScale(genWeights, file):
    """
    Find normalization value from coffea dictionary    
    
    Arguments:
        genWeights: coffea weight dictionary
        file: file to check
    Returns:
        scale: scale factor
    """
    scale = 1.0
    for sample in genWeights.keys():
        if(sample in file):
            # deal with ggZH/ZH samples 
            matches = 0
            if("ggZH" in sample):
                matches+=1
            if("ggZH" in file):
                matches+=1
            if(matches==1):
                continue
            
            if(scale!=1.0):
                print("double", file)
            scale = genWeights[sample]
            if(scale==1.0):
                print("zero", file)
    return(scale)

def doBin(coffeafile, coffeaWeights, topDir, signalProccesses, yearList, 
          channelList, name, minBinSize=0.01, maxBinSize=10,
          binSizeSearchIncrement=0.002, sigCut=0.002, uncertCut=0.3, 
          doPlot=False):
    """
    Perform the binning.
    

    Arguments:
        coffeafile: location of coffea file with normalizations
        coffeaWeights: name of branch in file with normalizations
        topDir: directory name containing the parquet files
        signalProccesses: list of processes used for signal
        yearList: list of years to be considered
        channelList: list of channels to be considered
        name: name of file to use
        minBinSize: starting target size for the number of signal events 
                    in each bin, default is 0.01
        maxBinSize: maximum number of signal events in a bin prior to 
                    merging, default is 10
        binSizeSearchIncrement: how much to increment the bin size in each
                                iteration, default is 0.01
        sigCut: amount of significance which can be lost with each merge,
                default is 0.01
        uncertCut: maximum stat uncertainty per initial bin, default is 0.3
        doPlot: whether or not to show plots, default False
    Returns: 
        sigSum: asimov significance with binning
        bins: derived binning
    """
    # get the normalization dictionary
    coffeafile = load(coffeafile)
    genWeights = coffeafile[coffeaWeights]

    subdirs = glob(topDir+"/*")
    sig_Gnn = []
    sig_weights = []
    bkg_Gnn = []
    bkg_weights = []

    for subd in subdirs:
        if "DATA_" in subd: continue

        passYear = False
        for year in yearList:
            if(year in subd):
                passYear = True
        if(not passYear):
            continue

        filelist = []
        for channel in channelList:
            filelist.extend(glob(f"{subd}/**/{channel}/*.parquet",recursive=True))

        filelist = [f for f in filelist if "DiJet_incl" not in f]
        if len(filelist) == 0: continue

        df = pd.read_parquet(filelist,columns=["weight","events_GNN"])
        sampname = subd.split('/')[-1]
        scale = genWeights[sampname]
        print("Done reading",sampname)

        isSig = False
        for sigProc in signalProccesses:
            if(sigProc in subd):
                isSig = True

        if(isSig):
            sig_Gnn.append(df["events_GNN"])
            sig_weights.append(df["weight"]/scale)
            print(f"\t{sampname} labelled as signal.")
        else:
            bkg_Gnn.append(df["events_GNN"])
            bkg_weights.append(df["weight"]/scale)
 
    sig_Gnn = np.concatenate(sig_Gnn)
    sig_weights = np.concatenate(sig_weights)
    bkg_Gnn = np.concatenate(bkg_Gnn)
    bkg_weights = np.concatenate(bkg_weights)
    
    # Run the algorithm
    sigSum, bins = makeBins(sig_weights, bkg_weights, sig_Gnn, bkg_Gnn, name,
                            minBinSize, maxBinSize, binSizeSearchIncrement,
                            sigCut, uncertCut, doPlot=True)
    return(sigSum, bins)
        
def main():
    coffeafile = "output_all.coffea"
    coffeaWeights = "sum_signOf_genweights"
    topdir = [d for d in glob('./*') if "Saved_columnar_arrays_" in d][0]
    print("Top directory:",topdir)

    if "ZLL" in topdir:
        channels = [["SR_ll_2J_cJ"],["SR_ll_2J_cJ"]] 
        names = ["2l_2022_preEE","2l_2022_postEE"]
    if "WLNu" in topdir:
        channels = [["SR_Wln_2J_cJ"]] 
        names = ["0l_2022_preEE"]
    if "ZNuNu" in topdir:
        channels = [["SR_Znn_2J_cJ"]] 
        names = ["0l_2022_postEE"]


    signalProccesses = ["Hto2C"]
    years = [["2022_preEE"]]
    
    for year in years:
        for name, channel in zip(names, channels):
            print(name, channel)
            sig, bins = doBin(coffeafile, coffeaWeights, topdir, signalProccesses, year, channel, name)
            print(bins)
if(__name__=="__main__"):
    main()