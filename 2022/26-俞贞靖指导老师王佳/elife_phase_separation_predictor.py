#!/usr/bin/python

import string
from sys import argv,stdout,version_info
from os import popen,system
from os.path import exists,basename
from math import log, sqrt
from numpy import cross, array, dot, vdot, arccos
import numpy as np
from random import shuffle
import random
import copy
import pickle

def Help():
    print( "USAGE:" )
    print("")
    print( "   ./elife_phase_separation_predictor.py [fasta file]" )
    print("")
    print( "   command line options" )
    print( "         -residue_scores")
    print( "            (writes out weighted per-residue scores)")
    print( "         -score_components")
    print( "            (writes out the eight unweighted per-residue score components)")
    print( "         -output [filename]")
    print( "            (writes to specified file instead of to the screen)" )
    print( "         -overwrite")
    print( "            (output will overwrite an existing file)" )
    print( "         -mute")
    print( "            (output [filename] mode will be silent)" )
    print("")
    if version_info < (3,5,0):
        version = str(version_info[0])+"."+str(version_info[1])
        print( "Note: You are running this with Python version "+version )
        print( "" )
        print( "      Loading the database takes 5x longer with any version under 3.5" )
        print( "      " )
        print( "      (the difference is only ~30 seconds vs. ~6 seconds)")
        print( "" )
    exit()

def CError():
    print("")
    print("####################ERROR########################")
    print("#          invalid command line flag            #")
    print("# only the fasta file should be in this list... #")
    print("#################################################")
    print("")
    for i in range(len(argv)-1):
        print("---> "+argv[i+1])
    exit()

def AError():
    print("")
    print("##########ERROR##############")
    print("#      -output [file]       #")
    print("# file either not specified #")
    print("#    or would not open      #")
    print("#############################")
    print("")
    exit()

def OError():
    print("")
    print("####################ERROR########################")
    print("#                                               #")
    print("#          output file already exists           #")
    print("#        (use '-overwrite' to replace it)       #")
    print("#                                               #")
    print("#################################################")
    print("")
    exit()

def DBError():
    print("")
    print("####################ERROR########################")
    print("#                                               #")
    print("#     script must be run from same location     #")
    print("#          as the database directory            #")
    print("#                                               #")
    print("#   [ or you can change the DBPATH variable ]   #")
    print("#     -->       DBPATH = './DBS/'       <--     #")
    print("#                                               #")
    print("#################################################")
    print("")
    exit()

## You can change the default database path here
DBPATH = './DBS/'

if exists(DBPATH) == False:
    DBError()

###############################################################################
#                                                                             #
#                             Command Line Options                            #
#                                                                             #
###############################################################################

## Change any of these to true if you want it to be the default behavior
cRSCORE = False       # -residue_scores
cCOMPONENTS = False   # -score_components
cOVERWRITE = False    # -overwrite (outfiles overwrite existing files)
cMUTE = False         # makes "-output [filename]" mode silent

if argv.count('-residue_scores'):
    pos = argv.index('-residue_scores')
    del( argv[pos] )
    cRSCORE = True


if argv.count('-score_components'):
    pos = argv.index('-score_components')
    del( argv[pos] )
    cRSCORE = True
    cCOMPONENTS = True

if argv.count('-overwrite'):
    pos = argv.index('-overwrite')
    del( argv[pos] )
    cOVERWRITE = True

if argv.count('-mute'):
    pos = argv.index('-mute')
    del( argv[pos] )
    cMUTE = True
    
cOUTFILE = False
if argv.count('-output'):
    pos = argv.index('-output')
    del( argv[pos] )

    try:
        if exists(argv[pos]) and cOVERWRITE == False: OError()
        OFILE = open(argv[pos], 'w')
        del( argv[pos] )
    except:
        OError()

    cOUTFILE = True

if argv.count('-outfile'):
    pos = argv.index('-outfile')
    del( argv[pos] )

    try:
        if exists(argv[pos]) and cOVERWRITE == False: OError()
        OFILE = open(argv[pos], 'w')
        del( argv[pos] )
    except:
        OError()

    cOUTFILE = True

if len(argv) > 2:
    CError()
if len(argv) < 2:
    Help()

if cOUTFILE and (cMUTE == False):
    print("Loading Database...")

###############################################################################
#                                                                             #
#                              Global Constants                               #
#                           ( please do not change )                          #
#                                                                             #
###############################################################################

max_xmer = 40
min_xmer = 1

###############################################################################
#                                                                             #
#                      Utility Dictionaries and Functions                     #
#                                                                             #
###############################################################################

## Heavy Atoms Dictionary
#  Carbon, Nitrogen, and Oxygen atom counts by residue type
##
HATOMS = {}
HATOMS["A"] = [3 , 1, 1 ]
HATOMS["C"] = [3 , 1, 1 ]
HATOMS["E"] = [5 , 1, 3 ]
HATOMS["D"] = [4 , 1, 3 ]
HATOMS["G"] = [2 , 1, 1 ]
HATOMS["F"] = [9 , 1, 1 ]
HATOMS["I"] = [6 , 1, 1 ]
HATOMS["H"] = [6 , 3, 1 ]
HATOMS["K"] = [6 , 2, 1 ]
HATOMS["M"] = [5 , 1, 1 ]
HATOMS["L"] = [6 , 1, 1 ]
HATOMS["N"] = [4 , 2, 2 ]
HATOMS["Q"] = [5 , 2, 2 ]
HATOMS["P"] = [5 , 1, 1 ]
HATOMS["S"] = [3 , 1, 2 ]
HATOMS["R"] = [6 , 4, 1 ]
HATOMS["T"] = [4 , 1, 2 ]
HATOMS["W"] = [11 , 2, 1]
HATOMS["V"] = [5 , 1, 1 ]
HATOMS["Y"] = [9 , 1, 1 ]

## Amino acid name conversion dictionary
#  3 letter residue names to 1 letter
##
longer_names={'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
              'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
              'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
              'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
              'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'            
              }

## returns average and standard deviation from a list of floats
def avg_and_sdev( flist ):

    sum = 0.0
    N = 0.0

    for S in flist:
        sum += S
        N += 1.0

    avg = sum / N

    devsum = 0.0
    for S in flist:
        devsum += (S-avg)**2
        
    sdev = sqrt( devsum / N )

    return avg, sdev

# Plist = a static list of the four hundred aa1:aa2 pairs
LK1 = longer_names.keys()
LK2 = longer_names.keys()
PList = []
for A1 in LK1:
    for A2 in LK2:
        PAIR = longer_names[A1]+longer_names[A2]
        PList.append(PAIR)
        
# XMERS = a static list of integers from 1 to 40
XMERS = []
for n in range(40):
    XMERS.append(n+1)

# PICOMP = dictionary of the 9 sp2 sidechains, for "if (AA in PICOMP)"
RList = ["W","F","Y","R","E","D","Q","N","H"]
PICOMP = {}
for A1 in RList:
    PICOMP[A1] = 0
    
###############################################################################
#                                                                             #
#                           Database Initialization                           #
#                                                                             #
###############################################################################


## PAIRSC = DATABASE DICTIONARY FOR DBPATH+"PCON2.FREQS.wBOOTDEV"
#
#  PAIRSC = { key:{ ptype: [freq,sdev] }}
#
#  key:   V_5_V means the motif VxxxxxV, database covers 20aa by 20aa and goes up to X_40_Y
#  ptype: observation type, ie: X_sr_sc vs. Y_lr_bbco
#                  X/Y -> which amino acid the frequencies are associated with (nterm or cterm of motif)
#                sr/lr -> short range and long range frequencies (contacts split by sequence locality)
#         sc/bbcn/bbco -> sp2 group, either sidechain, nterm flanking backbone or cterm flanking backbone
#
##
PAIRSC = {}
ffile = open(DBPATH+"PCON2.FREQS.wBOOTDEV").readlines()
for f in ffile:
    l = f.split()

    key = l[0]

    PAIRSC[key] = {}

    for t in range((len(l)-1)//3):
        ptype = l[ 1 + t*3 ]
        freq = float( l[ 1 + t*3 + 1] )
        sdev = float( l[ 1 + t*3 + 2] )

        PAIRSC[key][ptype] = [ freq, sdev ]

if version_info < (3,5,0):
    
    ####
    #### PYTHON2.X VERSION
    ####

    ## SC_GRIDDATA = PRECOMPUTED SIDECHAIN DATABASE DICTIONARY 
    SC_GRIDDATA = pickle.load(open(DBPATH+"SC_GRIDS.pickle0", 'rb'))
    ## BB_GRIDDATA = PRECOMPUTED BACKBONE DATABASE DICTIONARY 
    BB_GRIDDATA = pickle.load(open(DBPATH+"BB_GRIDS.pickle0", 'rb'))

else:
    
    ####
    #### PYTHON3.5 VERSION
    ####

    ## SC_GRIDDATA = PRECOMPUTED SIDECHAIN DATABASE DICTIONARY 
    SC_GRIDDATA = pickle.load(open(DBPATH+"SC_GRIDS.pickle4", 'rb'))
    ## BB_GRIDDATA = PRECOMPUTED BACKBONE DATABASE DICTIONARY 
    BB_GRIDDATA = pickle.load(open(DBPATH+"BB_GRIDS.pickle4", 'rb'))



## AVGSDEV = PRECOMPUTED AVERAGE VALUE DATABASE DICTIONARY
AVGSDEV = {}
for xmer in XMERS:
    AVGSDEV[xmer] = { "LR":{}, "SR":{} }
    PairAvgSdevFile = open(DBPATH+"AVGnSDEVS/PCON2.BB.xmer"+str(xmer)).readlines()
    for f in PairAvgSdevFile:
        l = f.split() # l[0] == two sequential amino acids flanking a single backbone group
        AVGSDEV[xmer]["SR"][l[0]] = [ float(l[2]), float(l[3]) ] # AVG, SDEV
        AVGSDEV[xmer]["LR"][l[0]] = [ float(l[5]), float(l[6]) ] # AVG, SDEV
    
    PairAvgSdevFile = open(DBPATH+"AVGnSDEVS/PCON2.SC.xmer"+str(xmer)).readlines()
    for f in PairAvgSdevFile:
        l = f.split() # l[0] == one amino acid containing a single sidechain group
        AVGSDEV[xmer]["SR"][l[0]] = [ float(l[2]), float(l[3]) ] # AVG, SDEV
        AVGSDEV[xmer]["LR"][l[0]] = [ float(l[5]), float(l[6]) ] # AVG, SDEV

###############################################################################
#                                                                             #
#                          Scoring Algorithm Functions                        #
#                                                                             #
###############################################################################

## make_linekey: grid line key generator
#
#  rounding function to put floats into the dictionary key format for the grid database
#  (bins by steps of 0.5, minimum -8.0 and maximum 12.0)
#
##
def make_linekey( Z ):
    gkey = ( "%12.1f" % ( float("%12.0f" % ( Z*2.0 ) ) / 2.0  ) )   # STEPS of 0.5

    if gkey.split()[0] == "-0.0": gkey = "0.0"
    
    gkey = float(gkey)

    if gkey > 12.0: gkey = 12.0
    if gkey < -8.0: gkey = -8.0

    return gkey
## GET_CLOSEST: finds closest X & Y grid keys for a pair of floats
#
#  uses distance to the grid key
#
##
def GET_CLOSEST( rxGRID, fv1, fv2 ):
    dlist = []
    for g1 in rxGRID.keys():
        fg1 = float(g1)
        for g2 in rxGRID[g1].keys():
            fg2 = float(g2)
            dist = sqrt( (fv1-fg1)**2 + (fv2-fg2)**2 )
            dlist.append( [dist, g1, g2] )
    dlist.sort()
    return dlist[0][1], dlist[0][2]


## WINDOW SCORING ALGORITHM
#
#  The overall scoring algorithm makes a range of scores based on different window lengths
#  for efficiency we just go from the smallest window to the largest, storing values
#  by window length
#
#  mseq      = a subsection of the sequence being scored,
#              where the position of interest starts at the max_xmer
#              
#  USE_FREQS = PCON2.FREQS.wBOOTDEV database dictionary 
#  min_xmer  = smallest window size to use
#  max_xmer  = largest window size to use
#
#  algorithm iterates over database observations to produce rolling sums..
#     SUMF = SUM[ database_frequency / database_stddev ]
#     TOT = SUM[ 1.0 / database_stddev ]
#
#  window scores are calculated by...
#     SCORE = SUMF / TOT
#
#  separate scores are calculated for four categories...
#     BB_SR_FREQ, BB_LR_FREQ, SC_SR_FREQ, SC_LR_FREQ
#
#  (note: you do not want to see what score_mseq_dumb looked like)
#
##      
def score_mseq_smart( mseq, USE_FREQS, min_xmer, max_xmer ):

    #Per residue scores, function fills then returns this
    SCORES = {}

    LR_SUMF = 0.0
    SR_SUMF = 0.0
    TOTLR = 0.0
    TOTSR = 0.0

    SC_LR_SUMF = 0.0
    SC_SR_SUMF = 0.0
    SC_TOTLR = 0.0
    SC_TOTSR = 0.0

    M1 = mseq[max_xmer]
    M2 = mseq[max_xmer+1]

    #### WINDOW LENGTH SCORE LOOP
    for P in range(max_xmer+1):
        #### BACKBONE SCORES
        ##
        ##  backbone groups have two associated residues flanking the peptide bond
        ##  observation data is summed over both independently
        ##
        ####

        ####################################
        # LEFT SCAN:                       #
        #   scan residues left of M1 & M2  #
        ####################################
        Yp = max_xmer - P     # Yp = neighborhood residue position being scanned
        yPOS1 = P - 1
        yPOS2 = P

        key1Y = "XXX"
        key2Y = "XXX"

        #####
        # scan vs. M1 (amino acid to the left of the backbone position)
        #####
        if yPOS1 != -1:
            key1Y = mseq[Yp]+"_"+str(yPOS1)+"_"+M1

            LR_SUMF += USE_FREQS[key1Y]["Y_lr_bbco"][0] * ( 1.0 / USE_FREQS[key1Y]["Y_lr_bbco"][1] )
            SR_SUMF += USE_FREQS[key1Y]["Y_sr_bbco"][0] * ( 1.0 / USE_FREQS[key1Y]["Y_sr_bbco"][1] )
            TOTLR += ( 1.0 / USE_FREQS[key1Y]["Y_lr_bbco"][1] )
            TOTSR += ( 1.0 / USE_FREQS[key1Y]["Y_sr_bbco"][1] )

        #####
        # scan vs. M2 (amino acid to the right of the backbone position)
        #####
        if yPOS2 != -1:
            key2Y = mseq[Yp]+"_"+str(yPOS2)+"_"+M2

            LR_SUMF += USE_FREQS[key2Y]["Y_lr_bbn"][0] * ( 1.0 / USE_FREQS[key2Y]["Y_lr_bbn"][1] )
            SR_SUMF += USE_FREQS[key2Y]["Y_sr_bbn"][0] * ( 1.0 / USE_FREQS[key2Y]["Y_sr_bbn"][1] )
            TOTLR += ( 1.0 / USE_FREQS[key2Y]["Y_lr_bbn"][1] )
            TOTSR += ( 1.0 / USE_FREQS[key2Y]["Y_sr_bbn"][1] )
        
        ####################################
        # RIGHT SCAN:                      #
        #   scan residues right of M1 & M2 #
        ####################################
        Xp = max_xmer + 1 + P
        xPOS1 = P
        xPOS2 = P - 1
        
        key1X = "XXX"
        key2X = "XXX"

        #####
        # scan vs. M1 (amino acid to the left of the backbone position)
        #####
        if xPOS1 != -1:
            key1X = M1+"_"+str(xPOS1)+"_"+mseq[Xp] 
            
            LR_SUMF += USE_FREQS[key1X]["X_lr_bbco"][0] * ( 1.0 / USE_FREQS[key1X]["X_lr_bbco"][1] )
            SR_SUMF += USE_FREQS[key1X]["X_sr_bbco"][0] * ( 1.0 / USE_FREQS[key1X]["X_sr_bbco"][1] )
            TOTLR += ( 1.0 / USE_FREQS[key1X]["X_lr_bbco"][1] )
            TOTSR += ( 1.0 / USE_FREQS[key1X]["X_sr_bbco"][1] )

        #####
        # scan vs. M2 (amino acid to the right of the backbone position)
        #####
        if xPOS2 != -1:
            key2X = M2+"_"+str(xPOS2)+"_"+mseq[Xp]

            LR_SUMF += USE_FREQS[key2X]["X_lr_bbn"][0] * ( 1.0 / USE_FREQS[key2X]["X_lr_bbn"][1] )
            SR_SUMF += USE_FREQS[key2X]["X_sr_bbn"][0] * ( 1.0 / USE_FREQS[key2X]["X_sr_bbn"][1] )
            TOTLR += ( 1.0 / USE_FREQS[key2X]["X_lr_bbn"][1] )
            TOTSR += ( 1.0 / USE_FREQS[key2X]["X_sr_bbn"][1] )

        #### SIDECHAIN SCORES
        ##
        ##  sidechain groups only have one associated residues, amino acid M1
        ##  (uses one less n-terminal position than backbones, and skips if not sp2)
        ##
        ####
        
        if (M1 in PICOMP) and P != max_xmer:

            scseq = mseq[:-1]#one less n-terminal position ( = same number of residues on both sides)

            ###############################
            # LEFT SCAN:                  #
            #   scan residues left of M1  #
            ###############################
            Yp = max_xmer - P - 1
            yPOS1 = P
            key1Y = scseq[Yp]+"_"+str(yPOS1)+"_"+M1    #X_separation_Y   so Y_lr_sc is lr_sc for M1

            SC_LR_SUMF += USE_FREQS[key1Y]["Y_lr_sc"][0] * ( 1.0 / USE_FREQS[key1Y]["Y_lr_sc"][1] )
            SC_SR_SUMF += USE_FREQS[key1Y]["Y_sr_sc"][0] * ( 1.0 / USE_FREQS[key1Y]["Y_sr_sc"][1] )
            SC_TOTLR += ( 1.0 / USE_FREQS[key1Y]["Y_lr_sc"][1] )
            SC_TOTSR += ( 1.0 / USE_FREQS[key1Y]["Y_sr_sc"][1] )
            
            ###############################
            # RIGHT SCAN:                 #
            #   scan residues right of M1 #
            ###############################
            Xp = max_xmer + 1 + P
            xPOS1 = P
            key1X = M1+"_"+str(xPOS1)+"_"+scseq[Xp]    #X_separation_Y   so X_lr_sc is lr_sc for M1

            SC_LR_SUMF += USE_FREQS[key1X]["X_lr_sc"][0] * ( 1.0 / USE_FREQS[key1X]["X_lr_sc"][1] )
            SC_SR_SUMF += USE_FREQS[key1X]["X_sr_sc"][0] * ( 1.0 / USE_FREQS[key1X]["X_sr_sc"][1] )
            SC_TOTLR += ( 1.0 / USE_FREQS[key1X]["X_lr_sc"][1] )
            SC_TOTSR += ( 1.0 / USE_FREQS[key1X]["X_sr_sc"][1] )

        ##
        ## SAVE CURRENT WINDOW SCORE
        ##
        if P >= min_xmer and P <= max_xmer:
            BB_LR_FREQ = LR_SUMF / TOTLR
            BB_SR_FREQ = SR_SUMF / TOTSR

            SC_LR_FREQ = 0.0
            if SC_TOTLR > 0.0:
                SC_LR_FREQ = SC_LR_SUMF / SC_TOTLR
            SC_SR_FREQ = 0.0
            if SC_TOTSR > 0.0:
                SC_SR_FREQ = SC_SR_SUMF / SC_TOTSR

            SCORES[P] = [BB_SR_FREQ, BB_LR_FREQ, SC_SR_FREQ, SC_LR_FREQ]

    return SCORES


## POSITION SCORING ALGORITHM
#
#  Predicts pi-contact frequency statistics for each position in the sequence
#
#  function 1)calculates a weighted average frequency over a range of sequence observations 
#           2)normalizes that to a z-score based on precomputed statistics for each sp2 identity (9 sidechain, 400 backbone)
#           3)looks up the contact frequency observed in the database for groups with the same identity and similar z-scores
#             (short range and long range zscores calculated independently, two dimensional grid lookup combines data by using both)
#
#           Values are calculated over a range of window lengths,
#           with z-scores averaged across the set and observation frequencies based on all observations
#
#  sequence = full sequence in a single string
#
#  idata = output dictionary linking residue index to a list of position specific values
#          used in calculating the final PScore. 
#
#   idata[i][0] = M1         = amino acid at position i
#   idata[i][0] =  AVG_BB_SRZ = zscore of the short range backbone frequency prediction (from second stage)
#   idata[i][0] =  SRBB_FbyG  = short range backbone frequency obtained from the grid lookup (third stage)
#   (... AVG_BB_LRZ, LRBB_FbyG, AVG_SC_SRZ, SRSC_FbyG, AVG_SC_LRZ, LRSC_FbyG, float(HATOMS[M1][0]))
##
def position_scores( sequence ):

    idata = {}

    #iterate over sequence[nt] from nt = 1 to nt < len(sequence) - 1 
    for nt in range(len(sequence)):
        if nt > 0 and nt + 2 < len(sequence):

            idata[nt] = []

            M1 = sequence[nt]
            M2 = sequence[nt+1]

            PAIR = M1+M2 #Backbone group identity based on 2 neighboring amino acids

            BBGD = [0.0, 0.0, 0.0]
            SCGD = [0.0, 0.0, 0.0]
            NOKEY = 0

            # find largest sequence window that can be used for position nt
            #   windows can't extend past the termini or into regions of the sequence with unknown amino acids
            use_xmer = min_xmer
            Mseq = sequence[nt-use_xmer:nt+use_xmer+2]
            while len(Mseq) == 2*use_xmer+2 and Mseq.find('X') == -1 and use_xmer <= max_xmer:
                use_xmer += 1
                Mseq = sequence[nt-use_xmer:nt+use_xmer+2]
            use_xmer -= 1
            Mseq = sequence[nt-use_xmer:nt+use_xmer+2]

            ##AVG_BB_SRF = 0.0
            ##AVG_BB_LRF = 0.0
            AVG_BB_SRZ = 0.0
            AVG_BB_LRZ = 0.0
            BTOT = 0.0

            ##AVG_SC_SRF = 0.0
            ##AVG_SC_LRF = 0.0
            AVG_SC_SRZ = 0.0
            AVG_SC_LRZ = 0.0
            STOT = 0.0

            # only score windows that have at least 1 flanking residue on each side and have no unknown amino acids
            if len(Mseq) >= 2*min_xmer+2 and Mseq.find('X') == -1:

                #################################################################
                # WINDOW SCORING ALGORITHM -> Window scores stored in ALLSCORES #
                #################################################################
                ALLSCORES = score_mseq_smart( Mseq, PAIRSC, min_xmer, use_xmer )
                
                for XN in range(max_xmer-min_xmer+1):
                    xmer = XN+min_xmer
                    xweight = (1.0 + (xmer-1))

                    if (xmer in ALLSCORES):
                        
                        #########################
                        #### BACKBONE SCORES ####
                        #########################
                        SR_FREQ = ALLSCORES[xmer][0]
                        LR_FREQ = ALLSCORES[xmer][1]

                        SRF_ZSCORE = ( SR_FREQ - AVGSDEV[xmer]["SR"][PAIR][0] ) / AVGSDEV[xmer]["SR"][PAIR][1]
                        LRF_ZSCORE = ( LR_FREQ - AVGSDEV[xmer]["LR"][PAIR][0] ) / AVGSDEV[xmer]["LR"][PAIR][1]

                        ##AVG_BB_SRF += SR_FREQ
                        ##AVG_BB_LRF += LR_FREQ
                        AVG_BB_SRZ += SRF_ZSCORE
                        AVG_BB_LRZ += LRF_ZSCORE
                        BTOT += 1.0

                        SK = make_linekey(SRF_ZSCORE)
                        LK = make_linekey(LRF_ZSCORE)

                        onNK = True
                        if (SK in BB_GRIDDATA[PAIR][xmer]):
                            if (LK in BB_GRIDDATA[PAIR][xmer][SK]):
                                GD = BB_GRIDDATA[PAIR][xmer][SK][LK]
                                BBGD[0] += GD[0]*xweight
                                BBGD[1] += GD[1]*xweight
                                BBGD[2] += GD[2]*xweight
                                onNK = False

                        if onNK:
                            NOKEY += 1
                            SK, LK = GET_CLOSEST( BB_GRIDDATA[PAIR][xmer], SRF_ZSCORE, LRF_ZSCORE )
                            GD = BB_GRIDDATA[PAIR][xmer][SK][LK]
                            BBGD[0] += GD[0]*xweight
                            BBGD[1] += GD[1]*xweight
                            BBGD[2] += GD[2]*xweight
                            onNK = False

                        ##########################
                        #### SIDECHAIN SCORES ####
                        ##########################
                        if (M1 in PICOMP): # position has an sp2 sidechain

                            SR_FREQ = ALLSCORES[xmer][2]
                            LR_FREQ = ALLSCORES[xmer][3]

                            SRF_ZSCORE = ( SR_FREQ - AVGSDEV[xmer]["SR"][M1][0] ) / AVGSDEV[xmer]["SR"][M1][1]
                            LRF_ZSCORE = ( LR_FREQ - AVGSDEV[xmer]["LR"][M1][0] ) / AVGSDEV[xmer]["LR"][M1][1]

                            SK = make_linekey(SRF_ZSCORE)
                            LK = make_linekey(LRF_ZSCORE)

                            #AVG_SC_SRF += SR_FREQ
                            #AVG_SC_LRF += LR_FREQ
                            AVG_SC_SRZ += SRF_ZSCORE
                            AVG_SC_LRZ += LRF_ZSCORE
                            STOT += 1.0

                            onNK = True
                            if (SK in SC_GRIDDATA[M1][xmer]):
                                if (LK in SC_GRIDDATA[M1][xmer][SK]):
                                    GD = SC_GRIDDATA[M1][xmer][SK][LK]
                                    SCGD[0] += GD[0]*xweight
                                    SCGD[1] += GD[1]*xweight
                                    SCGD[2] += GD[2]*xweight
                                    onNK = False
                            if onNK:
                                NOKEY += 1
                                SK, LK = GET_CLOSEST( SC_GRIDDATA[M1][xmer], SRF_ZSCORE, LRF_ZSCORE )
                                GD = SC_GRIDDATA[M1][xmer][SK][LK]
                                SCGD[0] += GD[0]*xweight
                                SCGD[1] += GD[1]*xweight
                                SCGD[2] += GD[2]*xweight
                                onNK = False

                ########
                #
                # FINAL SCORE CALCULATION
                #
                #    For each position the PScore takes 8 values, split by...
                #       LR vs SR
                #       BB vs. SC
                #       ZScore vs. Freq.
                #
                #    ZScores are calculated by an average of the zscores observed independently for each window length
                #
                #    Frequencies are taken from a sum of observed contact events divided by total observations.
                #
                ########
                                
                SRBB_FbyG = 0.0
                LRBB_FbyG = 0.0
                if BBGD[0] > 0:
                    SRBB_FbyG = BBGD[1]/BBGD[0]
                    LRBB_FbyG = BBGD[2]/BBGD[0]
                    ##AVG_BB_SRF /= BTOT
                    ##AVG_BB_LRF /= BTOT
                    AVG_BB_SRZ /= BTOT
                    AVG_BB_LRZ /= BTOT

                SRSC_FbyG = 0.0
                LRSC_FbyG = 0.0
                if SCGD[0] > 0:
                    SRSC_FbyG = SCGD[1]/SCGD[0]
                    LRSC_FbyG = SCGD[2]/SCGD[0]
                    ##AVG_SC_SRF /= STOT
                    ##AVG_SC_LRF /= STOT
                    AVG_SC_SRZ /= STOT
                    AVG_SC_LRZ /= STOT

                #ostring = "%5s %5i %1s %1s %2i SRBB %8.5f %8.5f LRBB %8.5f %8.5f SRSC %8.5f %8.5f LRSC %8.5f %8.5f" % \
                #          (seqname, nt, M1, M2, NOKEY, \
                #           AVG_BB_SRZ, SRBB_FbyG, \
                #           AVG_BB_LRZ, LRBB_FbyG, \
                #           AVG_SC_SRZ, SRSC_FbyG, \
                #           AVG_SC_LRZ, LRSC_FbyG)
                #ofile.write(ostring+"\n")

                idata[nt] = [M1, AVG_BB_SRZ, SRBB_FbyG, AVG_BB_LRZ, LRBB_FbyG, AVG_SC_SRZ, SRSC_FbyG, AVG_SC_LRZ, LRSC_FbyG, float(HATOMS[M1][0]) ]

    return idata

## make_window: calculates average idata values for all windows of a defined length
#
#  ondata = idata from position_scores
#
#  wlen = number of residues wanted on each side (half the window length minus one)
#
#  positions that are too close to the termini do not result in window values
#  (these positions will eventually get assigned values for the closest available window)
##
def make_window( ondata, wlen ):
    NK = list(ondata.keys())
    NK.sort()
    
    WN = {}
    for N in NK:
        if N > wlen and N+wlen < NK[len(NK)-1]:
            dat = []
            for d in range(9):
                dat.append(0.0)
            for w in range(2*wlen+1):
                Nw = N - wlen + w
                for d in range(9):
                    dat[d] += ondata[Nw][d+1]
            for d in range(9):
                dat[d] /= float(2*wlen+1)
            WN[N] = dat
    
    return WN

## SCORE_LINE: weighting function used to combine windows and observation times
#              into a position dependent phase separation propensity score
def SCORE_LINE( WEIGHTSET, L ):
    SCORE1 = 0.0
    SCORE2 = 0.0
    SCORE3 = 0.0
    for d in [2,5, 8,11,14,17,20,23]:
        SCORE1 += float(L[d])*WEIGHTSET[d]
    for d in [3,6, 9,12,15,18,21,24]:
        SCORE2 += float(L[d])*WEIGHTSET[d]
    for d in [4,7,10,13,16,19,22,25]:
        SCORE3 += float(L[d])*WEIGHTSET[d]

    SCORE1 /= (float(L[26])**(1.0+WEIGHTSET[26]))
    SCORE2 /= (float(L[27])**(1.0+WEIGHTSET[27]))
    SCORE3 /= (float(L[28])**(1.0+WEIGHTSET[28]))

    SCORE = 0.0
    SCORE += ((1.0+WEIGHTSET[29])*SCORE1)
    SCORE += ((1.0+WEIGHTSET[30])*SCORE2)
    SCORE += ((1.0+WEIGHTSET[31])*SCORE3)

    return SCORE



######
#
# Fasta File Parser
#
######
iseqs = {}
inputseqs = open(argv[1]).readlines()
onfasta = ""
seq = ""
for i in inputseqs:
    if i[0] == ">":
        if onfasta != "":
            iseqs[onfasta] = seq
        seq = ""
        onfasta = i.split()[0]
    else:
        seq += i.split()[0]

iseqs[onfasta] = seq


######
#
# Sequence Scoring
#
#    For every suitable sequence in the fasta file (matches test set minimum size, does not have ambigious residues)
#         1) Calculate full list of per-residue statistics based on 3 window size ranges
#         2) Combine statistics through optimized weighting function
#         3) Calculate average score of the top 60 residues
#
######


iseqcount = 0
for i in iseqs.keys():
    iseqcount += 1
    if cOUTFILE and (cMUTE == False):
        print("Working on sequence: %8i of %-8i" % (iseqcount, len(iseqs.keys())))
    
    sname = i
    sseq = iseqs[i]

    if len(sseq) >= 140 and sseq.find('X') == -1 and sseq.find('B') == -1 and sseq.find('U') == -1 and sseq.find('Z') == -1:

        seqdata = position_scores( sseq )

        ntkeys = list(seqdata.keys())
        ntkeys.sort()

        WIN = []
        WIN_NK = []
        for Wn in [20, 40, 60]:
            WINDOW = make_window(seqdata, Wn)
            WIN.append(WINDOW)
            NK = list(WINDOW.keys())
            NK.sort()
            WIN_NK.append(NK)


        NTSEQDATA = {}

        for N in ntkeys:

            ostrings = []
            for d in range(9):
                ostrings.append("")

            ostring = "%6i" %(N)

            for Wn in range(3):
                if (N in WIN[Wn]):
                    WDATA = WIN[Wn][N]
                else:
                    if N < WIN_NK[Wn][0]:
                        WDATA = WIN[Wn][WIN_NK[Wn][0]]
                    else:
                        WDATA = WIN[Wn][WIN_NK[Wn][len(WIN_NK[Wn])-1]]

                for d in range(9):
                    ostrings[d] += " %9.6f" % (WDATA[d])

            ostring = "%-21s %6i" % ("TEST", N)
            #ostring += " SBZ"
            ostring += ostrings[0]
            #ostring += " SBF"
            ostring += ostrings[1]
            #ostring += " LBZ"
            ostring += ostrings[2]
            #ostring += " LBF"
            ostring += ostrings[3]
            #ostring += " SSF"
            ostring += ostrings[4]
            #ostring += " SSZ"
            ostring += ostrings[5]
            #ostring += " LSF"
            ostring += ostrings[6]
            #ostring += " LSZ"
            ostring += ostrings[7]
            #ostring += " C"
            ostring += ostrings[8]

            NTSEQDATA[N] = ostring

        #####
        #
        # This are the optimized weight settings
        #
        #####
        hfile = open(DBPATH+"WeightSettings.txt").readlines()#T39.353.r4.Q.R.W").readlines()

        WEIGHTSET = {}
        for h in hfile:
            l = h.split()
            if l[0] != "ALLSCORE:" and l[0] != "LOG:":
                C = int(l[0])
                W = float(l[1])
                WEIGHTSET[C] = W

        PDATA = {}

        for N in range(len(sseq)):#ntkeys:

            ostring = ""

            if (N in NTSEQDATA):

                ostring = "ResiduePScore: %5s %1s " % (N, sseq[N])
                
                NSdata = seqdata[N]

                L = NTSEQDATA[N].split()

                ####
                #
                #  RPScore is the residue position specific PScore, and this is the final normalization step for that
                #
                ####                
                AV = 0.002027759051759575
                SDEV = 0.0028857425115758704
                RPSCORE = (SCORE_LINE( WEIGHTSET, L ) - AV) / SDEV
                #Individual residue PScores are normalized to z-scores vs. PDB average

                ####
                #
                #  Output format for residue position statistics used in calculating the RPScore
                #
                #  Statistics listed...
                #    short-range backbone frequency sum z-score, relative to identity
                #    final predicted short-range backbone frequency
                #
                #    short-range backbone frequency sum z-score, relative to identity
                #    final predicted short-range backbone frequency
                #
                #    short-range backbone frequency sum z-score, relative to identity
                #    final predicted short-range backbone frequency
                #
                #    short-range backbone frequency sum z-score, relative to identity
                #    final predicted short-range backbone frequency
                #
                #    residue position specific PScore
                #
                ####
                if cRSCORE:
                    ostring +=  " %8.3f" % (RPSCORE)
                if cCOMPONENTS:
                    ostring += "   Components (Zscore & Freq) SRBB: %6.3f %6.3f LRBB: %6.3f %6.3f SRSC: %6.3f %6.3f LRSC: %6.3f %6.3f" % \
                          (NSdata[1], NSdata[2], NSdata[3], NSdata[4], NSdata[5], NSdata[6], NSdata[7], NSdata[8])


                ostring += "  NAME: "+sname
                
                PDATA[int(N)] = RPSCORE

            ####
            #
            #  Residue statistics, if you want to print them
            #
            ####

            if (cRSCORE or cCOMPONENTS) and ostring != "":

                if cOUTFILE:
                    OFILE.write(ostring+"\n")
                else:
                    print(ostring)


        ####
        #
        #  The final step in calculating the PScore is then to take the
        #  average of the top 60 residues and then normalize the result
        #
        ####

        Dbelow = 0.0
        Dtot = float(len(PDATA.keys()))
        Zsum = 0.0

        MAX = 0.0
        MLIST = []
        ZLIST = []
        for W in PDATA.keys():
            odat = PDATA[W]
            if odat > MAX: MAX = odat
            MLIST.append(odat)
            Zsum += odat
            ZLIST.append(odat)


        MLIST.sort()
        MLIST.reverse()

        ZLIST.sort()
        ZLIST.reverse()

        TopX = ZLIST[60-1]
        ZWum = 5

        ZUSE = {}
        ZLIST = []
        for W in PDATA.keys():
            odat = PDATA[W]
            if odat >= TopX:
                for n in range(ZWum*2+1):
                    if ((W-ZWum+n) in PDATA):
                        ZUSE[W-ZWum+n] = 0


        ZSum = 0.0
        ZTot = 0.0
        for K in ZUSE.keys():
            ZSum += PDATA[K]
            ZTot += 1.0

            ####
            #
            #  If you want to know which residues contributed to the PScore
            #
            ###
            
            #print("RESIDUE_USED_IN_PSCORE:", K)
            
        ZAvg = ZSum / ZTot

        #Final PScore is normalized to z-score vs. PDB average
        ZScore = (ZAvg - 0.8425291305646876)/0.5933030779034423

        ####
        #
        #  Prints the final PScore per sequence
        #     (%6.2f = 2 decimal places, but if you need more...)
        #
        ####
        OSTRING = "%-20s %6.2f     %20s" % ("PScore:", ZScore, i)

        if cOUTFILE:
            OFILE.write(OSTRING+"\n")
        else:
            print(OSTRING)


