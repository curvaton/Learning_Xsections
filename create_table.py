#!/usr/bin/env python3

"""
Test file to create the data tables
"""

import pyslha
import csv
import os

from progressbar import ProgressBar               # just a simple progress bar

workingdirpath = os.getcwd()

datadirpath = workingdirpath+"/IDM_files/"

listdir = os.listdir(datadirpath)

listdir.sort()

pbar = ProgressBar()

#Initialising the data list
#data=[]

#defining a function for reading and extracting the relevant parameters
#from a slha file

def read_extract_slha( str ):
    
    #Reading the file
    d = pyslha.read(datafilepath)

    #Retrieving the relevant parameters
    MH0 = d.blocks['MASS'][35]
    MA0 = d.blocks['MASS'][36]
    MHC = d.blocks['MASS'][37]
    lam2 = d.blocks['FRBLOCK'][6]
    lamL = d.blocks['FRBLOCK'][5]

    #Retrieving the 13 TeV cross sections
    """
    Reminder for the PDG codes
    PDG 35 : H0
    PDG 36 : A0
    PDG 37 : H+
    PDG -37 : H-
    """

    #Getting the list of cross sections
    myproc3535 = d.xsections[2212,2212,35,35]
    myproc3636 = d.xsections[2212,2212,36,36]
    myproc3737 = d.xsections[2212,2212,-37,37]
    myproc3537 = d.xsections[2212,2212,35,37]
    myproc3637 = d.xsections[2212,2212,36,37]
    myproc3735 = d.xsections[2212,2212,-37,35]
    myproc3736 = d.xsections[2212,2212,-37,36]
    myproc3536 = d.xsections[2212,2212,35,36]

    #Getting the 13 Tev Xsec value
    xsec_3535_13TeV = myproc3535.get_xsecs(sqrts=13000.)[0].value
    xsec_3636_13TeV = myproc3636.get_xsecs(sqrts=13000.)[0].value
    xsec_3737_13TeV = myproc3737.get_xsecs(sqrts=13000.)[0].value
    xsec_3537_13TeV = myproc3537.get_xsecs(sqrts=13000.)[0].value
    xsec_3637_13TeV = myproc3637.get_xsecs(sqrts=13000.)[0].value
    xsec_3735_13TeV = myproc3735.get_xsecs(sqrts=13000.)[0].value
    xsec_3736_13TeV = myproc3736.get_xsecs(sqrts=13000.)[0].value
    xsec_3536_13TeV = myproc3536.get_xsecs(sqrts=13000.)[0].value

    datatemp = [MH0,MA0,MHC,lam2,lamL,xsec_3535_13TeV,xsec_3636_13TeV,xsec_3737_13TeV,xsec_3537_13TeV,xsec_3637_13TeV,xsec_3735_13TeV,xsec_3736_13TeV,xsec_3536_13TeV]

    return datatemp;


header = ['MH0','MA0','MHC','lam2','lamL','xsec_3535_13TeV','xsec_3636_13TeV','xsec_3737_13TeV','xsec_3537_13TeV','xsec_3637_13TeV','xsec_3735_13TeV','xsec_3736_13TeV','xsec_3536_13TeV']

#print(data)

with open('table_final.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in pbar(range(0,len(listdir))):
        datafilepath = datadirpath+listdir[i]
        datatemp = read_extract_slha(datafilepath)    
#    data.append(datatemp)
        writer.writerow(datatemp)
#        print(listdir[i]+" treated")


pbar.finish()

