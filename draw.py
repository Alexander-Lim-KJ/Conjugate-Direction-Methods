# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:56:23 2023

@author: uqalim8
"""

import matplotlib
import matplotlib.pyplot as plt
import utils, torch

FOLDER_PATH = "./pde2"#"./kernelResults_nolowrank"
SPLIT = ".json"


def normalize(x):
    x = x - torch.min(x)
    return x/torch.max(x)

def keys(x):
    n = x.split("%")[0]
    if x == "Single Sample.json":
        return 0
    else:
        return float(n)
    
def drawPlots(records, stats, folder_path):
    
    STATS = {"ite":"Iterations", "orcs":"Oracle Calls", "time":"Time",
                       "f":"Objective Values", "g_norm":"Gradient Norm",
                       "res-res":"$||b - Ax_k - r_k||$",
                       "<b,Apk>": "$<b,Ap_k>$",
                       "alpha": "Step Size"}
    for x, y in stats:
        for name, record in records:
            plt.semilogy(record[x], record[y], label = name)
        plt.xlabel(STATS[x])
        plt.ylabel(STATS[y])
        plt.legend()
        y = y.replace("<","").replace(">","")
        plt.savefig(folder_path + f"{x}_{y}.png")
        plt.close()


def plotGraphs(CG, CR, MR, folder_path = None):
    FONT = {'size':14,"weight":"bold"} 
    matplotlib.rc('font', **FONT)
    # CG
    if not CG is None:
        #CG = CG.stat 
        print("CG:", CG.keys())
        ite = len(CG['|rk|/|b|'])

        #plt.semilogy(range(1, ite), CG['A(αp)-(αA)p'][1:], label = r"$||\mathbf{A}(\alpha\mathbf{p}_k) - (\alpha\mathbf{A})\mathbf{p}_k||$")
        # plt.semilogy(range(1, ite), CG['|b-Ax-r|'][1 : ], linewidth = 2, linestyle = "--", label = "$||\mathbf{b} - \mathbf{Ax}_k - \mathbf{r}_k||$")
        # plt.semilogy(range(1, ite), CG['<b, Apk>'][1:], linewidth = 2, linestyle = "-.", label = r"$||diag\{<\mathbf{p}_k, \mathbf{Ap}_k>\} - \mathbf{P}_k^{\top}\mathbf{AP}_k||$")
        # plt.semilogy(range(1, ite), CG['<b, rk>'][1:], linewidth = 2, label = r"$||\mathbf{I}_k - \mathbf{R}_k^{\top}\mathbf{R}_k||$")
        #plt.semilogy(range(1, ite), CG['<b, Apk>'][1:], label = r"$<\mathbf{b},\mathbf{Ap}_k>$")
        #plt.semilogy(range(1, ite), CG['<b, rk>'][1:], label = r"$<\mathbf{b}, \mathbf{r}_k>$")
        #plt.semilogy(range(0, ite), CG['pred'][ : ite], linewidth = 2, label = "MSE")
        plt.semilogy(range(0, ite), CG['|Apk|/|Ab|'], linestyle = "--", linewidth = 2, label = "$||\mathbf{Ap}_k||/||\mathbf{Ab}||$")
        plt.semilogy(range(0, ite), CG['|Ark|/|Ab|'], linestyle = "-.", linewidth = 2, label = "$||\mathbf{Ar}_k||/||\mathbf{Ab}||$")
        plt.semilogy(range(0, ite), CG['|rk|/|b|'][:ite], linewidth = 2, label = "$||\mathbf{r}_k||/||\mathbf{b}||$")
        #plt.ylim(1e-2, 1e2)
        plt.title("Conjugate Gradient")
        plt.xlabel("iteration k")
        plt.legend()
        if folder_path is None:
            plt.show()
        else:
            plt.savefig(folder_path + "/CGrelres.png")
            plt.close()

    # CR
    if not CR is None:
        #CR = CR.stat
        print("CR:", CR.keys())
        ite = len(CR['|rk|/|b|'])
        
        #plt.semilogy(range(1, ite), CR['A(αp)-(αA)p'][1:], label = r"$||\mathbf{A}(\alpha\mathbf{p}_k) - (\alpha\mathbf{A})\mathbf{p}_k||$")
        # plt.semilogy(range(1, ite), CR['|b-Ax-r|'][1 : ], linewidth = 2, linestyle = "--", label = "$||\mathbf{b} - \mathbf{Ax}_k - \mathbf{r}_k||$")
        # plt.semilogy(range(1, ite), CR['<b, Ark>'][1 : ite], linewidth = 2, linestyle = "-.", label = r"$||diag\{<\mathbf{r}_k, \mathbf{Ar}_k>\} - \mathbf{R}_k^{\top}\mathbf{AR}_k||$")
        # plt.semilogy(range(1, ite), CR['<Ab, Ap>'][1 : ite], linewidth = 2, label = r"$||\mathbf{I}_k - (\mathbf{AP}_k)^{\top}\mathbf{AP}_k||$")
        #plt.semilogy(range(1, ite), CR['<b, Ark>'][1 : ite], label = r"$<\mathbf{b}, \mathbf{Ar}_k>$")
        #plt.semilogy(range(1, ite), CR['<Ab, Ap>'][1 : ite], label = r"$<\mathbf{Ab}, \mathbf{Ap}_k>$")
        #plt.semilogy(range(0, ite), CR['pred'][ : ite], linewidth = 2, label = "MSE")
        plt.semilogy(range(0, ite), CR['|Ark|/|Ab|'][ : ite], color = "orange", linewidth = 2, linestyle = "-.", label = "$||\mathbf{Ar}_k||/||\mathbf{Ab}||$")
        plt.semilogy(range(0, ite), CR['|rk|/|b|'][ : ite], color = "g", linewidth = 2, label = "$||\mathbf{r}_k||/||\mathbf{b}||$")
        #plt.ylim(1e-2, 1e2)
        plt.title("Conjugate Residual")
        plt.xlabel("iteration k")
        plt.legend()
        if folder_path is None:
            plt.show()
        else:
            plt.savefig(folder_path + "/CRrelres.png")
            plt.close()

    # MR.
    if not MR is None:
        #MR = MR.stat
        print("MR:", MR.keys())
        ite = len(MR['|rk|/|b|'])
        
        #plt.semilogy(range(1, ite), MR['A(αp)-(αA)p'][1:-1], label = r"$||\mathbf{A}(\tau\mathbf{d}_k) - (\tau\mathbf{A})\mathbf{d}_k||$")
        # plt.semilogy(range(1, ite), MR['|b-Ax-r|'][1 : ], linewidth = 2, linestyle = "--", label = "$||\mathbf{b} - \mathbf{Ax}_k - \mathbf{r}_k||$")
        # plt.semilogy(range(1, ite), MR['<b, Ar>'][1 : ite], linewidth = 2, linestyle = "-.", label = r"$||diag\{<\mathbf{r}_k, \mathbf{Ar}_k>\} - \mathbf{R}_k^{\top}\mathbf{AR}_k||$")
        # plt.semilogy(range(2, ite), MR['<Ad0, Adk>'][2 : ite], linewidth = 2, label = r"$||\mathbf{I}_k - (\mathbf{AD}_k)^{\top}\mathbf{AD}_k||$")
        #plt.semilogy(range(1, ite), MR['<b, Ar>'][1 : ite], label = r"$<\mathbf{b}, \mathbf{Ar}_k>$")
        #plt.semilogy(range(2, ite), MR['<Ad0, Adk>'][2 : ite], label = r"$<\mathbf{Ad}_0, \mathbf{Ad}_k>$")
        #plt.semilogy(range(0, ite), MR['pred'][ : ite], linewidth = 2, label = "MSE")
        plt.semilogy(range(0, ite), MR['|Ark|/|Ab|'][ : ite], color = "orange", linewidth = 2, linestyle = "-.", label = "$||\mathbf{Ar}_k||/||\mathbf{Ab}||$")
        plt.semilogy(range(0, ite), MR['|rk|/|b|'][ : ite], color = "g", linewidth = 2, label = "$||\mathbf{r}_k||/||\mathbf{b}||$")
        #plt.ylim(1e-2, 1e2)
        plt.title("Minimal Residual")
        plt.xlabel("iteration k")
        plt.legend()
        if folder_path is None:
            plt.show()
        else:
            plt.savefig(folder_path + "/MRrelres.png")
            plt.close()

def drawGaussTiger(folder_path):
    records = utils.openAllRecords(folder_path)
    img = torch.zeros(1024, 1024, 3)
    for j, i in enumerate(records.keys()):
        img[:,:,j] = normalize(torch.tensor(records[i]["xk_lifted"]).reshape(1024, 1024))
    plt.imshow(img)

if __name__ == "__main__":
    #drawGaussTiger(FOLDER_PATH)
    records = utils.openAllRecords(FOLDER_PATH)
    plotGraphs(records["CG"], records["CR"], records["MR"], FOLDER_PATH)