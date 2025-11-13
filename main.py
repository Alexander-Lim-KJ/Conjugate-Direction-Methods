import utils, torch
import constants

SEED = 1234 # integers
FOLDER_PATH = "./nonlinearLeastSquare" #f"./{sys.argv[1]}"
ALG = "CR"
PROBLEM = "GN" # "NN", "GN", "KR"
DATASET = "Covtype"
INITX0 = "torch" #zeros, ones, uniform, normal, torch
VERBOSE = True

if "CG" == ALG:
    ALG = [("NewtonCG", constants.cCG)]
elif "MR" == ALG:
    ALG = [("NewtonMR_NC", constants.cMR)]
elif "CR" == ALG:
    ALG = [("NewtonCR_NC", constants.cCR)]
elif "TR" == ALG:
    ALG = [("NewtonCG_TR_Steihaug", constants.cTR_STEI)]
elif "LBFGS" == ALG:
    ALG = [("L-BFGS", constants.cL_BFGS)]
    
def run(folder_path, dataset, alg, problem, x0, verbose):
    
    assert type(alg) == list
        
    print("***Running on", str(constants.cCUDA) + "***")
    
    for j, c in alg:
        print("\n" + 45 * ".")
        algo, x0 = utils.execute(folder_path, dataset, problem, j, x0, c, verbose)
        utils.saveRecords(folder_path, j, algo.record)

if __name__ == "__main__":
    if type(SEED) == int:
        torch.manual_seed(SEED)
    run(FOLDER_PATH, DATASET, ALG, PROBLEM, INITX0, VERBOSE)