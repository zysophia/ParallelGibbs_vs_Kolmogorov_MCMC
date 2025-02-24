import pandas as pd 
import numpy as np 
from collections import Counter

if __name__ == "__main__":



    datalist = []
    f = open("./logs/isingoutput_v1_241106_n3.log", 'r')
    for line in f.readlines():
        ss = line.split(",")
        if ss[0].startswith("INFO:root:Ising model"):
            chain = "Ising"
            n = int(ss[1][4:])
        elif ss[0].startswith("INFO:root:parameters: "):
            beta_max = float(ss[1][ss[1].find("= ")+2:])
            eps = float(ss[2][ss[2].find("= ")+2:])
        elif ss[0].startswith("INFO:root:real value z = "):
            real_z = float(ss[0][ss[0].find("= ")+2:])
        elif ss[0].startswith("INFO:root:Kolmogorov with eps_kol takes"):
            algorithm = "Kolmogorov"
            kolsteps = int(ss[0][ss[0].find("takes ")+6:-6])
            tpasteps = int(ss[1][ss[1].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, kolsteps, tpasteps, real_z, None, None])
        elif ss[0].startswith("INFO:root:Kolmogorov with Hoeffding takes"):
            algorithm = "Kolmogorov Hoeffding"
            kolsteps = int(ss[0][ss[0].find("takes ")+6:-6])
            tpasteps = int(ss[1][ss[1].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, kolsteps, tpasteps, real_z, None, None])
        elif ss[0].startswith("INFO:root:Kolmogorov (compute z) takes"):
            algorithm = "Kolmogorov"
            kolsteps = int(ss[0][ss[0].find("takes ")+6:-6])
            z = float(ss[1][ss[1].find("= ")+2:])
            tpasteps = int(ss[2][ss[2].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, kolsteps, tpasteps, real_z, z, z/real_z-1])
        elif ss[0].startswith("INFO:root:Kolmogorov Hoeffding (compute z) takes"):
            algorithm = "Kolmogorov Hoeffding"
            kolsteps = int(ss[0][ss[0].find("takes ")+6:-6])
            z = float(ss[1][ss[1].find("= ")+2:])
            tpasteps = int(ss[2][ss[2].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, kolsteps, tpasteps, real_z, z, z/real_z-1])
        elif ss[0].startswith("INFO:root:RunAlgorithm parallelGibbs"):
            algorithm = "Paralell Gibbs"
            steps = int(ss[1][ss[1].find("takes ")+6:-6])
            z = float(ss[2][ss[2].find("= ")+2:])
            tpasteps = int(ss[3][ss[3].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, steps, tpasteps, real_z, z, z/real_z-1])
        elif ss[0].startswith("INFO:root:RunAlgorithm MCMCPro"):
            algorithm = "MCMCPro"
            steps = int(ss[1][ss[1].find("takes ")+6:-6])
            z = float(ss[2][ss[2].find("= ")+2:])
            tpasteps = int(ss[3][ss[3].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, steps, tpasteps, real_z, z, z/real_z-1])
        
    df = pd.DataFrame(datalist, columns=["chain", "n", "beta_max", "epsilon", \
        "algorithm", "sample_complexity", "TPAsteps", "true_Q", "estimate_Q", "error"])
    print(df)  
    df.to_csv("../data/isingcompare_complexity_v1_241106.csv_n3", index=False)
            

