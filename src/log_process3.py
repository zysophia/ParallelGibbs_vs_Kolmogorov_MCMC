import pandas as pd 
import numpy as np 
from collections import Counter

if __name__ == "__main__":


# logging without calculating real_z and acutal precision. Used for large values of n 
    n=6
    b = -0.005  
    prefix1="./logs/isingoutput_n="
    prefix2="_beta="
    log_filename = "{0}{1}{2}{3}.log".format(prefix1, n, prefix2,b)

    datalist = []
    f = open(log_filename, 'r')
    for line in f.readlines():
        ss = line.split(",")
        if ss[0].startswith("INFO:root:Ising model"):
            chain = "Ising"
            n = int(ss[1][4:])
        elif ss[0].startswith("INFO:root:parameters: "):
            beta_max = float(ss[1][ss[1].find("= ")+2:])
            eps = float(ss[2][ss[2].find("= ")+2:])
      #  elif ss[0].startswith("INFO:root:real value z = "):
       #     real_z = float(ss[0][ss[0].find("= ")+2:])
        elif ss[0].startswith("INFO:root:Kolmogorov with eps_kol takes"):
            algorithm = "Kolmogorov"
            kolsteps = int(ss[0][ss[0].find("takes ")+6:-6])
            tpasteps = int(ss[1][ss[1].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, kolsteps, tpasteps,  None])
        elif ss[0].startswith("INFO:root:Kolmogorov with Hoeffding takes"):
            algorithm = "Hoeffding"
            kolsteps = int(ss[0][ss[0].find("takes ")+6:-6])
            tpasteps = int(ss[1][ss[1].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, kolsteps, tpasteps,  None])
        elif ss[0].startswith("INFO:root:Kolmogorov (compute z) takes"):
            algorithm = "Kolmogorov"
            kolsteps = int(ss[0][ss[0].find("takes ")+6:-6])
            z = float(ss[1][ss[1].find("= ")+2:])
            tpasteps = int(ss[2][ss[2].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, kolsteps, tpasteps,  z])
        elif ss[0].startswith("INFO:root:Kolmogorov Hoeffding (compute z) takes"):
            algorithm = "Hoeffding"
            kolsteps = int(ss[0][ss[0].find("takes ")+6:-6])
            z = float(ss[1][ss[1].find("= ")+2:])
            tpasteps = int(ss[2][ss[2].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, kolsteps, tpasteps, z])
        elif ss[0].startswith("INFO:root:RunAlgorithm MCMCPro"):
            algorithm = "MCMCPro"
            steps = int(ss[1][ss[1].find("takes ")+6:-6])
            z = float(ss[2][ss[2].find("= ")+2:])
            tpasteps = int(ss[3][ss[3].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, steps, tpasteps, z])
        elif ss[0].startswith("INFO:root:RunAlgorithm parallelGibbs"):
            algorithm = "2ADAGibbs"
            steps = int(ss[1][ss[1].find("takes ")+6:-6])
            z = float(ss[2][ss[2].find("= ")+2:])
            tpasteps = int(ss[3][ss[3].find("takes ")+6:-6])
            datalist.append([chain, n, beta_max, eps, algorithm, steps, tpasteps, z])
      
        
    df = pd.DataFrame(datalist, columns=["chain", "n", "beta_max", "epsilon", \
        "algorithm", "sample_complexity", "TPAsteps", "true_Q"])
    print(df)  
    prefix1="../data/isingoutput_n="
    csv_filename = "{0}{1}{2}{3}.csv".format(prefix1, n, prefix2,b)

    df.to_csv(csv_filename, index=False)
            

