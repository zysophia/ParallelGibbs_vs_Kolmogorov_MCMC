import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.patches as patches

palette ={"Kolmogorov": "darkblue","Hoeffding": "red",   "MCMCPro": "darkorange","2ADAGibbs": "darkred"}

sns.set(font_scale = 5)



def lineplot_complexity_vs_eps_Ising(n=6,b=-0.001):
    
  
    prefix1="../data/isingoutput_n="
    prefix2="_beta="
    csv_filename = "{0}{1}{2}{3}.csv".format(prefix1, n, prefix2,b)

    df = pd.read_csv(csv_filename)
    chain = "Ising"
    df_sub = df[(df["chain"]==chain)&(df["n"]==n)]#&(df["algorithm"]=="Kolmogorov")]
    # df_sub["log10 (sample complexity)"] = df_sub["sample_complexity"]
    df_sub["1/$\epsilon$"] = 1/df_sub["epsilon"]
    df_sub["complexity"] = df_sub["sample_complexity"]
    plt.figure(figsize=(5.4, 5.5))
    # plt.figure(figsize=(4.8,5.8))
    sns.set_theme(style="darkgrid")
    g = sns.lineplot(x="1/$\epsilon$", y="complexity", hue="algorithm", style = "algorithm", \
        errorbar=('ci', 95) ,data=df_sub, markers=True, markersize = 8, dashes=False, linewidth=2, palette=palette) # style = "different_weights", 
    # g.set(ylim=(10**5, 10**12))
    g.set_yscale('log')
    g.set_xscale('log')
    plt.grid(True,which="both",ls="--",c='gray', alpha=0.5)  

    g.set_xlabel("1/$\epsilon$",fontsize=15)
    g.set_ylabel("complexity",fontsize=15)
    plt.setp(g.get_legend().get_texts(), fontsize='16') # for legend text
    plt.setp(g.get_legend().get_title(), fontsize='19') # for legend title

    plt.show()
    prefix1="../new_figs/isingoutput_n="
    fig_name="{0}{1}{2}{3}.png".format(prefix1, n, prefix2,b)
    g.figure.savefig(fig_name, dpi=200)




if __name__ == "__main__":
    
   
    # lineplot_complexity_vs_eps_Ising(n=2)
    #lineplot_complexity_vs_eps_Ising(n=3,b=-0.001)
    # lineplot_complexity_vs_eps_Ising(n=4)
    lineplot_complexity_vs_eps_Ising(n=6,b=-0.005)
 

   



    



    


