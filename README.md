# Code of Online Submodular Maximization via Online Convex Optimization
This repository houses the code used for the experiments in our work:

> T. Si-Salem, G. Özcan, I. Nikolaou, E. Terzi, S. Ioannidis, "Online Submodular Maximization via Online Convex Optimization", Proceedings of the AAAI Conference on Artificial Intelligence, 2024.

The work explores efficient online algorithms for maximizing monotone submodular functions under general matroid constraints. Please cite this paper ([a preprint is available](https://arxiv.org/pdf/2309.04339.pdf)) if you intend to use this code for your research.

**Datasets.** The datasets that we used are contained in the `datasets/` folder.
- $\texttt{ZKC}$
- $\texttt{Epinions}$
- $\texttt{MovieLens}$
- $\texttt{TeamFormation}$

Dataset $\texttt{SynthWC}$  is generated on the fly in simple_exps.py. 
### Algorithms
All algorithms are implemented in the `oco_tools.py` file.
Implemented algorithms:
- $\texttt{RAOCO - OGA}$
- $\texttt{RAOCO - OMA}$
- $\texttt{FSF}^*$
- $\texttt{TabularGreedy}$

Example: \
Running the $\texttt{RAOCO - OGA}$ algorithm on the $\texttt{ZKC}$ dataset with partition matroid constaints,\
where from every partition we select at most $k$ nodes:\

``` bash
python3 main.py --problemType ZKC --eta 1  --policy OGA --k 2  --partitions datasets/ZKC_100_01_42_partitions --input datasets/ZKC_100_01_42
```

If we want to use the uniform matroid constraint we just ommit the `--partitions` argument.

### Results
The results of our experiments are saved in the `results/` folder.

### Plots
Given a results folder the `GeneratePlots.py` script will generate plots of the results, as well as tables containing\
the fractional optimal value, the integral (and fractional) rewards divided by the fractional optimal, and the parameters of the algorithms (e.g. $\eta$).\
We report rewards the average cumulative reward at iteration $t = T/3, 2T/3, T$, where $T$ is the time horizon.\
If the experiments are run with different random seeds (e.g. python3 main.py ... --seed 42), then
we calculate the average rewards as well as the standard deviations of the rewards.
