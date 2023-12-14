# Code of Online Submodular Maximization via Online Convex Optimization
This repository houses the code used for the experiments in our work:

> T. Si-Salem, G. Özcan, I. Nikolaou, E. Terzi, S. Ioannidis, "Online Submodular Maximization via Online Convex Optimization", Proceedings of the AAAI Conference on Artificial Intelligence, 2024.

The work explores efficient online algorithms for maximizing monotone submodular functions under general matroid constraints. Please cite this paper ([a preprint is available](https://arxiv.org/pdf/2309.04339.pdf)) if you intend to use this code for your research.

**Datasets.** All datasets used in this work are located in the `datasets/` folder.
- $\texttt{ZKC}$
- $\texttt{Epinions}$
- $\texttt{MovieLens}$
- $\texttt{TeamFormation}$

Dataset $\texttt{SynthWC}$  is generated on the fly in simple_exps.py. 
**Algorithms.**  All algorithms are implemented in the `oco_tools.py` file.
Implemented algorithms:
- $\texttt{RAOCO - OGA}$
- $\texttt{RAOCO - OMA}$
- $\texttt{FSF}^*$
- $\texttt{TabularGreedy}$

**Examples.** 
Execute $\texttt{RAOCO-OGA}$ algorithm on the $\texttt{ZKC}$ dataset under partition matroid constraints, restricting selection to $k$ nodes per partition.

``` bash
python3 main.py --problemType ZKC --eta 1  --policy OGA --k 2  --partitions datasets/ZKC_100_01_42_partitions --input datasets/ZKC_100_01_42
```

Uniform matroid? Leave out the `--partitions` option.

**Logged Results.**  The results of our experiments are saved in the `results/` folder.

**Visualization.** The GeneratePlots.py script processes a populated results/ folder by: (1) generating plots, and (2) building tables with: (a) fractional optimal rewards, (b) normalized integral/fractional rewards, and (c) key algorithm parameters (e.g., $\eta$). We report rewards as average cumulative values at $t \in \{T/3,2T/3,T\}$ (time slots). For experiments with varying seeds (e.g., `python3 main.py * --seed 42`), both the mean and standard deviation of rewards are calculated.
