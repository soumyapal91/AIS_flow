# Population Monte Carlo with Normalizing Flow

This is an implementation of the methodology in "Population Monte Carlo with Normalizing Flow". The arXiv version is available at [https://arxiv.org/pdf/2106.06064.pdf](https://arxiv.org/pdf/2106.06064.pdf)

&nbsp;
&nbsp;
&nbsp;

---
## To reproduce the results in the paper: choose 'example' and 'sigma_prop' for running different experiments with different settings, refer to our paper for the citations of the baseline algorithms 

1. PMC: python run_trials.py --num_trials 100 --example 'GMM'/'Logistic' --sigma_prop 1.0/2.0/3.0 --adaptation 'Resample'
        --resampling 'global' --weighting 'Standard'
   
2. GR-PMC: python run_trials.py --num_trials 100 --example 'GMM'/'Logistic' --sigma_prop 1.0/2.0/3.0 --adaptation 'Resample' 
           --resampling 'global' --weighting 'DM'
   
3. LR-PMC: python run_trials.py --num_trials 100 --example 'GMM'/'Logistic' --sigma_prop 1.0/2.0/3.0 --adaptation 'Resample' 
           --resampling 'local' --weighting 'DM'
   
4. SL-PMC: python run_trials.py --num_trials 100 --example 'GMM'/'Logistic' --sigma_prop 1.0/2.0/3.0 --adaptation 'Langevin' 
           --resampling 'local' --weighting 'DM'
   
5. O-PMC: python run_trials.py --num_trials 100 --example 'GMM'/'Logistic' --sigma_prop 1.0/2.0/3.0 --adaptation 'Newton'
          --resampling 'local' --weighting 'DM'
   
6. GRAMIS: python run_trials.py --num_trials 100 --example 'GMM'/'Logistic' --sigma_prop 1.0/2.0/3.0 --adaptation 'GRAMIS'
           --resampling 'local' --weighting 'DM'
   
7. HAIS (GMM): python run_trials.py --num_trials 100 --example 'GMM' --sigma_prop 1.0/2.0/3.0 --adaptation 'HMC' --                        resampling 'local' --weighting 'DM' --L_hmc 50 --eps_hmc 0.005

8. HAIS (Logistic): python run_trials.py --num_trials 100 --example 'Logistic' --sigma_prop 1.0/2.0/3.0 --adaptation 'HMC' --                   resampling  'local' --weighting 'DM' --L_hmc 50 --eps_hmc 0.05

9. VAPIS (GMM): python run_trials.py --num_trials 100 --example 'GMM' --sigma_prop 1.0/2.0/3.0 --adaptation 'VAPIS' --                        resampling 'local' --weighting 'DM' --lr_vi 0.25

10. VAPIS (Logistic): python run_trials.py --num_trials 100 --example 'Logistic' --sigma_prop 1.0/2.0/3.0 --adaptation                             'VAPIS' --resampling  'local' --weighting 'DM' --lr_vi 0.5

11. NF-PMC (GMM) (ours): python run_trials.py --num_trials 100 --example 'GMM' --sigma_prop 1.0/2.0/3.0 --adaptation 'NF' --                          resampling 'local' --weighting 'DM' --lr_nf 0.005 --step_nf 25 --gamma 0.1

12. NF-PMC (Logistic) (ours): python run_trials.py --num_trials 100 --example 'Logistic' --sigma_prop 1.0/2.0/3.0 --        
                              adaptation 'NF' --resampling 'local' --weighting 'DM' --lr_nf 0.05 --step_nf 25 --gamma 0.1


 ## Calculation of the performance metrics:

 Run python calculate_error.py
   
&nbsp;

Then the results will be saved in the ```./results```.


&nbsp;
&nbsp;
&nbsp;


## Requirements
See requirements.txt

## Cite

Please cite our paper if you use this code in your own work:

```
@article{pal2023, 
author={S. Pal and A. Valkanas and and M. Coates}, 
journal={IEEE Sig. Process. Lett.}, 
title={{Population Monte Carlo} with Normalizing Flow},
month={},
year={2023}}
```
