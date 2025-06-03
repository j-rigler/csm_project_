############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  ADAPTED BY CSM_SEXY_GRP_ 2025, ORIGIN: SOPHIA BAUM - 2024


### PARAMETERS ###

input_folder  = './input/'               # folder with parameters
output_folder = './results/'             # folder to write results to
losses        = './evaluation/'          # folder to store the refined results to


### IMPORT ###

import pandas as pd
from pandas import IndexSlice as idx

import scipy.io as io
import scipy.sparse as sprs

import numpy as np
from loss_calculation_header import*


### LOADING DATA ###

X_base  = pd.read_csv(output_folder + 'base.csv',                 index_col = [0, 1], header = [0])
XS_comp = pd.read_csv(output_folder + 'combined_shocks_comp.csv', index_col = [0, 1], header = [0])


### COMPUTATIONS ###

AL_comp = compute_losses(X_base, XS_comp)
save_calculation(losses, AL_comp)

RL_comp = compute_losses(X_base, XS_comp, relative = True)
save_calculation(losses, RL_comp, relative = True)
