############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  ADAPTED BY CSM_SEXY_GRP_ - 2025, ORIGIN: SOPHIA BAUM - 2024


### PARAMETERS ###

input_folder  = './input/'               # Folder with parameters
output_folder = './results/'             # Folder to write results to
losses        = './evaluation/'          # Folder to store the refined results to


### IMPORT ###

import pandas as pd
from pandas import IndexSlice as idx

import scipy.io as io
import scipy.sparse as sprs

import numpy as np


### LOADING DATA ###

X_base  = pd.read_csv(output_folder + 'base.csv',                 index_col = [0, 1], header = [0])
XS_comp = pd.read_csv(output_folder + 'combined_shocks_comp.csv', index_col = [0, 1], header = [0])


### COMPUTATIONS ###

x_i = XS_comp['combined_shocks'] # Extract values to transform

XS_comp['absolute_losses'] = (X_base['base'] - x_i).fillna(0)                       # Absolute loss calculation
XS_comp['relative_losses'] = (1 - x_i / X_base['base']).fillna(0).clip(lower = -1)  # Realative loss calculation and Manipulation

XS_comp.to_csv(losses + 'Losses-combined_shocks_comp.csv')
