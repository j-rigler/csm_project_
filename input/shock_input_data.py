############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  ADD ON BY CSM_SEXY_GRP_ - 2025, ORIGIN: SOPHIA BAUM - 2024


### IMPORTS ###

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pandas import IndexSlice as idx

import scipy.io as io
import scipy.sparse as sprs
import numpy as np


### DATA ###

shock_sectors_PAK = [('Pakistan','Rice and products'),
                     ('Pakistan','Cottonseed'),
                     ('Pakistan','Rape and Mustardseed'),
                     ('Pakistan', 'Peas'),
                     ('Pakistan', 'Dates')]
phi_0_PAK = [0.2146, 0.4124, 0.27, 0.29, 0.7274]  

shock_sectors_RUS = [('Russian Federation', 'Wheat and products'),
                     ('Russian Federation', 'Barley and products'),
                     ('Russian Federation', 'Cereals, Other'),
                     ('Russian Federation', 'Maize and products'),
                     ('Russian Federation', 'Oats'),
                     ('Russian Federation', 'Peas'),
                     ('Russian Federation', 'Potatoes and products'),
                     ('Russian Federation', 'Rye and products')]
phi_0_RUS = [0.3276, 0.5330, 0.3985, 0.2216, 0.4039, 0.2344, 0.3208, 0.6225]

shock_sectors_HOA = [('Kenya', 'Sugar cane'),
                     ('Ethiopia', 'Sorghum and products'),
                     ('Kenya', 'Tea (including mate)'),
                     ('Kenya', 'Tomatoes and products'),
                     ('Ethiopia', 'Milk - Excluding Butter'),
                     ('Ethiopia', 'Sugar cane'),
                     ('Ethiopia', 'Sweet potatoes'),
                     ('Kenya', 'Pineapples and products'),
                     ('Ethiopia', 'Millet and products'),
                     ('Kenya', 'Sorghum and products'),
                     ('Kenya', 'Eggs'),
                     ('Kenya', 'Wheat and products'),
                     ('Ethiopia', 'Groundnuts'),
                     ('Kenya', 'Millet and products'),
                     ('Ethiopia', 'Rice and products'),
                     ('Ethiopia', 'Sesame seed'),
                     ('Ethiopia', 'Honey'),
                     ('Ethiopia', 'Onions'),
                     ('Kenya', 'Lemons, Limes and products'),
                     ('Kenya', 'Onions'),
                     ('Kenya', 'Coconuts - Incl Copra'),
                     ('Djibouti', 'Beans'),
                     ('Somalia', 'Bovine Meat')]

phi_0_HOA = [0.1842, 0.1123, 0.1548, 0.4432, 0.0972, 0.1883, 0.19, 0.7051, 0.0973, 0.3685, 0.0883,
             0.2353, 0.4392, 0.4329, 0.2469, 0.2315, 0.3458, 0.1257, 0.4019,
             0.1719, 0.2577, 0.2516, 0.0385] #HOA values averages per year 2021-2023 - need to recheck these values
        
shock_sectors_URU = [('Uruguay', 'Soyabeans'),
                     ('Uruguay', 'Maize and products'),
                     ('Uruguay', 'Milk - Excluding Butter'),
                     ('Uruguay', 'Sorghum and products'),
                     ('Uruguay', 'Lemons, Limes and products'),
                     ('Uruguay', 'Rice and products'),
                     ('Uruguay', 'Oranges, Mandarines')]

phi_0_URU = [0.3795, 0.3455, 0.9296, 0.2178, 0.3683, 0.9790, 0.1889] #Uruguay values averages per year 2021-2023

# Transform lists into dictionaries
shock_dict_PAK = dict(zip(shock_sectors_PAK, phi_0_PAK))
shock_dict_RUS = dict(zip(shock_sectors_RUS, phi_0_RUS))
shock_dict_HOA = dict(zip(shock_sectors_HOA, phi_0_HOA))
shock_dict_URU = dict(zip(shock_sectors_URU, phi_0_URU))

#shock intensity curve constuction
mu = -0.5                                                # Assumption
def phi(phi_0, mu, t):                                      
    return np.round(phi_0 * np.exp(mu * t), 2)           # Assumed exponential decay of shock intensity over time