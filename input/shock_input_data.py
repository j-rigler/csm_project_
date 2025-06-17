############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  ADD ON BY CSM_SEXY_GRP_ -  2025, ORIGIN: SOPHIA BAUM # -  2024


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

shock_sectors_HOA = [('Kenya', 'Sugar cane'), # - 36.8610650242051 #2023 # -  worst year 
                     ('Ethiopia', 'Sorghum and products'), # - 21.332861080057995 #2021 # -  worst year
                     ('Kenya', 'Tea (including mate)'), # - 20.298360655737703 #2021 # -  worst year
                     ('Kenya', 'Tomatoes and products'), # - 32.87920997090733 #2021 # -  worst year
                     ('Ethiopia', 'Milk - Excluding Butter'), # - 17.62077684309846 #2021 # -  worst year
                     ('Ethiopia', 'Sugar cane'), # - 39.400315180764856 #2021 # -  worst year
                     ('Ethiopia', 'Sweet potatoes'), # - 42.84694759881593 #2021 # -  worst year
                     ('Kenya', 'Pineapples and products'), # - 61.45627225959635 #2023 # -  worst year
                     ('Ethiopia', 'Millet and products'), # - 22.680840873397766 #2022 # -  worst year
                     ('Kenya', 'Sorghum and products'), # - 55.904444444444444 #2021 # -  worst year
                     ('Kenya', 'Eggs'), # - 95.9267938846111 #2022 # -  worst year
                     ('Kenya', 'Wheat and products'), # - 16.057673530496675 #2023 # -  worst year
                     ('Ethiopia', 'Groundnuts'), # - 32.081281073435655 #2021 # -  worst year
                     ('Kenya', 'Millet and products'), # - 61.3202614379085 #2021 # -  worst year
                     ('Ethiopia', 'Rice and products'), # - 25.881845684577364 #2022 # -  worst year
                     ('Ethiopia', 'Sesame seed'), # - 47.04977525538828 #2021 # -  worst year
                     ('Ethiopia', 'Honey'), # - 59.7574651394807 #2021 # -  worst year
                     ('Ethiopia', 'Onions'), # - 39.990983909746625 #2021 # -  worst year
                     ('Kenya', 'Lemons, Limes and products'), # - 30.93136991085126 #2021 # -  worst year
                     ('Kenya', 'Onions'), # - 16.75193564238785 #2021 # -  worst year
                     ('Kenya', 'Coconuts - Incl Copra'), # - 21.323843545762777 #2021 # -  worst year
                     ('Djibouti', 'Beans'), # - 25.0 #2021 # -  worst year
                     ('Somalia', 'Bovine Meat')] # - 2.1653381604079702 #2023 # -  worst year

phi_0_HOA = [36.8610650242051, 21.332861080057995, 20.298360655737703, 32.87920997090733, 17.62077684309846, 39.400315180764856, 42.84694759881593, 61.45627225959635, 22.680840873397766, 55.904444444444444, 95.9267938846111,
             16.057673530496675, 32.081281073435655, 61.3202614379085, 25.881845684577364, 47.04977525538828, 59.7574651394807, 39.990983909746625, 30.93136991085126,
             16.75193564238785, 21.323843545762777, 25.0, 2.1653381604079702] #HOA values averages per year 2021# - 2023 

phi_0_HOA = [np.round(value*0.01, 4) for value in phi_0_HOA]  

shock_sectors_URU = [('Uruguay', 'Soyabeans'),                  #0.7667 # -  2022 worst year
                     ('Uruguay', 'Maize and products'),         #0.0442 # -  2021 worst year
                     ('Uruguay', 'Milk - Excluding Butter'), #0.0478 # -  2022 worst year
                     ('Uruguay', 'Sorghum and products'),       #0.7157 # -  2022 worst year
                     ('Uruguay', 'Lemons, Limes and products'), #0.2746 # -  2022 worst year
                     ('Uruguay', 'Rice and products'),          #0.0180 # -  2023 worst year
                     ('Uruguay', 'Oranges, Mandarines')]        #0.0728 # -  2023 worst year

phi_0_URU = [0.7667, 0.0442, 0.0478, 0.7157, 0.2746, 0.0180, 0.0728] #Uruguay values averages per year 2021# - 2023

# Transform lists into dictionaries
shock_dict_PAK = dict(zip(shock_sectors_PAK, phi_0_PAK))
shock_dict_RUS = dict(zip(shock_sectors_RUS, phi_0_RUS))
shock_dict_HOA = dict(zip(shock_sectors_HOA, phi_0_HOA))
shock_dict_URU = dict(zip(shock_sectors_URU, phi_0_URU))

#shock intensity curve constuction
mu = - 0.5                                                # Assumption
def phi(phi_0, mu, t):                                      
    return np.round(phi_0 * np.exp(mu * t), 2)            # Assumed exponential decay of shock intensity over time