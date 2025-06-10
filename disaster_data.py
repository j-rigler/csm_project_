#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
from geopandas import GeoDataFrame
import ast


# In[45]:


pd.set_option('display.float_format', '{:.2f}'.format) #avoid scientific notation


# In[46]:


gdf = gpd.read_file("data/world-administrative-boundaries/world-administrative-boundaries.shp")
print(gdf.crs)


# In[47]:
fao_disaster_HornofAfrica = pd.read_csv("data/fao_data/Horn of Africa 2019-2023_FAOSTAT_data_en_6-3-2025(1).csv")
fao_disaster_Pakistan = pd.read_csv("data/fao_data/Pakistan 2021-2022_FAOSTAT_data_en_6-3-2025(2).csv")
fao_disaster_Ururguay = pd.read_csv("data/fao_data/Uruguay 2021-2023_FAOSTAT_data_en_6-3-2025(1).csv")


# In[48]:

fao_all_Norm =pd.read_csv("data/fao_data/fao_all_with_coords.csv")


# In[49]:


fao_all_NormColumns = pd.DataFrame(fao_all_Norm.columns)
print(str(fao_all_Norm))

# In[55]: Geocoding the data


# print("Original DataFrame head (before geocoding):")
# print(fao_all_Norm[['Area', 'Value']].head())
# print("\nUnique Areas to geocode:")
# print(fao_all_Norm['Area'].unique()[:10]) # Print first 10 unique areas for brevity


# geolocator = Nominatim(user_agent="geocoder_blabla")
# geocode = RateLimiter(geolocator.geocode, min_delay_seconds=5)
# unique_areas = fao_all_Norm['Area'].unique()

# area_coordinates = {}
# print("\nStarting geocoding process...")
# for area in unique_areas:
#     try:
#         location = geocode(area)
#         if location:
#             area_coordinates[area] = (location.latitude, location.longitude)
#             print(f"Geocoded '{area}': ({location.latitude}, {location.longitude})")
#         else:
#             area_coordinates[area] = (np.nan, np.nan) # Store NaN if not found
#             print(f"Could not geocode '{area}'")
#     except Exception as e:
#         area_coordinates[area] = (np.nan, np.nan)
#         print(f"Error geocoding '{area}': {e}")
#         # Consider adding a longer sleep here or break if many errors occur

# print("\nGeocoding complete.")

# lat_map = {area: coords[0] for area, coords in area_coordinates.items()}
# lon_map = {area: coords[1] for area, coords in area_coordinates.items()}

# # Map the coordinates back to the original DataFrame
# fao_all_Norm['Latitude'] = fao_all_Norm['Area'].map(lat_map)
# fao_all_Norm['Longitude'] = fao_all_Norm['Area'].map(lon_map)

# print("\nDataFrame with new Latitude and Longitude columns:")
# print(fao_all_Norm[['Area', 'Latitude', 'Longitude']].head())

# # To check if all areas got coordinates (or NaN for missing ones)
# print("\nNumber of unique areas geocoded successfully:",
#       fao_all_Norm['Area'].nunique() - fao_all_Norm['Latitude'].isnull().sum())
# print("Number of unique areas failed to geocode:", fao_all_Norm['Latitude'].isnull().sum())


#calculate absolute difference per year in each sector
fao_all_Norm['Absolute Difference'] = fao_all_Norm.groupby(['Area', 'Item', 'Element'])['Value'].diff().fillna(0)

#calculate relative difference per year in each sector
fao_all_Norm['Relative Difference'] = (
    fao_all_Norm.groupby(['Area', 'Item', 'Element'])['Value']
    .transform(lambda x: x.diff() / x.shift().replace(0, np.nan)) # Prevent division by zero
)

fao_all_Norm.to_csv("data/fao_data/fao_all_with_coords.csv", index=False)



# In[57]:


str(filtered_event['Geometry'])

# Convert string → tuple
filtered_event['Geometry'] = filtered_event['Geometry'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Convert tuple (lat, lon) → Point(lon, lat)
filtered_event['geometry'] = filtered_event['Geometry'].apply(lambda coords: Point(coords[1], coords[0]) if coords != (None, None) else None)

# Create GeoDataFrame
fitlered_event_geo = gpd.GeoDataFrame(filtered_event, geometry='geometry', crs="EPSG:4326")


# In[58]:


fig, ax = plt.subplots(figsize=(15, 10))
#boundary layer
gdf.plot(ax=ax, color='lightgrey', edgecolor='black', alpha = 0.5)
# Plot the disaster data layer
fitlered_event_geo.plot(ax=ax, color='red', markersize=1)

# Customize and show the plot
plt.title("Disaster Data with Boundaries")
plt.show()


# In[59]:


fitlered_event_geo.to_file("data/fao_data/filtered_event_geocoded.shp")

