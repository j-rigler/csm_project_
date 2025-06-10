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
from shapely import wkt


# In[45]:


pd.set_option('display.float_format', '{:.2f}'.format) #avoid scientific notation


# In[46]:


gdf = gpd.read_file("data/world-administrative-boundaries/world-administrative-boundaries.shp")
print(gdf.crs)



# In[48]:

#fao_all_Norm =pd.read_csv("data/fao_data/ALL_norm/Production_Crops_Livestock_E_All_Data_(Normalized).csv")
fao_all_Norm = pd.read_csv("data/fao_data/fao_all_with_coords.csv")

fao_all_Norm = fao_all_Norm[fao_all_Norm['Year']>1992]
fao_all_Norm = fao_all_Norm[fao_all_Norm['Element'] == 'Production']
# In[49]:


fao_all_NormColumns = pd.DataFrame(fao_all_Norm.columns)
print(str(fao_all_Norm))

# # In[55]: Geocoding the data


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

# # Create a 'Geometry' column by combining Latitude and Longitude into Point objects
# # We'll use a lambda function to apply this row-wise, handling NaN values gracefully.
# fao_all_Norm['Geometry'] = fao_all_Norm.apply(
#     lambda row: Point(row['Longitude'], row['Latitude']) if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']) else None,
#     axis=1
# )

# fao_all_Norm.to_csv("data/fao_data/fao_all_with_coords.csv", index=False)


#calculate absolute difference per year in each sector
fao_all_Norm['Absolute Difference'] = fao_all_Norm.groupby(['Area', 'Item', 'Element'])['Value'].diff().fillna(0)

#calculate relative difference per year in each sector
fao_all_Norm['Relative Difference'] = (
    fao_all_Norm.groupby(['Area', 'Item', 'Element'])['Value']
    .transform(lambda x: x.diff() / x.shift().replace(0, np.nan)) # Prevent division by zero
)

fao_all_Norm.to_csv("data/fao_data/fao_all_with_coords.csv", index=False)


# In[57]:


str(fao_all_Norm['Geometry'])

# Function to safely convert WKT strings to shapely Point objects
def parse_wkt_to_point(wkt_string):
    if pd.isna(wkt_string): # Handle NaN values (missing geometries)
        return None
    if not isinstance(wkt_string, str):
        # If it's already a shapely object or not a string, return as is.
        # This handles cases where the column might already be correctly parsed.
        return wkt_string

    cleaned_wkt = wkt_string.strip()
    try:
        geom = wkt.loads(cleaned_wkt)
        if isinstance(geom, Point):
            return geom
        else:
            # If it's valid WKT but not a Point (e.g., LINESTRING, POLYGON),
            # you might want to log a warning or return None.
            print(f"Warning: WKT string '{wkt_string}' parsed into a {type(geom).__name__}, not a Point. Returning None.")
            return None
    except Exception as e:
        # Catch any errors during WKT parsing (e.g., malformed strings)
        print(f"Error parsing WKT string '{wkt_string}': {e}. Returning None.")
        return None

# Apply this conversion to your 'Geometry' column (uppercase 'G')
# and store the resulting shapely Point objects in a new 'geometry' column (lowercase 'g').
# This is the column that GeoPandas will recognize.
fao_all_Norm['geometry'] = fao_all_Norm['Geometry'].apply(parse_wkt_to_point)

fao_all_Norm.to_csv("data/fao_data/fao_all_with_coords.csv", index=False)


# Create GeoDataFrame
fao_all_Norm_geo = gpd.GeoDataFrame(fao_all_Norm, geometry='geometry', crs="EPSG:4326")


# In[58]:


fig, ax = plt.subplots(figsize=(15, 10))
#boundary layer
gdf.plot(ax=ax, color='lightgrey', edgecolor='black', alpha = 0.5)
# Plot the disaster data layer
fao_all_Norm_geo.plot(ax=ax, color='red', markersize=1)

# Customize and show the plot
plt.title("Disaster Data with Boundaries")
plt.show()


# In[59]:

fao_all_Norm_geo.to_file("data/fao_data/filtered_event_geocoded.shp")

