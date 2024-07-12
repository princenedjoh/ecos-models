import ee
import pandas as pd

# Initialize connection to Google Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-princenedjoh5')
print(ee.String('Hello from the Earth Engine servers!').getInfo())

bands = ['treecover2000', 'loss', 'gain', 'lossyear', 'first_b30', 'first_b40', 'first_b50', 'first_b70', 'last_b30', 'last_b40', 'last_b50', 'last_b70', 'datamask']

def load_vi(product_id):
  vi = ee.ImageCollection(product_id).select('NDVI')
  return vi.mean()

ghana_aoi = ee.FeatureCollection("projects/ee-princenedjoh5/assets/ghana_Ghana_Country_Boundary")

forest_cover = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").clip(ghana_aoi)
treeCover = forest_cover.select(['treecover2000'])

# Using CHIRPS data
precipitation = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filter(ee.Filter.date('2018-05-01', '2018-05-03')).filterBounds(ghana_aoi).select('precipitation');
precipitationVis = {
  'min': 1,
  'max': 17,
  'paletbte': ['001137', '0aab1e', 'e7eb05', 'ff4a2d', 'e90000'],
}

#elevation
elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(ghana_aoi)
slope = ee.Terrain.slope(elevation)
aspect = ee.Terrain.aspect(elevation)

#protected areas
protectedAreas = ee.FeatureCollection('WCMC/WDPA/current/points').style({
    'color': '4285F4',
    'pointSize': 3,
  }).clip(ghana_aoi)

#combined Image
feature_image = ee.Image.cat(forest_cover, slope, aspect)
print('forest cover:', feature_image.bandNames().getInfo())

# Define the class band (e.g., transition)
class_band = 'lossyear'

# Perform stratified sampling
sample = feature_image.stratifiedSample(
    numPoints=100,
    classBand=class_band,
    scale=30,
    region=ghana_aoi,
)

# Convert the Earth Engine feature collection to a pandas DataFrame
sampled_points = sample.getInfo()
df = pd.DataFrame(sampled_points['features'])

# Extract properties from the DataFrame
properties = df['properties'].apply(pd.Series)

# Save properties to a CSV file
csv_filename = 'sampled_points.csv'
properties.to_csv(csv_filename, index=False)

print(f'Saved sampled points to {csv_filename}')