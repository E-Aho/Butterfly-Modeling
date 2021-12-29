# Butterfly Project

Repo containing the code and data used for the Butterfly MLiS Part 1 module.

To install, make sure you have a compatible version of Python (< 3.9), and run the following command:

`pip install -r requirements.txt`

## Help
To check the version of Python you are using, run 
`python --version`

## Datasources:
### GIS data
Data was computed using open source GIS package QGIS, and the following datasources:

#### Environmental
* For country borders, [Natural Earth Data](http://www.naturalearthdata.com/downloads/)
* For net radiation, the [NASA Earth Observations CERES Data](https://neo.gsfc.nasa.gov/view.php?datasetId=CERES_NETFLUX_M), Dec 2020 to Nov 2021
* For Solar Insolation, the [NASA Earth Observations CERES Data for FLASHFlux](https://neo.gsfc.nasa.gov/view.php?datasetId=CERES_NETFLUX_M), Dec 2020 to Nov 2021
* For Temperature (Day/Night), the [NASA Earth Observation MODIS Data](https://neo.gsfc.nasa.gov/view.php?datasetId=MOD_LSTD_CLIM_M), Dec 2020 to Nov 2021
* For Land coverage data, [NASA Earth Observations LP DAAC MCD12Q1 Data](https://neo.gsfc.nasa.gov/view.php?datasetId=MCD12C1_T1)
* For rainfall data, [ArcGIS Average Annual Rainfall](https://hub.arcgis.com/datasets/fasgis::average-annual-rainfall/about)
* Bioclimatic data is from [WorldClim](https://www.worldclim.org/data/worldclim21.html)
* Elevation is from [NASA EROS STRM Plus](https://www2.jpl.nasa.gov/srtm/)
* Polygons for administrative borders is from [GADM (used for Brazil, China, Mexico, and Malaysia)](https://gadm.org/download_country.html)
* Polygon for Taman Kinabalu is from [ProtectedPlanet](https://www.protectedplanet.net/785) 

Citations:
Taman Kinabalu: UNEP-WCMC and IUCN (2022), Protected Planet: The World Database on Protected Areas (WDPA) and World Database on Other Effective Area-based Conservation Measures (WD-OECM) [Online], January 2022, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.
Bioclimatic indicators: [Reference article](https://www.nature.com/articles/s41597-020-00726-5#Sec2)
