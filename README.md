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
* Polygon for Taman Kinabalu is from [ProtectedPlanet 785](https://www.protectedplanet.net/785) 
* Polygon for Region de Calakmul is from [ProtectedPlanet 61401](https://www.protectedplanet.net/61401)

### Citations:

Taman Kinabalu: UNEP-WCMC and IUCN (2022), Protected Planet: The World Database on Protected Areas (WDPA) and World Database on Other Effective Area-based Conservation Measures (WD-OECM) [Online], January 2022, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.

Region de Calakmul: UNEP-WCMC and IUCN (2022), Protected Planet: The World Database on Protected Areas (WDPA) and World Database on Other Effective Area-based Conservation Measures (WD-OECM) [Online], January 2022, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.

Bioclimatic indicators: [Reference article](https://www.nature.com/articles/s41597-020-00726-5#Sec2)

#### Papers:

* 1.Fleishman, E., Austin, G. & Weiss, A. An Empirical Test of Rapoport’s Rule: Elevational Gradients in Montane Butterfly Communities. Ecology 79, 2482–2493 (1998).
* 2.Park, Y.-S., Céréghino, R., Compin, A. & Lek, S. Applications of artificial neural networks for patterning and predicting aquatic insect species richness in running waters. Ecological Modelling 160, 265–280 (2003).
* 3.Dallimer, M. et al. Biodiversity and the Feel-Good Factor: Understanding Associations between Self-Reported Human Well-being and Species Richness. BioScience 62, 47–55 (2012).
* 4.Stefanescu, C., Herrando, S. & Páramo, F. Butterfly species richness in the north-west Mediterranean Basin: the role of natural and human-induced factors: Butterfly species richness in the north-west Mediterranean Basin. Journal of Biogeography 31, 905–915 (2004).
* 5.Parmesan, C. Climate and Species’ Range. Nature 382, 765–766 (1996).
* 6.Brown, G., Pocock, A., Zhao, M.-J. & Lujan, M. Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection. 40.
* 7.Munguira, M. L. Conservation of butterfly habitats and diversity in European Mediterranean countries. in Ecology and Conservation of Butterflies (ed. Pullin, A. S.) 277–289 (Springer Netherlands, 1995). doi:10.1007/978-94-011-1282-6_19.
* 8.Hawkins, B. A. et al. Energy, Water, and Broad-Scale Geographic Patterns of Species Richness. Ecology 84, 3105–3117 (2003).
* 9.Global patterns and predictors of tropical reef fish species richness. https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1600-0587.2013.00291.x.
* 10.Hill, J. et al. Responses of butterflies to twentieth century climate warming: Implications for future ranges. Proceedings. Biological sciences / The Royal Society 269, 2163–71 (2002).
* 11.Würtz, P. & Annila, A. Roots of Diversity Relations. J Biophys 2008, 654672 (2008).
* 12.Parra, H. N. & Ranz, R. R. Taxocenotic and Biocenotic Study of Lepidoptera (Rhopalocera) in Rucamanque: A Forest Remnant in the Central Valley of the Araucanía Region, Chile. Lepidoptera (IntechOpen, 2017). doi:10.5772/intechopen.70924.
* 13.Preston, F. W. The Canonical Distribution of Commonness and Rarity: Part I. Ecology 43, 185 (1962).
* 14.SANCHEZ-RODRIGUEZ & Baz, A. The effects of elevation on the butterfly communities of a Mediterranean mountain, Sierra de Javalambre, central Spain. Journal of the Lepidopterist’ Society, 49, 192–207 (1995).
* 15.The imprint of the geographical, evolutionary and ecological context …. archive.ph http://archive.ph/KkmHi (2013).
* 16.Fraser, R. H. & Currie, D. J. The Species Richness-Energy Hypothesis in a System Where Historical Factors Are Thought to Prevail: Coral Reefs. The American Naturalist 148, 138–159 (1996).
