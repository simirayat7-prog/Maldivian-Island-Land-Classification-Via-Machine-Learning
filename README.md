# Cross-Island Land Cover Classification of Maldivian Islands Using Random Forest and Sentinel-2 Imagery
This project applies machine learning techniques to classify land cover types on Maldivian islands using satellite imagery. The aim is to distinguish between different surface types such as water, vegetation, urban areas, and sand.

In this READ ME: you will find an introduction to the project consisting of a problem description, details about data retrieval, image preprocessing, preparation for machine learning model application and a methodology justifying & evaluating the approach taken so far.

1.	PROBLEM DESCRIPTION 
2.	REMOTE SENSING TECHNIQUE 
3.	STUDY AREA
4.	DATA SOURCE
5.	IMAGE PREPROCESSING 
6.	CLASS DESCRIPTIONS
7.	MACHINE LEARNING BEGINS 
8.	LIMITATIONS OF METHODOLOGY

   
 <img width="759" height="526" alt="image" src="https://github.com/user-attachments/assets/e2c6bc27-336d-4acc-b629-a608c75a3468" />
*An image taken from (MULHERN, 2020) illustrating projected sea level rise at 2100 in the Maldives’ capital Male and its surrounding islands where 80% of the population reside. This highlights the importance of satellite data in monitoring the islands to see how they interact with changes associated with anthropogenic warming.*

## 1.	Problem Description: 
Small island environments like the Maldives are highly vulnerable to environmental change, including coastal erosion, urban expansion, and reef degradation. Accurate land cover classification is therefore important for environmental monitoring and management. Here we test the usability of Machine Learning Models (Specifically Random Forest Classifiers) on classifying these landscapes for ease of future monitoring. 


## 2.	Remote Sensing Technique:
Satellite sensors such as Sentinel-2 and Sentinel-3 are capable of capturing data across multiple spectral bands (e.g. 13–21 bands), which provide detailed information about surface properties.
However, in this project, only RGB imagery (3 bands) was used. This represents a simplified subset of the available spectral information. Despite this limitation, the model was still able to classify major land cover types effectively, as these classes exhibit distinct visual characteristics in the visible spectrum.


## 3.	Study Area:
The three islands: Fuvahmulah, Hon’daafushi and Kaashidhoo were all hand selected due to their diversity in land cover and usefulness in exploring the advantages and shortcomings of IRIS in image classification.

 <img width="853" height="550" alt="image" src="https://github.com/user-attachments/assets/36bc4a02-ffb1-48fa-b353-cd5398528960" />

*Figure 1: displays locations of each island across the Maldives. Island where chosen to be representative of typical Maldivian Island hence wide distribution.*

**Fuvahmulah:** Has distinguishable classes but has very small details that low resolution can miss. At low resolutions building colours fade to appear like sand.

**Hon’daafushi:** Has more of a natural environment and classes should be distinguishable despite lower resolutions.

**Kaashidhoo:** has a complex blend of all classes and algae in water vs vegetation at times could be difficult to differentiate due to their similar colours. Additionally building colours do resemble shades of sand once again.


## 4.	Data Source: 
Data was collected from the Copernicus Browser website where satellite imagery is derived from Sentinel 2 satellite. The images where initially downloaded in highest resolution possible but where later trimmed down for ease of imputing into IRIS (see next paragraph). All 3 islands images (RGB) where sourced from the same date: 24/02/2026 and at times with 0% cloud cover to ensure accurate classification. 


## 5.	Image Preprocessing 
Original Copernicus images were 2048 x 2048 but I resized to 512 x 512 for ease of classification in IRIS, crashes occurred with larger image sizes. However, this would have reduced number of pixels per image from about 4 million to 250,000. Reduced resolution would reduce accuracy of classification and resolution of masks. It was also noticeably more difficult to manually classify.
On the positive side, the very low sized images meant no further preprocessing or chunking of the satellite data was required before Machine Learning model training in Python!

 <img width="940" height="532" alt="image" src="https://github.com/user-attachments/assets/f9b536b2-27bd-4cb5-b79f-c1bc1e1bd8f0" />

*Figure 2: displaying changes made to original data prior to IRIS classification.*


## 6.	Class descriptions: 
Images were classified into the 5 classes using IRIS. For each image I manually classified approx.34,000 pixels and then allowed the AI feature to classify the rest of the image and produce masks in .png format. This was to ensure all images had a roughly equal chance at a quality classification prior to machine learning and to display their inherent unique difficulties in classification.  
(Config.json file as well as raw and reduced size satellite data can be found in repository folder)

 <img width="940" height="412" alt="image" src="https://github.com/user-attachments/assets/9ecd8786-034b-41b9-aa45-b917d34eca6b" />

*Figure 3: The images display my manual classification drawings and despite having a similar number of manually classified pixels each island has a different AI classification score. Fuvahmulah has the lowest at 48% whereas Hon’daafushi has the highest at 92%. These two were significantly different to classify as Fuvahmulha’s Urban area colour closely resembles its sand which I believe confused the algorithm. On the other hand, Hon’daafushi’s classes where easily distinguishable to the naked eye explaining it high score. Finally, Kaashidhoo has a median AI score of 70% and the greatest challenge here was also differentiating urban area and sand as well as algae rich lagoon waters and vegetation. These could be easily put into context by a human however clearly not so easily by the algorithm.*

 <img width="940" height="546" alt="image" src="https://github.com/user-attachments/assets/276c0c3a-4015-4078-bed8-d297ec915814" />
*Figure 4: displaying masks for each island created by IRIS and used for Machine Learning later.*


## 7.	Machine Learning Begins: 
A Random Forest classifier was selected due to its robustness, ability to handle high-dimensional feature spaces, and strong performance on relatively small datasets, where more complex models such as Convolutional Neural Networks (CNNs) may be prone to overfitting.
In this study, Sentinel-2 satellite imagery from three Maldivian islands was pre-processed and manually labelled using IRIS to generate segmentation masks. These masks provided ground truth data for supervised learning. Each pixel was represented using a 3×3 neighbourhood, incorporating local spatial context and resulting in 27 input features (3×3×3 RGB values).
To evaluate model generalisation, a cross-island validation approach was implemented. The model was trained on labelled data from two islands and tested on a third unseen island. This process was repeated for all island combinations, allowing for a systematic assessment of performance across different spatial environments.
Additional experiments were conducted to compare pixel-based (1×1) and patch-based (3×3) feature representations, as well as to evaluate the performance of Random Forest and Extra Trees classifiers. Model performance was assessed using accuracy, F1-scores, confusion matrices, and visual comparisons of predicted and ground truth maps.


## 8.	Limitations of Methodology: 

**Data set:** The dataset uses only three bands (RGB) of the 21 available from Sentinel 2’s data range which limits the model’s ability to distinguish between classes with similar visual appearance (e.g. sand vs urban very common issue). Additionally, only three Maldivian Islands were chosen limiting diversity of training data and reducing the model’s ability to generalise to unseen environments.
The preprocessing and JPG format of the raw satellite data would have significantly reduced resolution perhaps hindering the models ability to classify as well. This is a 16 fold decrease in pixel count which does reduce computational power however ideally should not be done! Later studies on higher memory computer can redo this experiment to improve our understanding of ML models.

**Classes:** Some land cover classes (e.g. water) dominate the dataset, while others (e.g. urban or sand) are underrepresented. This can bias the model towards majority classes.

**Model approach:** A Random Forest classifier was used, which does not fully exploit spatial structure in images compared to deep learning approaches such as Convolutional Neural Networks. Only a 3×3-pixel neighbourhood was used to represent each data point. This captures very local information but fails to consider larger spatial patterns such as coastline structure or urban layout. As well as this, each pixel patch is treated independently, ignoring relationships between neighbouring predictions. This can lead to noisy or fragmented classification outputs.

**Now it’s time to take a look at the notebook! There you will find the python code used to train our Random forest Model as well as critical results and evaluation!**

