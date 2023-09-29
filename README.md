# ML_Fungi_Binary_Classification_Xception

This GitHub repository is dedicated to the research project titled "Evaluating the Feasibility of Machine Learning-Based Fungal Identification Based on Growth Characteristics for Rapid and Cost-Efficient Quality Control in the Biopharmaceutical Industry: A Comparative Analysis of Transfer Learning and DeepNetts Platform."

This research aims to explore the feasibility of using digital imaging techniques combined with machine learning for the rapid identification of fungal species. The model created in this study is a binary classification model. The Xception CNN pre-trained model was utilized as the feature extractor from fungi images for the transfer learning method.

To train these machine learning models, four different fungal species - A.  fumigatus, A. niger, P. chrysogenum, and R. stolonifer were cultivated under varying incubation conditions. A. fumigatus was designated as positive data while other fungal species were utilized as negative data for this study. Colony images were captured using a 5-megapixel digital camera.

The model achieved a high overall accuracy of 0.981 and high precision score of 0.964. 


## This repository contains:

- Scripts: These scripts were used in the research for the implementation of transfer learning method using Xception pre-trained model for fungi classification. They cover the entire process, from feature extraction to making predictions and conducting analyses. Additionally, optimization techniques are provided used to enhance model performance.

- Anaconda Environment: The environment used for conducting this study

- Datasets: The datasets used for training, validation, and testing of the model. The labelling system used to label images can be found in the 'Labelling_System.txt'.

- Model's predictions on fungi images are available in the 'predictions.txt' file.



