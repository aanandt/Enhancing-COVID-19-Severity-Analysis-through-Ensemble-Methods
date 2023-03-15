# Enhancing-COVID-19-Severity-Analysis-through-Ensemble-Methods
 In this study, we propose a domain knowledge-based pipeline for extracting regions of infection in COVID-19 patients using a combination of image-processing algorithms and a pre-trained UNET model. The severity of the infection is then classified into different categories using an ensemble of three machine-learning models: Extreme Gradient Boosting, Extremely Randomized Trees, and Support Vector Machine. The proposed system was evaluated on a validation dataset in the AI-Enabled Medical Image Analysis Workshop and COVID-19 Diagnosis Competition (AI-MIA-COV19D) and achieved a macro F1 score of 64\%. These results demonstrate the potential of combining domain knowledge with machine learning techniques for accurate COVID-19 diagnosis using CT scans.

For the virtual environment: **pip install -r requirements.txt**

The pre-trained UNET model takes long time to generate the lung mask for the whole train, validation, and test datasets. 

The pre-trained UNET mask for the train data can be found [here](https://drive.google.com/drive/folders/17kwmu5-Xi3WAPLjSK06ACwXBL5st8vC2?usp=sharing).\\
The pre-trained UNET mask for the validation data can be found [here](https://drive.google.com/drive/folders/1Znx_NnX7xxxIY3aejT1OuLDT5MdzKlg9?usp=sharing).
The pre-trained UNET mask for the test data can be found [here](https://drive.google.com/drive/folders/1Ix2uhWO8_Hq200Uf2EhOCLBRn_dKcwi4?usp=sharing).
