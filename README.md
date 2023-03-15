# Enhancing-COVID-19-Severity-Analysis-through-Ensemble-Methods
 In this study, we propose a domain knowledge-based pipeline for extracting regions of infection in COVID-19 patients using a combination of image-processing algorithms and a pre-trained UNET model. The severity of the infection is then classified into different categories using an ensemble of three machine-learning models: Extreme Gradient Boosting, Extremely Randomized Trees, and Support Vector Machine. The proposed system was evaluated on a validation dataset in the AI-Enabled Medical Image Analysis Workshop and COVID-19 Diagnosis Competition (AI-MIA-COV19D) and achieved a macro F1 score of 64\%. These results demonstrate the potential of combining domain knowledge with machine learning techniques for accurate COVID-19 diagnosis using CT scans.

0. For the virtual environment: **pip install -r requirements.txt**

1. Download the datasets (train, validation, and test) and the .xlsx files (groundtruth) to the **Dataset** folder.

2. The pre-trained UNET model takes long time to generate the lung mask for the whole train, validation, and test datasets. 

- The pre-trained UNET mask for the train data can be found [here](https://drive.google.com/drive/folders/17kwmu5-Xi3WAPLjSK06ACwXBL5st8vC2?usp=sharing).
* The pre-trained UNET mask for the validation data can be found [here](https://drive.google.com/drive/folders/1Znx_NnX7xxxIY3aejT1OuLDT5MdzKlg9?usp=sharing).
+ The pre-trained UNET mask for the test data can be found [here](https://drive.google.com/drive/folders/1Ix2uhWO8_Hq200Uf2EhOCLBRn_dKcwi4?usp=sharing).

3. Feature level representation is generated from the pre-processed image and can be download from the following [link](https://drive.google.com/drive/folders/1uzww47_Iuj_V_h1cs8e--hVgbkjCauYU?usp=sharing).
4. Run the following command to train the models.
    - python Src/Train.py
    - If any new dataset is used then the istructions are provided in the Train.py file.
5. Run the following command to check the performance on the trained models in the validation dataset.
    - python Src/Validation.py
    - If any new dataset is used then the istructions are provided in the Validation.py file.
6. Run the following command to check the performance on the trained models in the challenge test dataset.
    - python Src/Test.py
    - If any new dataset is used then the istructions are provided in the Test.py file.
