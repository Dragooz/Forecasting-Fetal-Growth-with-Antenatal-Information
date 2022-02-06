# Forecasting Fetal Growth with Antenatal Information using Machine Learning Techniques
There's two part in this project:
- Train and develop Machine Learning models
- Build and deploy a web application

## Train and develop Machine Learning models

### Datasets
In this project, I'm working with antenatal information of singleton pregnant women, from 2 different countries, which are Singapore and Malaysia. Below are the flow chart of how I handled those 2 dataset.

#### Data Handling on Singapore dataset.
![singapore dataset flow drawiov6](https://user-images.githubusercontent.com/47239545/146946758-67fa536f-2e0b-4156-89ef-e47fde4efe30.png)

#### Data Handling on Malaysia dataset.
![malaysia dataset flow drawiov8](https://user-images.githubusercontent.com/47239545/146946820-91a90e6f-2bf6-4293-bb23-22fbb7c8da33.png)

### Training + Deploying Machine Learning model 
There's total of 4 machine learning models outcome here, 2 models for each country.

- Machine Learning model 1 - Prediction of estimated fetal weight range from week 28 to week 34 using antenatal information before week 28 collected from Singapore.
- Machine Learning model 2 - Prediction of estimated fetal weight from week 35 onwards using antenatal information before week 35 collected from Singapore.
- Machine Learning model 3 - Prediction of estimated fetal weight range from week 28 to week 34 using antenatal information before week 28 collected from Malaysia.
- Machine Learning model 4 - Prediction of estimated fetal weight from week 35 onwards using antenatal information before week 35 collected from Malaysia.

### Web system
https://fetal-weight-prediction.herokuapp.com/

### Conclusion

#### Summarized table
![models' summary](https://user-images.githubusercontent.com/47239545/146949809-ec397605-ae2b-48d4-9c1c-6caab49aa6ee.png)

From the table above, we can observe that:
- The ML models that predict the estimated fetal weight ranging from week 28 to week 34 picks GA(week)\_32 and AC\_22 as the common features in both countries.
- The ML models that predict the estimated fetal weight from week 35 onwards picks GA(week)\_35, GA(week)\_32, FL\_32, and AC\_32 as the common features in both countries.
- The ML models trained using Singapore higher priority as it's RMSE and MAPE are lower.

For your information: [https://www.statology.org/rmse-vs-r-squared/]
- The RMSE value tells us that the average deviation between the predicted EFW made by the model and the actual EFW.
- The R2 value tells us that the predictor variables in the model are able to explain 85.6% of the variation in the EFW.

Hence, in this project, I will pick to prioritizes RMSE over R2 in the case of deploying out the model, by minimizing the deviation of the predicted value with the actual one.

#### More details

For more details, kindly refer to the notebook uploaded. 
- FYP.ipynb focuses on Singapore dataset.
- FYP Study.ipynb focuses on Malaysia dataset.

## Build and deploy a web application
https://fetal-weight-prediction.herokuapp.com/

### Homepage
![homepage](https://user-images.githubusercontent.com/47239545/146952732-56faf3c2-a9c9-4878-9112-e3ba4056fb9d.png)

### Predictions
<img width="912" alt="malay - 28-34" src="https://user-images.githubusercontent.com/47239545/152687567-423f6df4-0614-4513-b784-b82092cda4e9.png">
<img width="903" alt="malay - 35onwards" src="https://user-images.githubusercontent.com/47239545/152687569-2bfccd71-32ef-4575-b84d-ca6d4f97141d.png">
<img width="909" alt="sing - 28-34" src="https://user-images.githubusercontent.com/47239545/152687571-11c4e3b4-f290-4d7c-9486-db134ea5023c.png">
<img width="904" alt="sing - 35onwards" src="https://user-images.githubusercontent.com/47239545/152687575-e66098bb-3a79-4fab-b899-523f688a85a4.png">

### FAQ
<img width="898" alt="faq" src="https://user-images.githubusercontent.com/47239545/152687589-78554538-ceae-48bc-aac2-de154bcc520e.png">

### Inquiry
<img width="949" alt="footer inquiry feedback" src="https://user-images.githubusercontent.com/47239545/152687602-9e397f06-4c9f-4a2c-a8bf-670f9d28b69f.png">

