# Plant Disease Classification
This projects includes both back-end (Keras, Flask with dependencies) and front-end/deployement (HTML, CSS, JS) parts. The project perform plant disease classification using VGG-16 with pretraining model. Saved files of inspected models are saved on specific folders in project directory.

The validation accuracy of the model is 87.29%. 

For deploying I decided to use Flask instead of Django because of it simplicity and compatibility with Amazon Web Services (AWS) and Google Cloud Storage (GCS). That works fine. App Engine in GCS is a perfect tool for deploying a Keras model as web application. Do not forget to carefully upload your deployment environment and crate Python YAML file that the server understand what is configuration of your application.

Important part is to understand how to upload an image file to the server (Google bucket) with Flask. By this, I modified permission for app. bucket to let user write and read (when app makes a prediction) files there.

Some additional work with HTML, CSS and Javascript and the model front-end looks bright, simply and user friendly, even compatible with mobile phones.

Link to web application: http://finalproject-bangkit-dps4a.appspot.com/

Link to Kagle dataset: https://www.kaggle.com/vipoooool/new-plant-diseases-dataset

Link to Presentation: https://docs.google.com/presentation/d/151jL5ancaQNPCFNrgH_eZEjYdGuNgeFqI8OO4h4OlaE/edit?usp=sharing
