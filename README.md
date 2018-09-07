# candidate_recommendation_model
MSc-project Automatic Candidate Recommendation System<br/>
This GitHub repository contains following:<br/>
1. Skill Entity Extraction System<br/>
skill_ners_models.ipynb	<br/>
skill_ners_preprocessing.ipynb<br/>
sequence_tagging(adapated from LSTM-CRF implemented by guillaumegenthial https://github.com/guillaumegenthial/sequence_tagging)<br/>

2. Interview State Model Prediction Model<br/>
data_util.py - data preprocessing library for interview state model<br/>
Job_title_embedding_prediction_interview-mse-2output-gpu.ipynb with two output layers to predict both job title embedding and interview state <br/>
Job_title_embedding_prediction_interview-mse-gpu.ipynb with one output layer to predict job title emebedding only <br/>

3.Representing a job title embedding by an average word embeddings in corresponding CV data<br/>
TfidfDocVectorizer.py<br/>

Due to data protection law, data used for this project can not be published here.<br/>
In order to run the code, you should be able to run ipython notebook after feeding your own data into my system.<br/>
