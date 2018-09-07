import subprocess
import nltk
import math
import pickle
import json
import numpy as np
from multiprocessing import Pool
import os
import collections
import string
import pandas as pd
import os.path
import time
import random
from InferSent.models import InferSent
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from tqdm import tqdm_notebook
import nltk
from ast import literal_eval
from nltk.corpus import stopwords
from torchtext import data
from nltk.stem import WordNetLemmatizer
import re
from torchtext import data
# import enchant

''' 
This function is used to search files with a pattern under a specific path to a maximum
depth of maxdepth.
'''
def find(path, pattern, maxdepth):
    return [line.decode("utf-8") for line in subprocess.check_output("find " + path \
        + " . -maxdepth " + maxdepth + " -type f -iname " + pattern, shell=True).splitlines()]   

'''
This function is used to load a pickle object from a given path.
'''
def load_obj(file_path):
    file = open(file_path, "rb")   
    obj = pickle.load(file)
    file.close()
    return obj

'''
This function is used to save a pickle object to a given path.
'''
def save_obj(obj, file_path):
    file = open(file_path, "wb")   
    pickle.dump(obj, file)
    file.close()

'''
This function is used to load a read Json object from a given path.
'''
def read_json(file_path):
    file = open(file_path, 'r', encoding="utf-8")
    data = json.load(file)
    file.close()
    return data

'''
This function is used to read the content from a file path.
'''
def read_text(path):
    print(path)
    file = open(path, "r", encoding="utf-8")
    text = ''.join(file.readlines())
    file.close()
    return text

def retrieve_cv_text(row, sys_id):
    cv_path = '/home/vX/CV/' + sys_id + "/" + row["cv_name"]
    cv_txt = read_text(cv_path)
    if len(cv_txt) >= 25:
        return cv_txt
    return np.nan

'''
This function is used to convert an English interview state to
an numerical interview state.
'''
def is_interviewed(state, company_id):
    """
    Return true if applicant got to interview stage.
    Return none if there was no manually determined interview outcome.
    """
    interview_outcomes = load_obj("data/interview_outcomes.pickle") 
    if str(state).lower() not in interview_outcomes[company_id]:
        return None
    else:
        return interview_outcomes[company_id][str(state).lower()]

def app_id_dir(app_id):
    """
    Given a numeric app_id, generate its location
    in the nested directories e.g. 295835  => 0/00/000/0002/00029/000295/0002958
    """
    app_id = int(app_id)

    # The nesting is odd. The deepest contains up to 100 files, but less deep
    # only contain up to 10 folders.

    depth = 7
    folders = []
    for i in range(0, depth):
        f = app_id // (1 * 10 ** (depth-i+1))
        f = str(f).zfill(i+1)
        folders.append(f)

    return '/'.join(folders)

def retrieve_data(row, sys_id, oppid_data_dict):
#     row['text'] = retrieve_cv_text(row, sys_id)
    opp_id = str(row['opp_id'])
    row['job_title'] = oppid_data_dict[opp_id]
    
    app_id = str(row['app_id'])
    app_dir = '/home/vX/apps_opps/app/' + sys_id + '/' + app_id_dir(app_id) + '/'
    app_file_name = 'app_' + sys_id + '_' + app_id + '.json' 
    app_file_path = app_dir + app_file_name
    
    if os.path.exists(app_file_path):
        state = read_json(app_file_path)['state']
        company_id = int(sys_id)
        row['interview'] = is_interviewed(state, company_id)
    else:
        row['interview'] = np.nan
    return row

def get_id_data_dict(search_path, sys_id, pattern, depth, field):
    pool = Pool()
    data_dict = collections.defaultdict(str)
    paths = find(search_path + sys_id, pattern, depth)
    for path in paths:
        base = os.path.basename(path)
        file, ext = os.path.splitext(base)
        opp_id = file.split('_')[-1]
        json_data = read_json(path)
        data = json_data[field]
        data_dict[opp_id] = data
    pool.close()
    pool.join()
    return data_dict

def get_csv_by_system(sys_id):
    paths = find('/home/vX/CV/' + sys_id, '*files_*', '2')[0]
    oppid_data_dict = get_id_data_dict('/home/vX/apps_opps/opp/', sys_id,
                                       '*opp_' + sys_id + '*', '8', 'title')
    df = pd.read_csv(paths)[:1000]
    df.dropna(inplace=True)
    opp_type_df = pd.read_csv("/home/vX/opp_types/opp_type_" + sys_id + "_live.csv")
    df = pd.merge(df, opp_type_df, on='opp_id')
    df = df[df['opp_type'] == "Vacancy"]
#     df = df.apply(lambda row : retrieve_data(row, sys_id, \
#                   oppid_data_dict), axis=1)
    for index, row in df.iterrows():
        df.loc[index] = retrieve_data(row, sys_id, oppid_data_dict)
    df = df.drop(['candidate_id', 'cv_name'], axis=1)
    df.dropna(inplace=True)
    
#     df['interview'] = df['interview'] * 1
#     print("average cv text length: " + str(df['text'].map(str).apply(len).mean()))
    df.to_csv("data/" + sys_id + "_CVs_outcome.csv", index=False)
    return df

'''
This function is used to encode a list of sentences using InferSent.
'''
def encode_sentences(sentences):
    MODEL_PATH =  'InferSent/encoder/infersent1.pkl'
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = 'InferSent/dataset/GloVe/glove.840B.300d.txt'
    model.set_w2v_path(W2V_PATH)
    model.build_vocab(sentences, tokenize=True)
    embeddings = model.encode(sentences, bsize=128, tokenize=True, verbose=True)
#     print('nb sentences encoded : {0}'.format(len(embeddings)))
    return embeddings

'''
This function is used to compress job title embedding to a reduced dimension
of 50.
'''
def compress_job_vec(job_vecs):
    pca = PCA(n_components=50)
    embeddings = pca.fit_transform(job_vecs)
#     mds = MDS(n_components=50,n_init=20)
#     embeddings = mds.fit_transform(job_vecs)
    return embeddings

def encoder_jobtitle(row, job_vec_dict):    
    row['job_title_vec'] = job_vec_dict[row['job_title']]
    return row

def create_jobtitle_vec_dict(jobtitles):
    job_titles = list(set(jobtitles))
    vec_dict = collections.defaultdict(list)
    job_vecs = list(encode_sentences(job_titles))
    job_vecs = compress_job_vec(job_vecs)
    for title, vec in tqdm_notebook(zip(job_titles, job_vecs)):
        vec_dict[title] = list(vec)
    return vec_dict

'''
This function is used to normalize the job titles text.
'''
def normalize_jobtitle(row):
    job_title = row['job_title']
    lemmatizer = WordNetLemmatizer()
    locations = ['amsterdam', 'australia', 'beijing', 'brazil', 'canada',
             'calgary', 'montreal', 'toronto', 'chester', 'dubai',
             'dublin', 'frankfurt', 'hong', 'kong', 'japan', 'korea',
             'johannesburg', 'latin', 'america', 'london', 'madrid',
             'milan', 'moscow', 'paris', 'shanghai', 'singapore', 'south',
             'africa', 'stockholm', 'sydney', 'taiwan', 'thailand', 'india',
             'jakarta', 'kuala', 'lumpur', 'melbourne', 'emea', 'apac']
    seasons = ['spring', 'summer', 'autumn', 'winter', 'fall']
    grades = ['associate', 'junior', 'global', 'programme', 'program', 'internship',
              'full', 'time', 'industrial', 'placement', 'intern']
    stop_words = list(set(nltk.corpus.stopwords.words('english'))) + seasons
        
    punctuation = string.punctuation
    stop_words += punctuation
    title_processed = []
    for token in nltk.word_tokenize(job_title):
        token = lemmatizer.lemmatize(token.lower().strip(punctuation))
        if token not in stop_words and token.isalpha():
            title_processed.append(token) 
    row['job_title'] = " ".join(title_processed)
    return row

'''
During loading job title embedding, this function is used to convert raw string
into their numerical values using ast.literal_eval.
'''
def label_field_preprocessing(x):
    x = literal_eval(x)
    return x

def label_field_postprocessing(batch):
    """ Process a list of examples to create a batch.
    Postprocess the batch with user-provided Pipeline.
    Args:
        batch (list(object)): A list of object from a batch of examples.
    Returns:
        object: Processed object given the input and custom
        postprocessing Pipeline.
    """
    batch = torch.from_numpy(np.array(batch, dtype=np.float32))
    return batch

'''
This function is used to load training, validation and test data into batches
for the model training.
'''
def load_data(text_field, label_field, batch_size):
    start_time = time.time()
    data_fields = [("text", text_field), ("interview", label_field), ('job_title_vec', label_field)]
    train, dev, test = data.TabularDataset.splits(path='data/', train='CVs_outcome_train.csv', \
                               validation='CVs_outcome_val.csv', test='CVs_outcome_test.csv', \
                               format='csv', skip_header=True, fields=data_fields)
    end_time = time.time()
    print(end_time - start_time)
    text_field.build_vocab(train, vectors="glove.6B.50d")
    end_time = time.time()
    print(end_time - start_time)
#     label_field.build_vocab(train)
    device = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), 
                batch_sizes=(batch_size, batch_size, batch_size), 
                sort_key=lambda x: len(x.text),
                shuffle=False, sort=False, repeat=False, device=device)
    end_time = time.time()
    print(end_time - start_time)
    ## for GPU run
#     train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
#                 batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=None)
    return train_iter, dev_iter, test_iter


'''
This function is used to preprocess CV text during loading data.
'''
def text_tokenize(text):
    stopword_list = stopwords.words('english') 
    punctuations = list(string.punctuation)
    lemmatizer = WordNetLemmatizer()
#     eng_dict = enchant.Dict("en_US")
#     word_re = re.compile(r'\w+')
    tokens = []
#     words = word_re.findall(text)
    for sentence in nltk.sent_tokenize(text):
#         word = lemmatizer.lemmatize(word.lower())
        for word in nltk.word_tokenize(sentence):
            word = word.lower()
            if word not in stopword_list and word not in punctuations:
                tokens.append(word)
    return tokens

'''
This function is used to convert batches of word IDs vector
back into their text representation.
'''
def batch_text_itos(batch_text, itos):
    res = []
    max_seq_len, batch_size = batch_text.size()
    for j in range(batch_size):
        text = []
        for i in range(max_seq_len):
            word_id = batch_text[i][j]
            word = itos[word_id]
            text.append(word)
        res.append(" ".join(text))
    return res