from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import nltk

def align_data(xs, ys):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    data_aligned = []

    # for each entry, create aligned string
    for x, y in zip(xs, ys):
        data_aligned.append(x + " " + y)

    return data_aligned

def predict(text):
    
    text = " ".join(nltk.word_tokenize(text))
    # create instance of config
    config = Config()
    
    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)
  
    preds = model.predict(text)
        
    words = text.strip().split(" ")
    words_tags = align_data(words, preds)
    return words_tags