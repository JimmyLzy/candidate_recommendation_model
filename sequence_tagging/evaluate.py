from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    
    sentence = "Computer scientists, also called computer and information scientists, can work for government agencies and private software publishers, engineering firms or academic institutions. Businesses and government agencies usually employ these scientists to develop new products or solve computing problems. Computer scientists employed by academic institutions are typically involved in more theoretical explorations of computing issues, often using experimentation and modeling in their research. Computer scientists often work as part of a research team with computer programmers, information technology professionals, and mechanical or electrical engineers. Their research often is used to design new computer technology. They typically investigate technological topics like artificial intelligence, robotics or virtual reality. The results of their research can lead to the improved performance of existing computer systems and software as well as the development of new hardware or computing techniques and materials. Most computer scientists hold a bachelor's degree with a major in computer science, information systems or software engineering. After completing this 4-year program, computer scientists often earn a Ph.D. in computer science, computer engineering or a similar area of study. This additional program includes coursework in hardware and software systems, program languages and computational modeling as well as a research project. "

    words_raw = sentence.strip().split(" ")

    preds = model.predict(words_raw)
    to_print = align_data({"input": words_raw, "output": preds})

    for key, seq in to_print.items():
        model.logger.info(seq)


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
#     interactive_shell(model)

if __name__ == "__main__":
    main()
