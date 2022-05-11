# coding=utf-8
import argparse
import os, langid, pickle, datetime, random, time, re
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm

from bertopic import BERTopic
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

from model.TrendingBaseModel import TrendingBaseModelBuilder
from flair.embeddings import TransformerDocumentEmbeddings

INPUT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/dataset/'
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/model/'
OUTPUT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/output/'
BAN_WORD_PATH = os.path.dirname(os.path.realpath(__file__)) + '/model/english_dictionary.txt'
# https://github.com/downloads/wear/harmonious_dictionary/dictionaries.zip

for path in [INPUT_PATH, MODEL_PATH, OUTPUT_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)


class EssayTopicPredictModel(TrendingBaseModelBuilder):
    """
    @:param rss3.io
    @:keyword Revery trending
    @:date: 2022/04/26
    """

    def __init__(self,):
        super().__init__()
        self.finalkey = None
        self.VISUAL_MODEL = False
        self.SAVE_MODEL = False
        self.SAVE_JSON = True
        self.THRESHOLD = 64
        self.config = {
            "load_data_batch": 1e6,
            "content_types": ['Twitter', 'Misskey'],
            "day_period": 3,
            "spec_delete_list": [' ', 'http', 'www', 'rt', 'media', 'class', 'jpg', 'com', 'twimg', 'image','png'],
        }

        self.lan_candidates = ['en']  # , 'zh']
        langid.set_languages(self.lan_candidates)
        self.en_filter = ['NN', 'NNS', 'NNP', 'NNPS']  # only reserve noun
        self.n_gram_range = (2, 2)
        self.min_topic_size = 5
        self.diversity = 0.2
        self.num_scores = 50000

    def word2VecGaussianModel(self):
        pass  # todo

    def BERTopicModel(self):
        """
        BERT tokenize and Clustering with DBSCAN
        :return: model saver
        """
        model_version = OUTPUT_PATH + "TopicModel" + "_range" + str(self.n_gram_range[0]) + "_min_size" \
                        + str(self.min_topic_size) + "_diversity" + str(self.diversity)

        roberta = TransformerDocumentEmbeddings('chinese-roberta-wwm-ext')
        if roberta:
            model = BERTopic(embedding_model=roberta)
        else:
            model = BERTopic(embedding_model="all-MiniLM-L6-v2", language="english", calculate_probabilities=True,
                             n_gram_range=self.n_gram_range, nr_topics='auto', min_topic_size=self.min_topic_size,
                             diversity=self.diversity, verbose=True)  # embedding can be any language

        if len(self.dataset) < 100:
            raise Exception(f"Too less feeds are fetched ({len(self.dataset)}<100), please set a longer day period.")

        topics, probabilities = model.fit_transform(self.dataset)
        f"{topics=}" \
        f"{probabilities=}"

        topic_count = deepcopy(model.topic_sizes)
        topic_names = deepcopy(model.topic_names)
        del topic_count[-1]
        # print(f"{first_Topic=}")

        if self.VISUAL_MODEL:
            fig_name = datetime.datetime.now().strftime('%Y%m%d')
            # there is a bug in the following fuction located in "python3.8/site-packages/bertopic/plotting/_topics.py" line 49.
            # need to change to  "topics = sorted(topic_model.get_topic_freq().Topic.to_list()[0:top_n_topics])"
            fig1 = model.visualize_topics(top_n_topics=None, width=700, height=700)
            fig1.write_html(OUTPUT_PATH + f"{fig_name}_topic.html")
            fig2 = model.visualize_barchart(top_n_topics=None, width=400, height=400)
            fig2.write_html(OUTPUT_PATH + f"{fig_name}_word_score.html")
            fig3 = model.visualize_term_rank()  # .visualize_distribution(probabilities[200], min_probability=0.015)
            fig3.write_html(OUTPUT_PATH + f"{fig_name}_3.html")

        if self.SAVE_MODEL:
            model.save(model_version)

        return topic_names

    def wordTokenPreprocessor(self):
        """
        # 1.stripped emoji, URLs/HTML tags, and common English ”stopwords”
        # 2.lowercase,tokenized duplication-reduce and stemming/Lemmatization
        # 3.filter infrequent words less than 5 time in the entire corpus and short documents
        """
        self._fetch_nft_scores(self.num_scores)
        combined_data = pd.DataFrame()
        for content_type in self.config["content_types"]:
            combined_data = combined_data.append(self.items_dict.get(content_type), ignore_index=True)

        self.finalkey = []
        bar = tqdm(combined_data.index,
                   desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Word tokenisation",
                   total=len(combined_data),
                   ncols=150)
        for idx in bar:
            list_value = combined_data.loc[idx].values.tolist()
            contenttype = list_value[9]
            if len(list_value[7]) < self.THRESHOLD: continue
            if contenttype == 'Mirror Entry':
                pass
                # For the summary of Mirror's long article, skip it currently
                # custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
                # custom_config.output_hidden_states = True
                # custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)
                # model = Summarizer(custom_model=custom_model)
                # sentence = model(list_value[7], num_sentences=3)  # Summarize the key sentences of an overly-long Mirror article
            # else:

            sentence = list_value[7]
            finalKey = str(list_value[0])+ "#" + str(list_value[13]) # feed_id + created time

            if sentence is None or finalKey is None:
                continue

            # 0. Check language (only consider English in the first version)
            lan_identify, _ = langid.classify(sentence)  # identify language the sentence is.
            if lan_identify != self.lan_candidates[0]:  # en
                continue

            # 1. Remove other special characters such as emojis, picture links, website external links and account addresses
            if 'http' in sentence:
                self._remove_after(sentence, 'http')
            regwords = re.sub(r'<.*?>|\\[.*?\\]|\b0\S*?\w\b|http|com', "", sentence)
            langflag = re.findall(
                u"[\u4e00-\u9fa5]+|[\uac00-\ud7ff]+|[\u30a0-\u30ff\u3040-\u309f]+|[\u0400-\u04FF]+", regwords)
            if len(langflag) > 0:
                continue

            regwords = re.findall(r'[a-zA-Z,\\.\\?]+', regwords)
            # 2. English lowercase, filter stop words + too short data (less than 2 TERMs)
            stop_words = set(stopwords.words('english')).union(set(self.config["spec_delete_list"]))
            regwords = [token.lower() for token in regwords if token.lower() not in stop_words]
            if (len(regwords)) <= 2:
                continue

            # 3. Normalization + stem extraction
            # Stemming / Bert will do automatical tokenization
            ps = PorterStemmer()
            regwords = [ps.stem(token) for token in regwords if len(token) <= 15]
            sentence = " ".join(regwords)
            self.finalkey.append(finalKey)
            self.dataset.append(sentence)

    def jsonSummaryCheck(self, dict_path):
        if self.json_dict:
            try:
                self._json_summary_check(dict_path)
            except Exception as e:
                raise f'summary check failed {e}.'
        else:
            print('blank json dict')

    def datePreprocess(self, finename):
        dataset = []
        """处理数据"""
        if finename.startswith("weibo"):
            pass
            self.items_dict["weibo"] = dataset

        elif finename.startswith("202"):
            pass
            self.items_dict["wenzhang"] = dataset

        elif finename.startswith("yangshi"):
            pass
            self.items_dict["yangshi"] = dataset

        elif finename.startswith("yangshi"):
            pass
            self.items_dict["yangshi"] = dataset

        with open(finename, 'rb') as f:
            items_dict = pickle.load(f)


        ##数据合并
        self.items_dict
        self.PAST_TIME = self._get_past_time()


def test():
    ORI_DATA_SAVE_PATH = 'rec_ori_data_20220316.pkl'  # user-dim data
    newDataHandler = EssayTopicPredictModel()

    for root, dirs, files in os.walk(INPUT_PATH):
        bar = tqdm(files,
                   total=len(files),
                   desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Process NFT, POAP, Donation",
                   ncols=150)
        for each in bar:
            filename = os.path.join(root, each)
            dataset = newDataHandler.datePreprocess(filename)


    newDataHandler.wordTokenPreprocessor()
    newDataHandler.BERTopicModel()
    newDataHandler.jsonSummaryCheck(BAN_WORD_PATH)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] finished")


def online():
    newDataHandler = EssayTopicPredictModel()
    newDataHandler.wordTokenPreprocessor()
    newDataHandler.BERTopicModel()
    newDataHandler.jsonSummaryCheck(BAN_WORD_PATH)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] finished")


def weekly_virsualisation():
    newDataHandler = EssayTopicPredictModel()
    newDataHandler.VISUAL_MODEL = True
    newDataHandler.SAVE_JSON = False
    newDataHandler.config['day_period'] = 7
    newDataHandler.loadData_ProGod(type_list=newDataHandler.config["content_types"])
    newDataHandler.wordTokenPreprocessor()
    newDataHandler.BERTopicModel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', default=False, type=str, required=False, help='open trigger')
    parser.add_argument('--visualize', default=False, type=str, required=False, help='visual trigger')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.online:
        online()
    else:
        test()

    if args.visualize:
        weekly_virsualisation()
