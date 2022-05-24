# coding=utf-8
import argparse
import logging
import os, langid, datetime, re
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm

from bertopic import BERTopic


from model.TrendingBaseModel import TrendingBaseModelBuilder
from flair.embeddings import TransformerDocumentEmbeddings

INPUT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/dataset/'
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/model/'
OUTPUT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/output/'
TMP_PATH = os.path.dirname(os.path.realpath(__file__)) + '/tmp/'
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
        self.THRESHOLD = 8
        self.config = {
            "load_data_batch": 1e6,
            "content_types": ['weibo', 'wenzhang', 'yangshi', 'daily'],
            "day_period": 3,
            "spec_delete_list": [' ', 'http', 'www', 'rt', 'media', 'class', 'jpg', 'com', 'twimg', 'image','png'],
        }
        self.items_dict["weibo"] = []
        self.items_dict["wenzhang"] = []
        self.items_dict["yangshi"] = []
        self.items_dict["daily"] = []

        self.lan_candidates = ['zh']  # , 'zh']
        langid.set_languages(self.lan_candidates)
        self.en_filter = ['NN', 'NNS', 'NNP', 'NNPS']  # only reserve noun
        self.n_gram_range = (2, 2)
        self.min_topic_size = 10
        self.diversity = 0.1
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

        roberta = TransformerDocumentEmbeddings('hfl/chinese-roberta-wwm-ext')
        if roberta:
            model = BERTopic(embedding_model=roberta, verbose=True, low_memory=True, n_gram_range=self.n_gram_range,
                             min_topic_size=self.min_topic_size, diversity=self.diversity)
        else:
            model = BERTopic(embedding_model="all-MiniLM-L6-v2", language="english", calculate_probabilities=True,
                             n_gram_range=self.n_gram_range, nr_topics='auto', min_topic_size=self.min_topic_size,
                             diversity=self.diversity, verbose=True)  # embedding can be any language

        if len(self.dataset) < 100:
            raise Exception(f"Too less feeds are fetched ({len(self.dataset)}<100), please set a longer day period.")

        f"model has been load through hugging face, then start training in{model_version}..."
        topics, probabilities = model.fit_transform(self.dataset)
        f"{topics=}" \
        f"{probabilities=}"



        topic_count = deepcopy(list(model.topic_sizes.values())[:])
        topic_names = deepcopy(list(model.topic_names.values())[:])
        result = pd.DataFrame(zip(topic_names, topic_count))
        result.to_csv("topic_result.csv", encoding='utf_8_sig', mode='w', index=False, sep=',', header=False)

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

    @staticmethod
    def saveFile(path, filename, data):
        if not os.path.exists(path):
            os.makedirs(path)
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(path + filename + ".csv", encoding='utf_8_sig', mode='w', index=False, sep=',', header=False)

    """创建停用词列表"""
    def stopwordslist(self):
        stopwords = [line.strip() for line in open('./stopwords.txt', encoding='UTF-8').readlines()]
        return stopwords

    def wordTokenPreprocessor(self):
        """
        # 1.stripped emoji, URLs/HTML tags, and common English ”stopwords”
        # 2.lowercase,tokenized duplication-reduce and stemming/Lemmatization
        # 3.filter infrequent words less than 5 time in the entire corpus and short documents
        """
        global local_cache
        combined_data = pd.DataFrame()
        step = 0
        local_cache = False
        try:
            for root, dirs, files in os.walk(TMP_PATH):
                for file in files:
                    filename = os.path.join(root, file)
                    if os.path.isfile(filename):
                        combined_data = pd.read_csv(filename, encoding='utf_8_sig', sep=',')
                        combined_data = combined_data.sample(n=20000, replace=False, weights=None, axis=0)
                        local_cache = True
                        break

            # self._fetch_nft_scores(self.num_scores)
            if local_cache is not True:
                for content_type in self.config["content_types"]:
                    combined_data = combined_data.append(self.items_dict.get(content_type), ignore_index=True)

            combined_data.drop_duplicates(keep='last')
            combined_data.dropna()

            self.finalkey = []
            bar = tqdm(combined_data.index,
                       desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Word tokenisation",
                       total=len(combined_data),
                       ncols=150)
            for idx in bar:
                list_value = combined_data.loc[idx].values.tolist()
                if len(str(list_value[0])) < self.THRESHOLD: continue

                sentence = list_value[0]
                if sentence is None:
                    continue
                # 0. Check language (only consider English in the first version)
                lan_identify, _ = langid.classify(sentence)  # identify language the sentence is.
                if lan_identify != self.lan_candidates[0]:  # en
                    continue

                # 1. Remove other special characters such as emojis, picture links, website external links and account addresses
                if 'http' in sentence:
                    self._remove_after(sentence, 'http')
                regwords = re.sub(r'<.*?>|\\[.*?\\]|\b0\S*?\w\b|http|com', "", sentence)
                stopwords = '|'.join(self.stopwordslist())
                sentence = re.sub(stopwords, "", regwords)

                if (len(sentence)) <= 2:
                    continue
                self.dataset.append(sentence)
        except Exception as e:
            logging.Logger.info("catch error: ", e)
        finally:
            if local_cache is not True:
                self.saveFile(TMP_PATH, "processed_data", self.dataset)
            print("final dataset has been saved, with %d" % len(self.dataset))
        self.dataset = [str(x) for x in self.dataset]

    def jsonSummaryCheck(self, dict_path):
        if self.json_dict:
            try:
                self._json_summary_check(dict_path)
            except Exception as e:
                raise f'summary check failed {e}.'
        else:
            print('blank json dict')

    def datePreprocess(self, data_path):
        for root, dirs, files in os.walk(data_path):
            bar = tqdm(files,
                       total=len(files),
                       desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Process daily, yangshi, weibo",
                       ncols=150)
            for file in bar:
                try:
                    filename = os.path.join(root, file)
                    """处理数据"""
                    if file.startswith("weibo"):
                        dataset = pd.read_csv(filename, encoding="utf-8", header=None)
                        dataset = list(set([x[0] for x in np.array(dataset).tolist()]))
                        self.items_dict["weibo"].extend(dataset)

                    elif file.startswith("202"):
                        f = open(filename, "r", encoding="utf-8")
                        dataset = f.readlines()
                        if "责编" not in dataset:
                            self.items_dict["wenzhang"].extend(dataset)

                    elif file.startswith("yangshi"):
                        dataset = pd.read_csv(filename, encoding="utf-8")
                        dataset["concat"] = dataset["title"] + "。" + dataset["brief"]
                        dataset = list(set(np.array(dataset["concat"]).tolist()))
                        self.items_dict["yangshi"].extend(dataset)

                    elif file.startswith("daily"):
                        dataset = pd.read_csv(filename, encoding="utf-8")
                        dataset["concat"] = dataset.iloc[:, 1] + "。" + dataset.iloc[:, 6]
                        dataset = list(set(np.array(dataset["concat"]).tolist()))
                        self.items_dict["daily"] = dataset
                except Exception as e:
                    print("logging error as %s" % e)

        self.PAST_TIME = self._get_past_time()


def test():
    # from sklearn.datasets import fetch_20newsgroups
    # docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    newDataHandler = EssayTopicPredictModel()
    newDataHandler.datePreprocess(INPUT_PATH)
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
