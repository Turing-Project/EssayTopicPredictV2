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
# https://github.com/downloads/wear/harmonious_dictionary/dictionaries.zip
BAN_WORD_PATH = os.path.dirname(os.path.realpath(__file__)) + '/model/english_dictionary.txt'

for path in [INPUT_PATH, MODEL_PATH, OUTPUT_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)


class TrendingRankTopicModel(TrendingBaseModelBuilder):
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
            'all_types': ['Crossbell', 'Ethereum NFT', 'Solana NFT', 'Flow NFT', 'Gitcoin Contribution',
                           'Mirror Entry', 'Twitter Tweet', 'Misskey Note', 'Jike Post'],
            "content_types": ['Twitter', 'Misskey'],
            "noncontent_types": ['Crossbell', 'Ethereum NFT', 'Solana NFT', 'Flow NFT', 'Gitcoin Contribution'],
            "day_period": 3,

            "spec_delete_list": [' ', 'http', 'www', 'rt', 'media', 'class', 'jpg', 'com', 'twimg', 'images',
                                 'image', 'retweet', 't', 'co', 'mv48', 'video', 'testtesttest', 'dev', 'png',
                                 'gm', 'gn', 'file', 'bio', 'images.mirror', 't.co'],
            "nft_rank_num_to_pick": 50,
            "element_skip": 3,
            "max_content_feeds": 100,
            "max_feeds": 200
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

    def NFTTrendingModel(self):
        noncontent_candidate_dict = {}
        for i in range(self.config["nft_rank_num_to_pick"]):
            noncontent_candidate_dict[i] = []
        address_list = []
        bar = tqdm(self.config['noncontent_types'],
                   total=len(self.config['noncontent_types']),
                   desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Process NFT, POAP, Donation",
                   ncols=150)
        for item_type in bar:
            items_pd = self.items_dict[item_type]
            item_ids_idx = items_pd.index
            contract_count_uids_dict = {}
            for idx, item_id in enumerate(item_ids_idx):
                feed_id = items_pd.loc[item_id]['identifier']
                user_id = items_pd.loc[item_id]['authors'][0].split('account:')[1].split('@')[0]
                if "POAP" in items_pd.loc[item_id]['tags']:
                    item_contract = items_pd.loc[item_id]['title']
                else:
                    try:
                        item_contract = items_pd.loc[item_id]['metadata']['collection_address']
                    except KeyError:
                        item_contract = feed_id.split('note:')[1].split('-')[0]
                if item_contract not in contract_count_uids_dict:
                    contract_count_uids_dict[item_contract] = [1, {}]
                else:
                    contract_count_uids_dict[item_contract][0] += 1
                contract_count_uids_dict[item_contract][1][f'{feed_id}#{user_id}'] = 0
            contract_count_uids = sorted(contract_count_uids_dict.items(), key=lambda x: x[1][0], reverse=True)[
                                  :self.config["nft_rank_num_to_pick"]]

            # pull nft score for users, have to do it in a new loop for speeding up.
            for idx, contract_data in enumerate(contract_count_uids):
                uid_dict = contract_data[1][1]
                for uid_idx, uid in enumerate(uid_dict.keys()):
                    bar.set_postfix(type=item_type,
                                    info=f'Rank {idx + 1} NFT score {uid_idx + 1}/{len(uid_dict.keys())}')
                    usr_address = uid.split("#")[1]
                    self._check_nft_score(usr_address)
                    contract_count_uids[idx][1][1][uid] = self.usr_address_dict[usr_address]
                contract_count_uids[idx][1][1] = sorted(contract_count_uids[idx][1][1].items(), key=lambda x: x[1],
                                                        reverse=True)
            # extract each type's top candidate into noncontent_candidate_dict
            for ccidx, cc in enumerate(contract_count_uids):
                user_idx = 0
                cur_candidate = cc[1][1][user_idx][0].split("#")[0]
                while cur_candidate.split('-')[0] in address_list:
                    user_idx += 1
                    if user_idx >= len(cc[1][1]):
                        cur_candidate = cc[1][1][0][0]
                        break
                    cur_candidate = cc[1][1][user_idx][0]
                noncontent_candidate_dict[ccidx].append(cur_candidate)
        self.noncontent_candidate_list.clear()
        for x in noncontent_candidate_dict.values():
            random.shuffle(x)  # shuffle the uid among each rank, while keep the rank.
            self.noncontent_candidate_list.extend(x)
        bar.close()

    def BERTopicModel(self):
        """
        BERT tokenize and Clustering with DBSCAN
        :return: model saver
        """
        model_version = OUTPUT_PATH + "TopicModel" + "_range" + str(self.n_gram_range[0]) + "_min_size" \
                        + str(self.min_topic_size) + "_diversity" + str(self.diversity)


        roberta = TransformerDocumentEmbeddings('roberta-base')
        topic_model = BERTopic(embedding_model=roberta)
        if os.path.isfile(model_version):
            model = BERTopic.load(model_version)
        else:
            model = BERTopic(embedding_model="all-MiniLM-L6-v2", language="english", calculate_probabilities=True,
                             n_gram_range=self.n_gram_range, nr_topics='auto', min_topic_size=self.min_topic_size,
                             diversity=self.diversity, verbose=True)  # embedding can be any language
        if len(self.dataset)< 100:
            raise Exception(f"Too less feeds are fetched ({len(self.dataset)}<100), please set a longer day period.")
        topics, probabilities = model.fit_transform(self.dataset)
        f"{topics=}" \
        f"{probabilities=}"

        topic_embeddings = model.topic_embeddings
        topic_mapping = {}

        # For the doc mapped to each topic, calculate the similarity / distance between embedding
        bar = tqdm(range(len(self.dataset)),
                   total=len(self.dataset),
                   desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Similarity",
                   ncols=150)
        for idx in bar:
            topic = topics[idx]
            if topic == -1: continue  # Skip data that does not belong to any class
            doc_embedding = model._extract_embeddings(self.dataset[idx], method="document", verbose=False)
            topic_embedding = topic_embeddings[topic]
            doc_topic_distance = cosine_similarity(doc_embedding.reshape([1, -1]), topic_embedding.reshape([1, -1]))
            if topic_mapping.get(topic) is None:
                topic_mapping[topic] = [(self.finalkey[idx], self.dataset[idx], doc_topic_distance.tolist()[0][0])]
            else:
                topic_mapping[topic].append((self.finalkey[idx], self.dataset[idx], doc_topic_distance.tolist()[0][0]))
        bar.close()
        topic_count = deepcopy(model.topic_sizes)
        topic_names = deepcopy(model.topic_names)
        # first_Topic = model.get_topic(0)
        # print(f"{first_Topic=}")
        del topic_count[-1]
        candidate_pool = {}
        N_MAX = list(topic_count.values())[0]
        N_MIN = list(topic_count.values())[-1]

        # class size score + inner-class ranking score - time penalty = final score
        # Generate total candidate set based on final score rank
        for topic, t_size in topic_count.items():
            rank_dict = {}
            filter_freq = set()
            topic_freq = 0
            topic_size_scale = topic_count[topic] / self.config["max_content_feeds"]
            if topic_size_scale > 1:
                topic_size = self.config["max_content_feeds"]
            else:
                topic_size = topic_count[topic]
            topic_score = np.log(np.sqrt(t_size) - np.sqrt(N_MIN) / (np.sqrt(N_MAX) - np.sqrt(N_MIN) + 1)) / (
                np.sqrt(N_MAX - N_MIN))
            for item in topic_mapping[topic]:
                # control co-frequency of similar content and feed in topic
                if item[1].split(" ")[0] in filter_freq or topic_freq > np.log(N_MAX):
                    continue
                filter_freq.add(item[1].split(" ")[0])
                rank_dict[item[0] + "#" + str(topic) + "#" + item[1]] = float(item[2]) + float(topic_score)
                topic_freq += 1
            candidate_pool.update(sorted(rank_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:topic_size])

        for feed_key, scores in candidate_pool.items():
            user_address = feed_key.split("-")[0]
            self._check_nft_score(user_address)
            NFT_max = int(np.max(list(self.usr_address_dict.values())))
            NFT_mean = int(np.mean(list(self.usr_address_dict.values())))
            try:
                nft_score = 1 / (1 + np.exp(-float((self.usr_address_dict[user_address] - NFT_mean)) / NFT_max))
            except:
                nft_score = 0

            # time penalty
            timeArray = time.strptime(feed_key.split("#")[1].split("+")[0], "%Y-%m-%d %H:%M:%S")
            item_time = int(time.mktime(timeArray))
            item_time = datetime.datetime.fromtimestamp(item_time)
            init_time = datetime.datetime.fromtimestamp(self.CURT_TIME)
            hour_diff = (init_time - item_time).seconds / 3600
            if self.config["day_period"]==0:
                time_penalty = np.exp(hour_diff)
            else:
                time_penalty = np.exp((np.log(100) / (24 * self.config["day_period"])) * hour_diff)
            candidate_pool[feed_key] = (scores + nft_score) / time_penalty

        final_result = sorted(candidate_pool.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        self.content_feeds = [feed[0].split("#")[0] for feed in final_result]
        topics = ["size:" + str(topic_count[int(feed[0].split("#")[2])]) +
                  "_topic:" + topic_names[int(feed[0].split("#")[2])].split("_")[1] for feed in
                  final_result]
        if self.SAVE_JSON:
            self._save_to_json(topics, OUTPUT_PATH)

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

    def postToBFF(self, post_path):
        if self.json_dict:
            try:
                self._post_to_BFF(post_path)
            except Exception as e:
                raise f'post failed {e}.'
        else:
            print('blank json dict')

    def jsonSummaryCheck(self, dict_path):
        if self.json_dict:
            try:
                self._json_summary_check(dict_path)
            except Exception as e:
                raise f'summary check failed {e}.'
        else:
            print('blank json dict')

def post_test():
    # only for post test
    import requests, json
    with open('output/2022042621_recommendation_feed.json', 'rb') as f:
        datas = json.load(f)
        r = requests.post(args.postpath, json=datas)
        print(r.text)


def test():
    ORI_DATA_SAVE_PATH = 'rec_ori_data_20220316.pkl'  # user-dim data
    newDataHandler = TrendingRankTopicModel()

    if not os.path.isfile(INPUT_PATH + ORI_DATA_SAVE_PATH):
        newDataHandler.loadData_ProGod(save_path = INPUT_PATH + ORI_DATA_SAVE_PATH)
    else:
        with open(INPUT_PATH + ORI_DATA_SAVE_PATH, 'rb') as f:
            items_dict = pickle.load(f)
            newDataHandler.items_dict = items_dict
            newDataHandler.PAST_TIME = newDataHandler._get_past_time()
    newDataHandler.wordTokenPreprocessor()
    newDataHandler.NFTTrendingModel()
    newDataHandler.BERTopicModel()
    newDataHandler.jsonSummaryCheck(BAN_WORD_PATH)
    newDataHandler.postToBFF(args.postpath)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] finished")


def online():
    newDataHandler = TrendingRankTopicModel()
    newDataHandler.loadData_ProGod()

    newDataHandler.wordTokenPreprocessor()
    newDataHandler.NFTTrendingModel()
    newDataHandler.BERTopicModel()
    newDataHandler.jsonSummaryCheck(BAN_WORD_PATH)
    newDataHandler.postToBFF(args.postpath)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] finished")


def weekly_virsualisation():
    newDataHandler = TrendingRankTopicModel()
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

    parser.add_argument('--postpath', default='https://api-dev.revery.so/v0/admin/recommendation/trending', type=str, required=False,help='post path')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.online:
        online()
    else:
        test()

    if args.visualize:
        weekly_virsualisation()
