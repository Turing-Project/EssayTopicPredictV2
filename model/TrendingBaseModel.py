# coding=utf-8
import copy
import pickle, json, requests, psycopg2, datetime, pytz
import pandas as pd
from tqdm import tqdm
from sshtunnel import SSHTunnelForwarder

from nltk.tokenize import RegexpTokenizer


class TrendingBaseModelBuilder(object):
    """
    @:param rss3.io
    @:keyword Revery trending
    @:date: 2022/04/27
    """

    def __init__(self):
        self.sign_filter = RegexpTokenizer(r'\w+')
        self.items_dict = {}
        self.config = {}
        self.usr_address_dict = {}
        self.dataset = []
        self.json_dict = {}
        self.save_json_name = f"{datetime.datetime.now().strftime('%Y%m%d%H')}_recommendation_feed_pregod.json"
        self.noncontent_candidate_list = []
        self.content_feeds = []
        self.CURT_TIME = self._get_current_time()
        self.PAST_TIME = None

    def __repr__(self):
        py_version = "3.8.10"
        f"{py_version=}"

    @staticmethod
    def _remove_after(string, suffix):
        return string[:string.index(suffix) + len(suffix)]

    def _fetch_nft_scores(self, n=50000):
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching {n} users' nft scores...")
        page = json.loads(requests.get(f'https://raas.cheer.bio/range?from=1&to={n}', timeout=(10, 30)).text)
        for user_dict in page:
            address = user_dict['address']
            if address not in self.usr_address_dict:
               self.usr_address_dict[address] = user_dict['score']
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done.")

    def _get_past_time(self):
        past_time = (datetime.datetime.fromtimestamp(self.CURT_TIME)
                     - datetime.timedelta(days=self.config["day_period"]))
        print(f"past time: {past_time}")
        return past_time.timestamp()

    @staticmethod
    def _get_current_time():
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] UTC timezone is in used.")
        print(f"current time: {datetime.datetime.now(tz=pytz.utc)}")
        current_time = datetime.datetime.now(tz=pytz.utc)
        return current_time.timestamp()


    @staticmethod
    def _insert_every_n_element(x, y, n):
        """
        insert y's each elements into x in every n element.
        """
        if len(y) == 0: return x
        i = n
        iy = 0
        result = copy.deepcopy(x)
        while i < len(result) and iy < len(y):
            result.insert(i, y[iy])
            i += (n + 1)
            iy += 1
        # in case there are elements left in y.
        if iy < len(y):
            result.extend([y[iiy] for iiy in range(iy, len(y))])
        return result

    def _check_nft_score(self, address):
        if address not in self.usr_address_dict:
            try:
                page = json.loads(
                    requests.get(f'https://raas.cheer.bio/user/{address}', timeout=3).text)
                user_nft_score = page['user']['score']
            except:
                user_nft_score = 0
            self.usr_address_dict[address] = user_nft_score

    def loadData_ProGod(self, save_path=None, type_list=None):
        print("Load data from PreGod.")
        self.PAST_TIME = self._get_past_time()
        with SSHTunnelForwarder(ssh_address_or_host=('34.219.145.46', 22),
                                ssh_username="ec2-user",
                                remote_bind_address=('pregod.cbecaot23mdi.us-west-2.rds.amazonaws.com', 5432)) as server:
            conn = psycopg2.connect(database="pregod",
                                    user="pregod",
                                    password="bgsPR5i38ZTdb6hK",
                                    host="127.0.0.1",
                                    port=server.local_bind_port)
            if type_list is None:
                type_list = self.config["all_types"]
            with conn.cursor() as cursor:
                items = {}
                sql = 'SELECT * from note WHERE source IN %s ORDER BY date_created DESC'
                bar = tqdm(type_list,
                           desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetch data",
                           total=len(type_list),
                           ncols=150)
                for item_type in bar:
                    items[item_type] = []
                    type_tuple = (item_type, '')
                    cursor.execute(sql, (type_tuple,))
                    while True:
                        # cur_items: list of items that satisfies {sql} and {item_type}
                        cur_items = cursor.fetchmany(size=self.config["load_data_batch"])  # TODO O(number of data on database)
                        if len(cur_items) < 1: break
                        if self.config["day_period"] > 0:
                            date_batch = []
                            for item_idx, item in enumerate(cur_items):
                                bar.set_postfix(fetched=f"{item_idx + 1}/{len(cur_items)} {item_type}")
                                cur_date = item[13].timestamp()
                                if cur_date > self.PAST_TIME:
                                    date_batch.append(item)
                                else:
                                    # As the feeds are from new to old in a batch, so break if donot match the time thres.
                                    break
                            items[item_type].extend(date_batch)
                        else:
                            items[item_type].extend(cur_items)
                bar.close()
                item_keys = cursor.description
            conn.close()
            for k, v in items.items():
                self.items_dict[k] = pd.DataFrame(v, columns=[x[0] for x in item_keys])
            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump(self.items_dict, f, protocol=0)

    def _save_to_json(self, topics, OUTPUT_PATH):
        finale_feed_results = self._insert_every_n_element(self.content_feeds,
                                                           self.noncontent_candidate_list,
                                                           self.config["element_skip"])[:self.config["max_feeds"]]
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Gain {len(self.content_feeds)} / {len(self.noncontent_candidate_list)} of content-based / non-content-based feeds, "
              f"where {len(finale_feed_results)} feeds recommended in total.")
        self.json_dict = {"start": int(self.PAST_TIME), "end": int(self.CURT_TIME),
                          "feeds": finale_feed_results,
                          "topics": topics
                          }
        self._uidToSummary(f"{OUTPUT_PATH}/{self.save_json_name}")

    def _uidToSummary(self, save_path): #TODO
        reco_feeds = self.json_dict['feeds']
        all_item_pd = None
        for type_name, item_pd in self.items_dict.items():
            if all_item_pd is None:
                all_item_pd = item_pd
            else:
                all_item_pd = pd.concat([all_item_pd, item_pd], axis=0)
        reco_summary = []
        for fid in reco_feeds:
            the_item = all_item_pd[all_item_pd['identifier'] == fid]
            if not len(the_item) == 0:
                if the_item['summary'][the_item['summary'].index[0]] is not None:
                    reco_summary.append(str(the_item['date_created'].tolist()[0]).split("+")[0] + " " + the_item['summary'][the_item['summary'].index[0]])
                else:
                    reco_summary.append(str(the_item['date_created'].tolist()[0]).split("+")[0] + " None")
        self.json_dict['summary'] = reco_summary
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(self.json_dict, f, indent=1)

    def _post_to_BFF(self, post_path):
        print('post to: ' + post_path + ' ...')
        datas = {"start": self.json_dict['start'], "end": self.json_dict['end'], "feeds": self.json_dict['feeds']}
        # only for post test
        # with open('output/post_to_BFF.json', "w", encoding='utf-8') as f:
            # json.dump(datas, f, indent=1)

        r = requests.post(post_path, json=datas)
        print(r.text)

    def _json_summary_check(self, dict_path):
        print('json summary check with ' + dict_path + '...')
        f = open(dict_path)
        ban_word_list = []
        for line in f:
            ban_word_list.append(line[:-1])

        ban_list = []
        for i in range(len(self.json_dict['summary'])):
            for block_word in ban_word_list:
                if block_word in self.json_dict['summary'][i]:
                    ban_list.append(i)
                    break
        ban_list = list(set(ban_list))
        ban_list.reverse()
        # print(checked_list)
        for feed_index in ban_list:
            self.json_dict['feeds'].pop(feed_index)
            self.json_dict['summary'].pop(feed_index)
        print(str(len(ban_list)) + ' feeds banned.')