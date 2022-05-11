from neo4j import GraphDatabase, basic_auth
from tqdm import tqdm

driver = GraphDatabase.driver('bolt://localhost', auth=basic_auth("neo4j", "1231"))
session = driver.session()
with open('people.csv', encoding='utf-8') as f:
    text = tqdm(f.readlines()[1:])
    for rel in text:
        newsid = rel.split(',')[0]
        orgname = rel.split(',')[1].strip('\n')
        query = "MATCH (news:News {news_id:" +'"'+ str(newsid) +'"' + "}), (p: Person {person_name:" + '"' + orgname + '"' + "}) MERGE (news) - [r:参与人员] ->(p)"
        # print(query)
        session.run(query)
        session.close()
