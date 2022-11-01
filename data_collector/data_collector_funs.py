"""
Class to collect/ upload data to the Database (MySQL) and MongoDB
"""
import warnings

import pandas as pd
from tqdm import tqdm

from data_collector.database_access import *

warnings.filterwarnings("ignore")

"""
Class to collect/ upload data to the Database (MySQL) and MongoDB
"""

import pymongo
import pymysql as MySQLdb
from sshtunnel import SSHTunnelForwarder
from utils import get_list_of_dict


class DataCollector:

    def __init__(self, ):

        try:
            client = pymongo.MongoClient(serverSelectionTimeoutMS=30000)

            # get the mongoDB database instance
            self.mongo_db = client[MONGO_DB_NAME]

            self.mysql_conn = MySQLdb.connect(host='127.0.0.1',
                                              user=MYSQL_USERNAME,
                                              password=MYSQL_PASS,
                                              database=MYSQL_DB_NAME,
                                              charset='utf8')
            self.mysql_cursor = self.mysql_conn.cursor()
        except:

            # Since the database is hosted at the University, not locally, we need SSH Tunnel to connect.
            # If you want to use this code from the server, not locally from your computer, you do NOT need for a tunnel
            # If you run the code from your laptop, the VPN connection is needed.

            # Create a tunnel to connect to the server from outside with VPN
            self.mongodb_ssh_tunnel_server = SSHTunnelForwarder(
                SERVER_IP,
                ssh_username=SERVER_USERNAME,
                ssh_password=SERVER_PASSWORD,
                # 27017 is the default port for MongoDB
                remote_bind_address=('127.0.0.1', 27017)
            )

            # Start the SSH tunnel server
            self.mongodb_ssh_tunnel_server.start()

            # connect to mongoDB data base
            client = pymongo.MongoClient('127.0.0.1', self.mongodb_ssh_tunnel_server.local_bind_port)

            # get the mongoDB database instance
            self.mongo_db = client[MONGO_DB_NAME]

            # Create a tunnel to connect to the server from outside with VPN (MySql DB)
            self.mysql_ssh_tunnel_server = SSHTunnelForwarder(
                SERVER_IP,
                ssh_username=SERVER_USERNAME,
                ssh_password=SERVER_PASSWORD,
                remote_bind_address=('127.0.0.1', 3306)
            )

            # Start the tunnel server
            self.mysql_ssh_tunnel_server.start()

            # connect to the database
            self.mysql_conn = MySQLdb.connect(user=MYSQL_USERNAME,
                                              password=MYSQL_PASS,
                                              host='127.0.0.1',
                                              port=self.mysql_ssh_tunnel_server.local_bind_port,
                                              database=MYSQL_DB_NAME,
                                              charset='utf8', use_unicode=True, )

            self.mysql_cursor = self.mysql_conn.cursor()

    def _get_text_mongodb(self, project_id, collection_name="all_newcrawl_combined_271022"):
        """
        get text from MongoDB
        :param project_id: the ID of the project, can be string.
        :param collection_name: mongoDB collection name
        """

        all_docs = []
        # if not old_collection:
        cursor = self.mongo_db[collection_name]
        found_projects = cursor.find({"database_project_id": int(project_id)})
        for document in found_projects:
            del document['_id']
            all_docs.append(document)

        # query the DB
        documents = self.mongo_db.crawl20190109_translated.find({"mysql_databaseID": str(project_id)})
        documents2 = self.mongo_db.crawl20180801_wayback_translated.find({"mysql_databaseID": str(project_id)})
        for document in documents:
            all_docs.append(document)

        for document in documents2:
            all_docs.append(document)

        for doc in all_docs:
            doc['database_url'] = doc.get('url', '') or doc.get('database_url', '')
            doc['database_project_id'] = doc.get('mysql_databaseID', '') or doc.get('database_project_id', '')
            doc['projectname'] = doc.get('name', '') or doc.get('projectname', '')
            doc['content'] = doc.get('translation', '') or doc.get('content', '')

        return all_docs

    def _get_text_mysql(self, project_id):
        """
        get text from MySQl DB
        :param project_id: the ID of the project, can be string.
        :param do_postprocessing: to clean the text afterward
        :return: the text of the project
        """

        sql = f"""
        SELECT distinct(value) FROM {DATABASE_NAME}.AdditionalProjectData where FieldName NOT IN 
        ('Description_XLNet_02','Description_XLNet_Esramanual','Description_XLNet_01',
        'Description_XLNet_03','Description_XLNet_03_about') and Projects_idProjects={project_id}
        """

        # above, I added FieldName part with the 'in'to the original code - RA 04/08/2022
        self.mysql_cursor.execute(sql)
        list_of_tuples = self.mysql_cursor.fetchall()

        # Function to convert the output into a list of dicts. Each dict has the following keys
        output = get_list_of_dict(keys=['text'], list_of_tuples=list_of_tuples)

        mysql_text = ' '.join([x.get('text', '') for x in output]).strip()

        return mysql_text

    def load_training_project_ids(self, project_ids=None, top_x=None):
        """
        Function to load the dataset from MySQL database.
         It selects from table "esid_prediction" the projects that have been annotated by human
        """

        sql = f"""
        SELECT
        Project_id,
        CriterionActors,
        CriterionInnovativeness,
        CriterionObjectives,
        CriterionOutputs         
        FROM {DATABASE_NAME}.esid_prediction WHERE AnnSource='HUMAN'         
        """

        if project_ids and isinstance(project_ids, list):
            sql = sql + f" AND Project_id IN ({','.join([str(x) for x in project_ids])})"

        if top_x and isinstance(top_x, int):
            sql = sql + f' limit {top_x}'

        self.mysql_cursor.execute(sql)
        list_of_tuples = self.mysql_cursor.fetchall()

        # converts the output into a list of dicts
        projects = get_list_of_dict(keys=[
            'Project_id',
            'CriterionActors',
            'CriterionInnovativeness',
            'CriterionObjectives',
            'CriterionOutputs'
        ],
            list_of_tuples=list_of_tuples)

        # return projects: which is a list of dict. each element has 6 keys, they are:
        # 'Project_id',
        # 'CriterionActors',
        # 'CriterionInnovativeness',
        # 'CriterionObjectives',
        # 'CriterionOutputs'

        return projects

    def load_project_ids(self, run_name):
        """
        Load the project IDs which are not annotated by human yet --> to be annotated by machine.
        """

        all_p_sql = f"SELECT distinct(idProjects) FROM {DATABASE_NAME}.Projects"
        training_p_sql = f"SELECT distinct(Project_id) FROM {DATABASE_NAME}.esid_prediction WHERE AnnSource='HUMAN' OR" \
                         f" expName='{run_name}'"

        self.mysql_cursor.execute(all_p_sql)
        all_p_ids = [str(x[0]) for x in self.mysql_cursor.fetchall()]

        self.mysql_cursor.execute(training_p_sql)
        training_p_ids = [str(x[0]) for x in self.mysql_cursor.fetchall()]

        project_ids = set(all_p_ids).difference(training_p_ids)
        return list(project_ids)

    def get_project_text(self, project_id, mongodb_max_pages=10):
        """
        Function to extract the project text from MySQL and MongoDB
        """

        # extract text from MySQL
        mysql_text = self._get_text_mysql(project_id)

        # extract text from MongoDB
        project_data = self._get_text_mongodb(project_id)
        project_data_short = self.filter_project_pages(project_data, top_x=mongodb_max_pages)

        mongodb_text = ''
        for page in project_data_short:
            content = '\n'.join([c for c in page.get('content', '').split('\n') if len(c.split()) > 2])
            mongodb_text = mongodb_text + '\n' + content

        text_list = [l for l in (mysql_text + " " + mongodb_text).split('\n') if len(l.split()) > 2]
        text_list = list(dict.fromkeys(text_list))

        return '\n'.join(text_list)

    def project_already_exists(self, project_id, run_name):
        sql = f'SELECT * FROM {DATABASE_NAME}.esid_prediction ' \
              f'WHERE Project_id ="' + str(project_id) + '" AND expName="' + run_name + '"'
        self.mysql_cursor.execute(sql)
        rows = self.mysql_cursor.fetchall()
        if len(rows) > 0:
            return True
        return False

    def create_esid_predictions_table(self):
        self.mysql_cursor.execute(f"""
                CREATE TABLE {DATABASE_NAME}.esid_prediction (
                 Project_id TEXT,
                 CriterionObjectives TEXT,
                 CriterionActors TEXT,
                 CriterionOutputs TEXT,
                 CriterionInnovativeness TEXT,
                 Social_Innovation_overall TEXT,
                 AnnSource TEXT,
                 ModelName TEXT,
                 expName TEXT
                )
                """)
        self.mysql_conn.commit()
        print('table created')

    def insert_predictions(self, df_prediction):
        """
        function to save the classifier's output to the database
        param: df_prediction: is a dataframe contains the prediction result.
        The dataframe has the following columns:
        Project_id, CriterionObjectives, CriterionActors, CriterionOutputs, CriterionInnovativeness
        """

        projects_inserted_counter = 0
        projects_updated_counter = 0
        projects_failed = []

        # iterate over the dataframe and insert the rows one by one. If a project already exists, it will be updated
        for index, row in tqdm(df_prediction.iterrows(), "Inserting predictions into the DB"):
            Project_id = row['Project_id']
            CriterionObjectives = row['CriterionObjectives']
            CriterionActors = row['CriterionActors']
            CriterionOutputs = row['CriterionOutputs']
            CriterionInnovativeness = row['CriterionInnovativeness']
            Social_Innovation_overall = '-1'
            AnnSource = row['AnnSource']  # --> HUMAN or MACHINE
            ModelName = row['ModelName']
            expName = row['expName']

            if self.project_already_exists(project_id=Project_id, run_name=expName):
                sql = f"""
                UPDATE {DATABASE_NAME}.esid_prediction 
                SET                
                CriterionObjectives='{CriterionObjectives}',
                CriterionActors='{CriterionActors}',
                CriterionOutputs='{CriterionOutputs}',
                CriterionInnovativeness='{CriterionInnovativeness}',
                Social_Innovation_overall='{Social_Innovation_overall}',
                AnnSource='{AnnSource}',
                ModelName='{ModelName}'                
                WHERE
                Project_id='{Project_id}' and expName='{expName}' 
                """
            else:
                sql = f"""
                insert into {DATABASE_NAME}.esid_prediction (
                Project_id,
                CriterionObjectives,
                CriterionActors,
                CriterionOutputs,
                CriterionInnovativeness,      
                Social_Innovation_overall,      
                AnnSource,
                ModelName,
                expName
                ) VALUES (
                '{Project_id}',
                '{CriterionObjectives}',
                '{CriterionActors}',
                '{CriterionOutputs}',
                '{CriterionInnovativeness}',
                '{Social_Innovation_overall}',
                '{AnnSource}',
                '{ModelName}',
                '{expName}'
                )        
                """
            try:
                rows = self.mysql_cursor.execute(sql)

                if 'update' in sql.lower():
                    projects_updated_counter += rows
                else:
                    projects_inserted_counter += rows

            except:
                projects_failed.append(Project_id)
        try:
            self.mysql_conn.commit()
        except:
            return 0, len(df_prediction)
        return projects_inserted_counter, projects_updated_counter, projects_failed

    def free_select_sql(self, sql):
        self.mysql_cursor.execute(sql)
        res = self.mysql_cursor.fetchall()
        df = pd.DataFrame(res)
        df.to_csv('tmp123.csv')
        print(f'Done!. The table has {len(df)} items!')
        return res



if __name__ == '__main__':
    data_collector = DataCollector()
    # data_collector.load_project_ids('bert-base-cased_2022-10-12_07-36')
    # data_collector.load_testing_set()
    # data_collector.free_select_sql(f'SELECT * FROM {DATABASE_NAME}.esid_prediction')
    text = data_collector.get_project_text('25659')
    print(text)

    p_id = '25659'
    # from text_processing.text_processing_unit import TextProcessingUnit
    #
    # text_processing_unit = TextProcessingUnit()
    # t = text_processing_unit.clean_text(
    #                 text_processing_unit.shorten_text(
    #                     data_collector.get_project_text(p_id, mongodb_max_pages=10)
    #                 )
    #             )
    #
    # print(t)
