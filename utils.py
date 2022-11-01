import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score


def get_logger(run_name, run_type, log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # logging dir
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    file_handler = logging.FileHandler(f'{log_path}/{run_name}_{run_type}_{timestamp}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def return_df(df):
    """
    function to split dataframe into training/testing (80/20)
    :param df: dataframe to split
    :return: training and testing dataframes. Each has two columns: text/ labels
    """
    np.random.seed(42)
    msk = np.random.rand(len(df)) < 0.9
    train_df = df[msk]
    train_df = train_df.sample(frac=1)
    train_df.columns = ['Project_id', "text", "labels"]
    eval_df = df[~msk]
    eval_df.columns = ['Project_id', "text", "labels"]
    train_df = pd.DataFrame(
        {
            "Project_id": train_df['Project_id'],
            "text": train_df['text'].replace(r"\n", " ", regex=True),
            "labels": train_df['labels'].astype(int)
        }
    )

    eval_df = pd.DataFrame(
        {
            "Project_id": eval_df['Project_id'],
            "text": eval_df['text'].replace(r"\n", " ", regex=True),
            "labels": eval_df['labels'].astype(int)
        }
    )

    train_df = train_df[train_df['text'].notna()]
    eval_df = eval_df[eval_df['text'].notna()]

    return train_df, eval_df


def get_list_of_dict(keys, list_of_tuples):
    """
    This function will accept keys and list_of_tuples as args and return list of dicts
    """
    list_of_dict = [dict(zip(keys, values)) for values in list_of_tuples]
    return list_of_dict


def print_result(y, y_pred):
    """
    function to print results
    """

    result = {
        'macro F1': f1_score(y, y_pred, average='macro'),
        'macro recall': recall_score(y, y_pred, average='macro'),
        'macro precision': precision_score(y, y_pred, average='macro'),
        'micro F1': f1_score(y, y_pred, average='micro'),
        'micro recall': recall_score(y, y_pred, average='micro'),
        'micro precision': precision_score(y, y_pred, average='micro'),
        'accuracy': metrics.accuracy_score(y, y_pred)
    }

    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
    return result


def filter_project_pages(project_data, top_x=10):
    """
    When a project has many ULRs in MongoDB, we have to filter them. This function is to filter these pages and to
    take only the first level pages after the home page.
    :param project_data: a dictionary of the project webpages as collected from MongoDB
    :param top_x: number of minimum pages to be included. The URLs are sorted by their length.
    :returns a list of the selected pages content
    """

    blocking_words = ['policy', 'tos', 'legal', 'static', '.php?', 'web-content', 'login']

    if isinstance(project_data, dict):
        project_data = [project_data]

    project_data = [p for p in project_data if
                    p.get('content').strip() and
                    p.get('content', '').strip() != 'no content' and
                    len([p for kw in blocking_words if kw in p['url']]) == 0 and
                    p.get('language', '') in ['en', '']
                    ]

    if len(set([x['url'] for x in project_data])) <= top_x:
        return project_data

    new_project_data = []
    for p in project_data:
        p['content'] = p.get('content').strip()
        if p['url'][-1] == '/':
            p['url'] = p['url'][:-1]
        if p.get('content') == '' or len(p.get('content', '').split()) <= 2:
            continue
        else:
            new_project_data.append(p)

    tmp_urls = {x['url']: x.get('url').count('/') for x in new_project_data}
    urls = sorted(tmp_urls, key=tmp_urls.get, reverse=False)[:top_x]
    return [p for p in new_project_data if p['url'] in urls]
