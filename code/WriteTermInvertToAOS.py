#!/usr/bin/env python
# coding: utf-8

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
import boto3

AOS_ENDPOINT = 'vpc-chatbot-knn-3qe6mdpowjf3cklpj5c4q2blou.us-east-1.es.amazonaws.com'
INDEX_NAME = 'chatbot-invert-index'
REGION = 'us-east-1'

#credentials = boto3.Session().get_credentials()
#auth = AWSV4SignerAuth(credentials, REGION)

client = OpenSearch(
    hosts=[{'host': AOS_ENDPOINT, 'port': 443}],
    # http_auth = auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

def qa_file_to_json(file_path):
    """
    transfrom QA file to a json list
    :param file_path: localfile path
    :return: [{json}]
    """
    QA_json = []

    with open(file_path, 'r', encoding='utf8') as fp:
        para = []  # paragraph
        while True:
            line = fp.readline()
            # not a new paragraph
            if line != '\n':
                para.append(line)
            else:  # new paragraph to json
                QA_json.append({'question':para[0].split(': ')[1].strip(), 'answer': para[1].split(': ')[1].strip()})
                para = []
            if not line:
                if para != [] and para != ['']:
                    print(para)
                    QA_json.append({'question': para[0].split(': ')[1].strip(), 'answer': para[1].split(': ')[1].strip()})
                break
    return QA_json


def WriteTermInvertToAOS(qa_file, aos_endpoint=AOS_ENDPOINT, region=REGION, index_name=INDEX_NAME):
    """
    write paragraph to AOS for Term indexing.
    :param qa_file : Source QA file
    :param aos_endpoint : AOS endpoint
    :param index_name : AOS index name
    :return None
    """

    client = OpenSearch(
        hosts=[{'host': aos_endpoint, 'port': 443}],
        # http_auth = auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    qa_jsons = qa_file_to_json(qa_file)

    def yield_doc(idx_name):
        for doc in qa_jsons:
            yield {"_index": idx_name, "_source": doc}

    response = helpers.bulk(client, yield_doc(index_name))
    return response


if __name__ == '__main__':
    file_path = 'jy-faq.txt'
    WriteTermInvertToAOS(file_path)
