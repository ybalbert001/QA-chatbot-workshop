import boto3
import json
import requests
import time
from collections import defaultdict
from requests_aws4auth import AWS4Auth
import os
from opensearchpy import OpenSearch, RequestsHttpConnection

source_includes = ["question","answer"]
runtime= boto3.client('runtime.sagemaker')
headers = { "Content-Type": "application/json" }


########parse k-NN search resule###############
# input:
#  r: AOS returned json
# return:
#  result : array of topN text
#############################################
def parse_results(r):
    res = []
    result = []
    for i in range(len(r['hits']['hits'])):
        h = r['hits']['hits'][i]
        if h['_source']['question'] not in clean:
            result.append(h['_source']['question'])
            res.append('<第'+str(i+1)+'条信息>'+h['_source']['question'] + '。</第'+str(i+1)+'条信息>\n')
    print(res)
    return result

########get embedding vector by SM llm########
# input:
#  questions:question texts(list)
#  sm_client: sagemaker runtime client
#  sm_endpoint:Sagemaker embedding llm endpoint
# return:
#  result : vector of embeded text
#############################################
def get_vector_by_sm_endpoint(questions,sm_client,endpoint_name,parameters):
    response_model = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "inputs": questions,
                "parameters": parameters
            }
        ),
        ContentType="application/json",
    )
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj['sentence_embeddings']
    return embeddings


########get embedding vector by lanchain vector search########
# input:
#  questions:question texts(list0
#  embedings:lanchain embeding models
# return:
#  result : vector of embeded text
#############################################################
def get_vector_by_lanchain(questions , embedings):
    doc_results = embeddings.embed_documents(questions)
    print(doc_results[0])
    return doc_results


########k-nn search by lanchain########
# input:
#  q:question text
#  vectorSearch: lanchain VectorSearch instance
# return:
#  result : k-NN search result
#############################################################
def search_using_lanchain(question, vectorSearch):
    docs = vectorSearch.similarity_search(query)
    return docs



########k-nn by native AOS########
# input:
#  q_embedding:embeded question text(array)
#  index:AOS k-NN index name
#  hostname: aos endpoint
#  username: aos username
#  passwd: aos password
#  source_includes: fields to return
#  k: topN
# return:
#  result : k-NN search result
#############################################################
def search_using_aos_knn(q_embedding, hostname, username,passwd, index, source_includes, size):
    print(1, q)
    awsauth = (username, passwd)
    query = {
        "size": num_output,
        "_source": {
            "includes": source_includes
        },
        "size": size,
        "query": {
            "knn": {
                "sentence_vector": {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    r = requests.post(hostname + index + '/_search', auth=awsauth, headers=headers, json=query)
    return r.text



########k-nn ingestion by native AOS########
# input:
#  docs:ingestion source documents
#  index:AOS k-NN index name
#  hostname: aos endpoint
#  username: aos username
#  passwd: aos password
# return:
#  result : N/A
#############################################################
def k_nn_ingestion_by_aos(docs,index,hostname,username,passwd):
    auth = (username, passwd)
    search = OpenSearch(
        hosts = [{'host': hostname, 'port': 443}],
        ##http_auth = awsauth ,
        http_auth = auth ,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    for doc in docs:
        vector_field = doc['sentence_vector']
        question_filed = doc['question']
        answer = doc['answer']
        document = { "question": question_filed, 'answer':answer_field, "sentence_vector": vector_field}
        search.index(index=index, body=document)


########k-nn ingestion by lanchain #########################
# input:
#  docs:ingestion source documents
#  vectorStore: lanchain AOS vectorStore instance
# return:
#  result : N/A
#############################################################
def k_nn_ingestion_by_lanchain(docs,vectorStore):
    for doc in docs:
        opensearch_vector_search.add_texts(docs,batch_size=10)