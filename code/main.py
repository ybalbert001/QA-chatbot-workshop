import json
import logging
import os
from botocore import config
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
import boto3
import time
import requests
import uuid
# from transformers import AutoTokenizer
from enum import Enum
from opensearchpy import OpenSearch, RequestsHttpConnection


logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client("sagemaker-runtime")
llm_endpoint = 'bloomz-7b1-mt-2023-04-19-09-41-24-189-endpoint'

class ErrorCode:
    DUPLICATED_INDEX_PREFIX = "DuplicatedIndexPrefix"
    DUPLICATED_WITH_INACTIVE_INDEX_PREFIX = "DuplicatedWithInactiveIndexPrefix"
    OVERLAP_INDEX_PREFIX = "OverlapIndexPrefix"
    OVERLAP_WITH_INACTIVE_INDEX_PREFIX = "OverlapWithInactiveIndexPrefix"
    INVALID_INDEX_MAPPING = "InvalidIndexMapping"


class APIException(Exception):
    def __init__(self, message, code: str = None):
        if code:
            super().__init__("[{}] {}".format(code, message))
        else:
            super().__init__(message)


def handle_error(func):
    """Decorator for exception handling"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIException as e:
            logger.exception(e)
            raise e
        except Exception as e:
            logger.exception(e)
            raise RuntimeError(
                "Unknown exception, please check Lambda log for more details"
            )

    return wrapper

# kendra


def query_kendra(Kendra_index_id="", lang="zh", search_query_text="what is s3?", Kendra_result_num=3):
    # 连接到Kendra
    client = boto3.client('kendra')

    # 构造Kendra查询请求
    query_result = client.query(
        IndexId=Kendra_index_id,
        QueryText=search_query_text,
        AttributeFilter={
            "EqualsTo": {
                "Key": "_language_code",
                "Value": {
                    "StringValue": lang
                }
            }
        }
    )
    # print(query_result['ResponseMetadata']['HTTPHeaders'])
    # kendra_took = query_result['ResponseMetadata']['HTTPHeaders']['x-amz-time-millis']
    # 创建一个结果列表
    results = []

    # 将每个结果添加到结果列表中
    for result in query_result['ResultItems']:
        # 创建一个字典来保存每个结果
        result_dict = {}

        result_dict['score'] = 0.0
        result_dict['doc_type'] = "P"

        # 如果有可用的总结
        if 'DocumentExcerpt' in result:
            result_dict['doc'] = result['DocumentExcerpt']['Text']
        else:
            result_dict['doc'] = ''

        # 将结果添加到列表中
        results.append(result_dict)

    # 输出结果列表
    return results[:Kendra_result_num]


# AOS
def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name):
    parameters = {
        # "early_stopping": True,
        # "length_penalty": 2.0,
        "max_new_tokens": 50,
        "temperature": 0,
        "min_length": 10,
        "no_repeat_ngram_size": 2,
    }

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

def search_using_aos_knn(q_embedding, hostname, index, source_includes, size):
    # awsauth = (username, passwd)
    # print(type(q_embedding))
    headers = {"Content-Type": "application/json"}
    query = {
        "size": size,
        "query": {
            "knn": {
                "embedding": {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    r = requests.post("https://"+hostname + "/" + index +
                        '/_search', headers=headers, json=query)
    return r.text

def aos_search(host, index_name, field, query_term):
    """
    search opensearch with query.
    :param host: AOS endpoint
    :param index_name: Target Index Name
    :param field: search field
    :param query_term: query term
    :return: aos response json
    """
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    query = {
        "query": {
            'match': {
                field: query_term
            }
        }
    }
    query_response = client.search(
        body=query,
        index=index_name
    )

    result_arr = [ {'doc':item['_source']['doc'], 'doc_type': item['_source']['doc_type'], 'score': item['_score']} for item in query_response["hits"]["hits"]]

    return result_arr
# def query_opensearch(api_url, query_type, query_params):
#     headers = {'Content-Type': 'application/json'}
#     if query_type == 'vector':  # 向量查询
#         query_json = {
#             "size": query_params['size'],
#             "query": {
#                 "knn": {
#                     query_params['field']: {
#                         "vector": query_params['vector_values'],
#                         "k": query_params['k']
#                     }
#                 }
#             }
#         }
#     elif query_type == 'match': # 字符串匹配查询
#         query_json = {
#             "size": query_params['size'],
#             "query": {
#                 "match": {
#                     query_params['field']: query_params['match_keyword']
#                 }
#             }
#         }

#     response = requests.post(api_url, headers=headers, data=json.dumps(query_json))
#     if response.status_code == 200:
#         # 请求成功
#         response_json = json.loads(response.text)
#         hits = response_json['hits']['hits']
#         return [hit['_source'] for hit in hits]
#     else:
#         # 请求失败
#         print(f"请求失败: {response.status_code} - {response.text}")
#         return []


# def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name, parameters):
#     response_model = sm_client.invoke_endpoint(
#         EndpointName=endpoint_name,
#         Body=json.dumps(
#             {
#                 "inputs": questions,
#                 "parameters": parameters
#             }
#         ),
#         ContentType="application/json",
#     )
#     json_str = response_model['Body'].read().decode('utf8')
#     json_obj = json.loads(json_str)
#     embeddings = json_obj['sentence_embeddings']
#     return embeddings


# def search_using_aos_knn(q_embedding, hostname, index, source_includes, size):
#     # awsauth = (username, passwd)
#     print(type(q_embedding))
#     query = {
#         "size": size,
#         "query": {
#             "knn": {
#                 "sentence_vector": {
#                     "vector": q_embedding,
#                     "k": size
#                 }
#             }
#         }
#     }
#     r = requests.post("https://"+hostname + "/" + index +
#                       '/_search', headers=headers, json=query)
#     return r.text

# DDB


def get_session(session_id):

    table_name = "chatbot-session"
    dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb.Table(table_name)
    operation_result = ""

    response = table.get_item(Key={'session-id': session_id})

    if "Item" in response.keys():
        print("****** " + response["Item"]["content"])
        operation_result = json.loads(response["Item"]["content"])
    else:
        print("****** No result")
        operation_result = ""

    return operation_result


# param:    session_id
#           question
#           answer
# return:   success
#           failed
def update_session(session_id, question, answer, intention):

    table_name = "chatbot-session"
    dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb.Table(table_name)
    operation_result = ""

    response = table.get_item(Key={'session-id': session_id})

    if "Item" in response.keys():
        print("****** " + response["Item"]["content"])
        chat_history = json.loads(response["Item"]["content"])
    else:
        print("****** No result")
        chat_history = []

    chat_history.append([question, answer, intention])
    content = json.dumps(chat_history)

    # inserting values into table
    response = table.put_item(
        Item={
            'session-id': session_id,
            'content': content
        }
    )

    if "ResponseMetadata" in response.keys():
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            operation_result = "success"
        else:
            operation_result = "failed"
    else:
        operation_result = "failed"

    return operation_result





def Generate(smr_client, llm_endpoint, prompt):
    parameters = {
        # "early_stopping": True,
        "length_penalty": 100.0,
        "max_new_tokens": 200,
        "temperature": 0,
        "min_length": 20,
        "no_repeat_ngram_size": 200
    }

    response_model = smr_client.invoke_endpoint(
        EndpointName=llm_endpoint,
        Body=json.dumps(
            {
                "inputs": prompt,
                "parameters": parameters
            }
        ),
        ContentType="application/json",
    )
    
    json_ret = json.loads(response_model['Body'].read().decode('utf8'))

    return json_ret['outputs']


class QueryType(Enum):
    KeywordQuery   = "KeywordQuery"       #用户仅仅输入了一些关键词（2 token)
    KnowledgeQuery = "KnowledgeQuery"     #用户输入的需要参考知识库有关来回答
    Conversation   = "Conversation"       #用户输入的是跟知识库无关的问题


def intention_classify(post_text, prompt_template, few_shot_example):
    prompt = prompt_template.format(
        fewshot=few_shot_example, question=post_text)
    print(prompt)
    result = Generate(sm_client, llm_endpoint, prompt)
    print(result)
    len_prompt = len(prompt)
    return result[len_prompt:]

# different scan
def prompt_build(post_text, opensearch_respose, opensearch_knn_respose, kendra_respose, conversations, tokenizer):
    """
    Detect User intentions, build prompt for LLM. For Knowledge QA, it will merge all retrieved related document paragraphs into a single prompt
    Parameters examples:
        post_text : "你好么？"
        opensearch_respose: [{"score" : 0.7, "doc": "....", "doc_type": "Q|A|P"}]
        opensearch_knn_respose: [
            {"score" : 0.7, "doc": "....", "doc_type": "Q|A|P"}
        ]
        kendra_respose: [{"score" : 0.7, "doc": "....", "doc_type": "Q|A|P"}]
        conversations: [["Q1", "A1",'Qtype1'], ("Q1", "A1",'Qtype1'), ...]
        tokenizer: which aim to calculate length of query's token
    return: prompt string
    """
    # prompt templates:
    # LLM
    Game_Intention_Classify_Examples="""玩家输入: 介绍一下联盟?
    输出: 功能相关

    玩家输入: 介绍一下强化部件?
    输出: 功能相关

    玩家输入:我玩你们的游戏导致失恋了
    输出: 功能无关

    玩家输入: 我要给你们提个建议！
    输出: 功能无关

    玩家输入:怎么才能迁移区服？
    输出: 功能相关"""

    Game_Intention_Classify_Prompt = """
    任务定义: 判断玩家输入是否在询问游戏功能相关的问题, 回答 "功能相关" 或者 "功能无关".

    {fewshot}

    玩家输入:{question}
    输出: """

    Game_Knowledge_QA_Prompt = """
    Jarvis 是一个游戏智能客服，能够回答玩家的各种问题，以及陪用户聊天，比如

    {fewshot}

    玩家:{question}
    Jarvis:"""

    Game_Free_Chat_Examples ="""玩家: 我要给你们提个建议！
    Jarvis: 感谢您的支持，我们对玩家的建议都是非常重视的，您可以点击右上角联系客服-我要提交建议填写您的建议内容并提交，我们会转达给团队进行考量。建议提交

    玩家: 我玩你们的游戏导致失恋了
    Jarvis:亲爱的玩家，真的非常抱歉听到这个消息，让您受到了困扰。感情的事情确实很复杂，但请相信，时间会治愈一切。请保持乐观积极的心态，也许未来会有更好的人陪伴您。我们会一直陪在您身边，为您提供游戏中的支持与帮助。请多关注自己的生活，适当调整游戏与生活的平衡。祝您生活愉快，感情美满~(づ｡◕‿‿◕｡)づ
    """

    Game_Free_Chat_Prompt = """
    Jarvis 是一个游戏智能客服，能够回答玩家的各种问题，以及陪用户聊天，比如

    {fewshot}

    {chat_history}

    玩家: {question}
    Jarvis:"""

    q_type = None
    final_prompt = ""

    # tokens = tokenizer.encode(post_text)

    # if user input exactly match special term, return the special term's description directly
    if (post_text in ["强化备件", "联盟"]): # reimplement this judge later
        q_type = QueryType.KeywordQuery
        if opensearch_respose:
            final_prompt = opensearch_respose[-1]
    else:  # NormalQuery
        detect_intention = intention_classify(post_text, Game_Intention_Classify_Prompt, Game_Intention_Classify_Examples)
        print(f"detect_intention : {detect_intention}")
        if detect_intention == "功能无关":
            chat_histories = '\n\n'.join(['\n'.join(conversation[:2]) for conversation in conversations])
            final_prompt = Game_Free_Chat_Prompt.format(fewshot=Game_Free_Chat_Examples, chat_history=chat_histories, question=post_text)
            q_type = QueryType.Conversation
        else:  # detect_intention == "功能相关"
            # Combine opensearch_knn_respose and kendra_respose
            recall_dict = {}
            # for recall_item in opensearch_respose:
            #     if recall_item["doc"] in recall_item.keys():
            #         if recall_item["score"] > recall_dict[recall_item["doc"]]:
            #             recall_dict[recall_item["doc"]] = recall_item["score"]
            #     else:
            #         recall_dict[recall_item["doc"]] = recall_item["score"]

            for recall_item in opensearch_knn_respose:
                if recall_item["doc_type"] != 'P':
                    continue

                if recall_item["doc"] in recall_item.keys():
                    if recall_item["score"] > recall_dict[recall_item["doc"]]:
                        recall_dict[recall_item["doc"]] = recall_item["score"]
                else:
                    recall_dict[recall_item["doc"]] = recall_item["score"]

            for recall_item in kendra_respose:
                if recall_item["doc"] in recall_item.keys():
                    if recall_item["score"] > recall_dict[recall_item["doc"]]:
                        recall_dict[recall_item["doc"]] = recall_item["score"]
                else:
                    recall_dict[recall_item["doc"]] = recall_item["score"]

            example_list = [k for k, v in sorted(
                recall_dict.items(), key=lambda item: item[1])]
            q_type = QueryType.KnowledgeQuery
            prompt_context = '\n\n'.join(example_list)

            final_prompt = Game_Knowledge_QA_Prompt.format(fewshot=prompt_context, question=post_text)
            final_prompt = final_prompt.replace("Question", "玩家").replace("Answer", "Jarvis")

    json_obj = {
        "query": post_text,
        "opensearch_doc":  opensearch_respose,
        "opensearch_knn_doc":  opensearch_knn_respose,
        "kendra_doc": kendra_respose,
        "detect_query_type": str(q_type),
        "LLM_input": final_prompt
    }

    return q_type, final_prompt, json_obj

def main_entry(session_id:str, query_input:str, embedding_model_endpoint:str, llm_model_endpoint:str, aos_endpoint:str, aos_index:str, aos_knn_field:str, aos_result_num:int, kendra_index_id:str, kendra_result_num:int):
    """
    Entry point for the Lambda function.

    Parameters:
        session_id (str): The ID of the session.
        query_input (str): The query input.
        embedding_model_endpoint (str): The endpoint of the embedding model.
        llm_model_endpoint (str): The endpoint of the language model.
        aos_endpoint (str): The endpoint of the AOS engine.
        aos_index (str): The index of the AOS engine.
        aos_knn_field (str): The knn field of the AOS engine.
        aos_result_num (int): The number of results of the AOS engine.
        kendra_index_id (str): The ID of the Kendra index.
        kendra_result_num (int): The number of results of the Kendra Service.

    return: answer(str)
    """
    sm_client = boto3.client("sagemaker-runtime")
    
    # 1. get_session
    session_history = get_session(session_id=session_id)

    # 2. get kendra recall 
    kendra_respose = query_kendra(kendra_index_id, "zh", query_input, kendra_result_num)

    # 3. get AOS knn recall 
    query_embedding = get_vector_by_sm_endpoint(query_input, sm_client, embedding_model_endpoint)
    source_includes = ["doc_type", "doc"]
    knn_result = search_using_aos_knn(query_embedding[0], aos_endpoint, aos_index, source_includes, aos_result_num)
    result = json.loads(knn_result)["hits"]["hits"]
    opensearch_knn_respose = []
    for item in result[:3]:
        opensearch_knn_respose.append( {"doc": item["_source"]["doc"],"doc_type":item["_source"]["doc_type"],"score":item["_score"]} )
    
    # 4. todo: get AOS invertedIndex recall
    opensearch_query_response = aos_search(aos_endpoint, aos_index, "doc", query_input)
    logger.info(opensearch_query_response)

    # 5. build prompt
    TOKENZIER_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    tokenizer = None # AutoTokenizer.from_pretrained(TOKENZIER_MODEL_NAME)

    query_type, prompt_data, json_obj = prompt_build(post_text=query_input, opensearch_respose=opensearch_query_response, opensearch_knn_respose=opensearch_knn_respose,
                                  kendra_respose=kendra_respose, conversations=session_history, tokenizer=tokenizer)
    
    answer = None
    try:
        llm_generation = Generate(sm_client, llm_endpoint, prompt=prompt_data)
        answer = llm_generation[len(prompt_data):]
        json_obj['session_id'] = session_id
        json_obj['chatbot_answer'] = answer
        json_obj['conversations'] = session_history
        json_obj['log_type'] = "all"
        json_obj_str = json.dumps(json_obj, ensure_ascii=False)
        logger.info(json_obj_str)
    except Exception as e:
        logger.info(f'Exceptions: str({e})')
    finally:
        json_obj_str = json.dumps(json_obj, ensure_ascii=False)
        logger.info(json_obj_str)

    update_session(session_id=session_id, question=query_input, answer=answer, intention=str(query_type))

    return answer


@handle_error
def lambda_handler(event, context):

    # "model": 模型的名称
    # "chat_name": 对话标识，后端用来存储查找实现多轮对话 session
    # "prompt": 用户输入的问题
    # "max_tokens": 2048
    # "temperature": 0.9
    logger.info(f"event:{event}")
    session_id = event['chat_name']
    question = event['prompt']

    # 获取当前时间戳
    request_timestamp = int(time.time())  # 或者使用 time.time_ns() 获取纳秒级别的时间戳
    logger.info(f'request_timestamp :{request_timestamp}')
    logger.info(f"event:{event}")
    logger.info(f"context:{context}")

    # 创建日志组和日志流
    log_group_name = '/aws/lambda/{}'.format(context.function_name)
    log_stream_name = context.aws_request_id
    client = boto3.client('logs')
    # 接收触发AWS Lambda函数的事件
    logger.info('The main brain has been activated, aws🚀!')

    # 1. 获取环境变量

    embedding_endpoint = os.environ.get("embedding_endpoint", "")
    aos_endpoint = os.environ.get("aos_endpoint", "")
    aos_index = os.environ.get("aos_index", "")
    aos_knn_field = os.environ.get("aos_knn_field", "")
    aos_result_num = int(os.environ.get("aos_results", ""))

    Kendra_index_id = os.environ.get("Kendra_index_id", "")
    Kendra_result_num = int(os.environ.get("Kendra_result_num", ""))
    # Opensearch_result_num = int(os.environ.get("Opensearch_result_num", ""))

    logger.info(f'embedding_endpoint : {embedding_endpoint}')
    logger.info(f'aos_endpoint : {aos_endpoint}')
    logger.info(f'aos_index : {aos_index}')
    logger.info(f'aos_knn_field : {aos_knn_field}')
    logger.info(f'aos_result_num : {aos_result_num}')
    logger.info(f'Kendra_index_id : {Kendra_index_id}')
    logger.info(f'Kendra_result_num : {Kendra_result_num}')
    
    answer = main_entry(session_id, question, embedding_endpoint, llm_endpoint, aos_endpoint, aos_index, aos_knn_field, aos_result_num,
                       Kendra_index_id, Kendra_result_num)

    # 2. return rusult

    # 处理

    # Response:
    # "id": "设置一个uuid"
    # "created": "1681891998"
    # "model": "模型名称"
    # "choices": [{"text": "模型回答的内容"}]
    # "usage": {"prompt_tokens": 58, "completion_tokens": 15, "total_tokens": 73}}]

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': [{"id": str(uuid.uuid4()),
                             "created": request_timestamp,
                             "useTime": int(time.time()) - request_timestamp,
                             "model": "main_brain",
                             "choices":
                             [{"text": answer}],
                             "usage": {"prompt_tokens": 58, "completion_tokens": 15, "total_tokens": 73}},
                            # {"id": uuid.uuid4(),
                            #  "created": request_timestamp,
                            #  "useTime": int(time.time()) - request_timestamp,
                            #  "model": "模型名称",
                            #  "choices":
                            #  [{"text": "2 模型回答的内容"}],
                            #  "usage": {"prompt_tokens": 58, "completion_tokens": 15, "total_tokens": 73}}
                            ]
    }

