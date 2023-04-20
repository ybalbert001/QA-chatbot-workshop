#!/usr/bin/env python
# coding: utf-8

import boto3
from transformers import AutoTokenizer
from enum import Enum
from prompt_template import Game_Intention_Classify_Prompt
from Bloomz_LLM import Generate

TOKENZIER_MODEL_NAME='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

tokenizer = AutoTokenizer.from_pretrained(TOKENZIER_MODEL_NAME)
smr_client = boto3.client("sagemaker-runtime")

class QueryType(Enum):
    KeyWordOnly = 1   #用户仅仅输入了一些关键词（2 token)
    NormalQuery = 2   #用户输入的需要参考知识库有关来回答
    NonKnowledge = 3  #用户输入的是跟知识库无关的问题

def intention_classify(post_text, prompt_template, few_shot_example):   
    prompt = prompt_template.format(fewshot=few_shot_example, question=post_text)
    result = Generate(smr_client, prompt)
    len_prompt = len(prompt)
    return result[len_prompt:]

def prompt_build(post_text, opensearch_respose, opensearch_knn_respose, kendra_respose, conversations, tokenizer):
    """
    Merge all retrieved related document paragraphs into a single prompt
    opensearch_respose: [{"score" : 0.7, "doc": "....", "doc_type": "Q|A|P"}]
    opensearch_knn_respose: [{"score" : 0.7, "doc": "....", "doc_type": "Q|A|P"}]
    kendra_respose: [{"score" : 0.7, "doc": "....", "doc_type": "Q|A|P"}]
    conversations: [ ("Q1", "A1"), ("Q1", "A1"), ...]
    tokenizer: which aim to calculate length of query's token
    return: prompt string
    """
    q_type = QueryType.NormalQuery

    tokens = tokenizer.encode(post_text)
    if (len(tokens) <= 2):
        q_type = QueryType.NormalQuery
        if opensearch_respose:
            prompt = opensearch_respose[-1]
            return q_type, prompt
    else: # NormalQuery
        if "功能无关" == intention_classify(post_text, Game_Intention_Classify_Prompt):
            return QueryType.NonKnowledge, "\n".join(conversations)
        else: # 功能相关，
            # Combine opensearch_knn_respose and kendra_respose
            recall_dict = {}
            for recall_item in opensearch_respose:
                if recall_item["doc"] in recall_item.keys():
                    if recall_item["score"] > recall_dict[recall_item["doc"]]:
                        recall_dict[recall_item["doc"]] = recall_item["score"]

            for recall_item in opensearch_knn_respose:
                if recall_item["doc"] in recall_item.keys():
                    if recall_item["score"] > recall_dict[recall_item["doc"]]:
                        recall_dict[recall_item["doc"]] = recall_item["score"]

            for recall_item in kendra_respose:
                if recall_item["doc"] in recall_item.keys():
                    if recall_item["score"] > recall_dict[recall_item["doc"]]:
                        recall_dict[recall_item["doc"]] = recall_item["score"]

            example_list = [ k for k, v in sorted(recall_dict.items(), key=lambda item: item[1])]
            return QueryType.NormalQuery, '\n'.join(example_list)

if __name__ == '__main__':
    post_text = "介绍一下强化部件？"
    # opensearch_respose = [
    #     {"score":0.7, "doc": ""}
    # ]