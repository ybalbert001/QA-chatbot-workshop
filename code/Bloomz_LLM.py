#!/usr/bin/env python
# coding: utf-8
import json
import boto3
import numpy as np

ENDPOINT_NAME='bloomz-7b1-mt-2023-04-19-09-41-24-189-endpoint'

def Generate(smr_client, prompt):
    parameters = {
      # "early_stopping": True,
      "length_penalty": 100.0,
      "max_new_tokens": 200,
      "temperature": 0,
      "min_length": 20,
      "no_repeat_ngram_size": 200
    }

    response_model = smr_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            Body=json.dumps(
            {
                "inputs": prompt,
                "parameters": parameters
            }
            ),
            ContentType="application/json",
        )

    print(response_model['Body'].read().decode('utf8'))