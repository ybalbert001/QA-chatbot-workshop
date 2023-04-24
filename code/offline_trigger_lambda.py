import json
import boto3

def lambda_handler(event, context):

    glue = boto3.client('glue')

    bucket = event['Records'][0]['s3']['bucket']['name']
    object_key = event['Records'][0]['s3']['object']['key']
    
    print("**** in lambda : " + bucket)
    print("**** in lambda : " + object_key)

    glue.start_job_run(JobName='chatbot-from-s3-to-aos', Arguments={"--bucket": bucket, "--object_key": object_key})

    return {
        'statusCode': 200,
        'body': json.dumps('Successful ')
    }
