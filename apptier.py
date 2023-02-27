"""
@author: FinApples
"""

import boto3
import os
import time

# Setting Boto3 AWS credentials
region = 'us-east-1'
os.environ['AWS_ACCESS_KEY_ID'] = "AKIA3OBARTQ6GRCZUZ6V"
os.environ['AWS_SECRET_ACCESS_KEY'] = "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"

# SQS Queue URL's
REQUESTS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/786047278140/FinApples_Input"
RESPONSES_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/786047278140/FinApples_Output"

# S3 Bucket Names
INPUT_BUCKET = "finapples-input"
OUTPUT_BUCKET = "finapples-output"

# Clients
sqs = boto3.client('sqs', region)
s3 = boto3.client("s3")

def readMessage():
    # Receive message from SQS queue
    print("polling.....")
    response = sqs.receive_message(
        QueueUrl=REQUESTS_QUEUE_URL,
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'RequestID'
        ],
    )
    
    if response.get("Messages") is None:
        return None, None
    
    message = response['Messages'][0]
    receipt_handle = message['ReceiptHandle']
    requestId = message['MessageAttributes']['RequestID']['StringValue']
    file_name = message['Body']
    # Delete received message from queue
    sqs.delete_message(
        QueueUrl=REQUESTS_QUEUE_URL,
        ReceiptHandle=receipt_handle
    )
    
    print("received message attributes: ", message['MessageAttributes'])
    print("Received: ", file_name, " with RequestID: ", requestId)
    return file_name, requestId

def sendMessage(message, request_id):
    # Receive message from SQS queue
    response_queue_attrs = {
        'RequestID': {
            'StringValue': request_id,
            'DataType': 'String',
        }
    }
    response = sqs.send_message(
        QueueUrl=RESPONSES_QUEUE_URL,
        MessageBody=message,
        MessageAttributes=response_queue_attrs
    )
    
    print("message sent: ", message, " with RequestID: ", request_id)
    
def getFile(fileName):
    s3.download_file(Bucket=INPUT_BUCKET, Key=fileName, Filename=fileName)
    print("File downloaded: ", fileName)
    
def classify(fileName):
    import torch
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    from urllib.request import urlopen
    from PIL import Image
    import numpy as np
    import json
    import sys
    import time
    import warnings
    warnings.filterwarnings("ignore")

    url = fileName
    #img = Image.open(urlopen(url))
    img = Image.open(url)

    model = models.resnet18(pretrained=True)

    model.eval()
    img_tensor = transforms.ToTensor()(img).unsqueeze_(0)
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs.data, 1)

    with open('/home/ubuntu/imagenet-labels.json') as f:
        labels = json.load(f)
    result = labels[np.array(predicted)[0]]

    save_name = f"{fileName},{result}"
    with open(fileName[5:-4]+".txt","w") as f:
        f.write(save_name)
    return result

def uploadFile(fileName):
    s3.upload_file(
        Filename=fileName[5:-4]+".txt",
        Bucket=OUTPUT_BUCKET,
        Key=fileName[5:-4]+".txt",
    )
    
    print("File uploaded: ", fileName[5:-4]+".txt", " to bucket: ", OUTPUT_BUCKET, " with key: ", fileName)
    
def terminateInstance():
    ec2 = boto3.client("ec2", region)
    from ec2_metadata import ec2_metadata
    print(ec2_metadata.instance_id)
    instances = [ec2_metadata.instance_id]
    ec2.terminate_instances(InstanceIds=instances)

def main():
    while True:
        file, request_id = readMessage()
        if file == None:
            continue
        getFile(file)
        result = classify(file)
        uploadFile(file)
        sendMessage(result, request_id)
        os.remove(file)

main()