#!/bin/bash
# How to get S3: Access Key / Secret Access Key
# 1. Open the IAM console, Select your IAM user name
# 2. Click User Actions, and then click Manage Access Keys.
# 3. Click Create Access Key.
# 4. Copy the keys, alternatively download the file `accessKeys.csv`

ACCESS_KEY="<fill-in>"
SECRET_KEY="<fill-in>"
REGION="<fill-in>"
BUCKET="<fill-in>"
KEY="<fill-in>"

python ./s3_file_source.py --aws-access-key-id $ACCESS_KEY \
                 --aws-secret-access-key $SECRET_KEY \
                 --region $REGION \
                 --bucket $BUCKET \
                 --key $KEY

