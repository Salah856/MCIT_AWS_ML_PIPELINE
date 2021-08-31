# Train an ML Model using Apache Spark in EMR and deploy in SageMaker

In this article, we will see how you can train your Machine Learning (ML) model using Apache Spark and then take the trained model artifacts to create an endpoint in SageMaker for online inference. Apache Spark is one of the most popular big-data analytics platforms & it also comes with an ML library with a wide variety of feature transformers and algorithms that one can use to build an ML model.

Apache Spark is designed for offline batch processing workload and is not best suited for low latency online prediction. In order to mitigate that, we will use MLeap library. MLeap provides an easy-to-use Spark ML Pipeline serialization format & execution engine for low latency prediction use-cases. Once the ML model is trained using Apache Spark in EMR, we will serialize it with MLeap and upload to S3 as part of the Spark job so that it can be used in SageMaker in inference.

After the model training is completed, we will use SageMaker Inference to perform predictions against this model. The underlying Docker image that we will use in inference is provided by sagemaker-sparkml-serving. It is a Spring based HTTP web server written following SageMaker container specifications and its operations are powered by MLeap execution engine.

We will work with Sparkmagic (PySpark) kernel while performing operations on the EMR cluster and in the second segment, we need to switch to conda_python2 kernel to invoke SageMaker APIs using sagemaker-python-sdk.

## Setup an EMR cluster and connect a SageMaker to the cluster

You will need to have an EMR cluster running and make sure that the notebook can connect to the master node of the cluster.

At this point, sagemaker-sparkml-serving only supports models trained with Spark version 2.2 for performing inference. Hence, please create an EMR cluster with Spark 2.2.0 or Spark 2.2.1 if you want to use your Spark ML model for online inference or batch transform.

Please follow the guide here on how to setup an EMR cluster and connect it to a notebook.
https://aws.amazon.com/blogs/machine-learning/build-amazon-sagemaker-notebooks-backed-by-spark-in-amazon-emr/ .

## Install additional Python dependencies and JARs in the EMR cluster

In order to serialize a Spark model with MLeap and upload to S3, we will need some additional Python dependencies and JAR present in the EMR cluster. Also, you need to setup your cluster with proper aws configurations.

### Configure aws credentials
First, please configure the aws credentials in all the nodes using aws configure

## Install the MLeap JAR in the cluster 

You need to have the MLeap JAR in the classpath to be successfully able to use it during model serialization. Please download the JAR (it is an assembly/fat JAR) from the following link using wget:

https://s3-us-west-2.amazonaws.com/sparkml-mleap/0.9.6/jar/mleap_spark_assembly.jar

and put it in /usr/lib/spark/jars in all the nodes.


### Install Python dependencies

After you have placed the JAR in the right location, please download a couple of necessary dependencies from PyPI. You have to download boto3 and mleap. You can run the below commands to download the dependencies from PyPI:

```sh
sudo python -m pip install boto3

sudo python -m pip install mleap
```

### Importing PySpark dependencies
Next we will import all the necessary dependencies that will be needed to execute the following cells on our Spark cluster. Please note that we are also importing the boto3 and mleap modules here.

You need to ensure that the import cell runs without any error to verify that you have installed the dependencies from PyPI properly. Also, this cell will provide you with a valid SparkSession named as spark.

```py
from __future__ import print_function

import os
import shutil
import boto3

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.types import StructField, StructType, StringType, DoubleType
from pyspark.ml.feature import (
    StringIndexer,
    VectorIndexer,
    OneHotEncoder,
    VectorAssembler,
    IndexToString,
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import *
from mleap.pyspark.spark_support import SimpleSparkSerializer

```

### Machine Learning task: Predict the age of an Abalone from its physical measurement 

The dataset is available from UCI Machine Learning. The aim for this task is to determine age of an Abalone (a kind of shellfish) from its physical measurements. At the core, it's a regression problem. The dataset contains several features - sex (categorical), length (continuous), diameter (continuous), height (continuous), whole_weight (continuous), shucked_weight (continuous), viscera_weight (continuous), shell_weight (continuous) and rings (integer).Our goal is to predict the variable rings which is a good approximation for age (age is rings + 1.5).

We'll use SparkML to pre-process the dataset (apply one or more feature transformers) and train it with the Random Forest algorithm from SparkML.

You can download the dataset from here using wget:

https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/abalone/abalone.csv

Alternatively, you can download the dataset from the UCI website as well and rename it to abalone.csv.

For this example, we will leverage EMR's capability to work directly with files residing in S3. Hence, after you download the data, you have to upload it to an S3 bucket in your account in the same region where your EMR cluster is running.

Alternatively, you can also use the HDFS storage in your EMR cluster to save this data.

### Define the schema of the dataset 

In the next cell, we will define the schema of the Abalone dataset and provide it to Spark so that it can parse the CSV file properly.

```py
schema = StructType(
    [
        StructField("sex", StringType(), True),
        StructField("length", DoubleType(), True),
        StructField("diameter", DoubleType(), True),
        StructField("height", DoubleType(), True),
        StructField("whole_weight", DoubleType(), True),
        StructField("shucked_weight", DoubleType(), True),
        StructField("viscera_weight", DoubleType(), True),
        StructField("shell_weight", DoubleType(), True),
        StructField("rings", DoubleType(), True),
    ]
)

```

Next we will use in-built CSV reader from Spark to read data directly from S3 into a Dataframe and inspect its first five rows.

After that, we will split the Dataframe into 80-20 train and validation so that we can train the model on the train part and measure its performance on the validation part.

```py
total_df = spark.read.csv(
    "s3://<your-input-bucket>/abalone/abalone.csv", header=False, schema=schema
)
total_df.show(5)
(train_df, validation_df) = total_df.randomSplit([0.8, 0.2])

```

### Define the feature transformers
Abalone dataset has one categorical column - sex which needs to be converted to integer format before it can be passed to the Random Forest algorithm.

For that, we are using StringIndexer and OneHotEncoder from Spark to transform the categorical column and then use a VectorAssembler to produce a flat one dimensional vector for each data-point so that it can be used with the Random Forest algorithm.

```py
sex_indexer = StringIndexer(inputCol="sex", outputCol="indexed_sex")

sex_encoder = OneHotEncoder(inputCol="indexed_sex", outputCol="sex_vec")

assembler = VectorAssembler(
    inputCols=[
        "sex_vec",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
    ],
    outputCol="features",

```

### Define the Random Forest model and perform training
After the data is preprocessed, we define a RandomForestClassifier, define our Pipeline comprising of both feature transformation and training stages and train the Pipeline calling .fit().

```py
rf = RandomForestRegressor(labelCol="rings", featuresCol="features", maxDepth=6, numTrees=18)

pipeline = Pipeline(stages=[sex_indexer, sex_encoder, assembler, rf])

model = pipeline.fit(train_df)

```

### Use the trained Model to transform train and validation dataset
Next we will use this trained Model to convert our training and validation dataset to see some sample output and also measure the performance scores.

The Model will apply the feature transformers on the data before passing it to the Random Forest.

```py
transformed_train_df = model.transform(train_df)

transformed_validation_df = model.transform(validation_df)

transformed_validation_df.select("prediction").show(5)

```

### Evaluating the model on train and validation dataset 

Using Spark's RegressionEvaluator, we can calculate the rmse (Root-Mean-Squared-Error) on our train and validation dataset to evaluate its performance. 

If the performance numbers are not satisfactory, we can train the model again and again by changing parameters of Random Forest or add/remove feature transformers.

```py
evaluator = RegressionEvaluator(labelCol="rings", predictionCol="prediction", metricName="rmse")

train_rmse = evaluator.evaluate(transformed_train_df)

validation_rmse = evaluator.evaluate(transformed_validation_df)

print("Train RMSE = %g" % train_rmse)
print("Validation RMSE = %g" % validation_rmse)

```

### Using MLeap to serialize the model 

By calling the serializeToBundle method from the MLeap library, we can store the Model in a specific serialization format that can be later used for inference by sagemaker-sparkml-serving.

```py
SimpleSparkSerializer().serializeToBundle(
    model, "jar:file:/tmp/model.zip", transformed_validation_df
)

```

SageMaker expects any model format to be present in tar.gz format, but MLeap produces the model zip format. In the next code, we unzip the model artifacts and store it in tar.gz format.

```py
import zipfile

with zipfile.ZipFile("/tmp/model.zip") as zf:
    zf.extractall("/tmp/model")

import tarfile

with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
    tar.add("/tmp/model/bundle.json", arcname="bundle.json")
    tar.add("/tmp/model/root", arcname="root")

```

At the end, we need to upload the trained and serialized model artifacts to S3 so that it can be used for inference in SageMaker.

Please note down the S3 location to where you are uploading your model.

```py
s3 = boto3.resource("s3")
file_name = os.path.join("emr/abalone/mleap", "model.tar.gz")
s3.Bucket("<your-output-bucket-name>").upload_file("/tmp/model.tar.gz", file_name)

```

If you are training multiple ML models on the same host and using the same location to save the MLeap serialized model, then you need to delete the model on the local disk to prevent MLeap library failing with an error - file already exists.

```py
os.remove("/tmp/model.zip")
os.remove("/tmp/model.tar.gz")
shutil.rmtree("/tmp/model")

```


## Hosting the model in SageMaker 

Hosting a model in SageMaker requires two components:

1. A Docker image residing in ECR.
2. A trained Model residing in S3.

We have to create an instance of SparkMLModel from sagemaker-python-sdk which will take the location of the model artifacts that we uploaded to S3 as part of the EMR job.

SparkML server also needs to know the payload of the request that'll be passed to it while calling the predict method. In order to alleviate the pain of not having to pass the schema with every request, sagemaker-sparkml-serving lets you to pass it via an environment variable while creating the model definitions.

This schema definition should also be passed while creating the instance of SparkMLModel.

```py
import json

schema = {
    "input": [
        {"name": "sex", "type": "string"},
        {"name": "length", "type": "double"},
        {"name": "diameter", "type": "double"},
        {"name": "height", "type": "double"},
        {"name": "whole_weight", "type": "double"},
        {"name": "shucked_weight", "type": "double"},
        {"name": "viscera_weight", "type": "double"},
        {"name": "shell_weight", "type": "double"},
    ],
    "output": {"name": "prediction", "type": "double"},
}
schema_json = json.dumps(schema)
print(schema_json)

```

```py
from time import gmtime, strftime
import time

timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

import sagemaker
from sagemaker import get_execution_role
from sagemaker.sparkml.model import SparkMLModel

sess = sagemaker.Session()
role = get_execution_role()

# S3 location of where you uploaded your trained and serialized SparkML model
sparkml_data = "s3://{}/{}/{}".format(
    "<your-output-bucket-name>", "emr/abalone/mleap", "model.tar.gz"
)
model_name = "sparkml-abalone-" + timestamp_prefix
sparkml_model = SparkMLModel(
    model_data=sparkml_data,
    role=role,
    sagemaker_session=sess,
    name=model_name,
    # passing the schema defined above by using an environment
    # variable that sagemaker-sparkml-serving understands
    env={"SAGEMAKER_SPARKML_SCHEMA": schema_json},
)


endpoint_name = "sparkml-abalone-ep-" + timestamp_prefix
sparkml_model.deploy(
    initial_instance_count=1, instance_type="ml.c4.xlarge", endpoint_name=endpoint_name
)

```

#### Passing the payload in CSV format

```py
from sagemaker.predictor import (
    json_serializer,
    csv_serializer,
    json_deserializer,
    RealTimePredictor,
)
from sagemaker.content_types import CONTENT_TYPE_CSV, CONTENT_TYPE_JSON

payload = "F,0.515,0.425,0.14,0.766,0.304,0.1725,0.255"

predictor = RealTimePredictor(
    endpoint=endpoint_name,
    sagemaker_session=sess,
    serializer=csv_serializer,
    content_type=CONTENT_TYPE_CSV,
    accept=CONTENT_TYPE_CSV,
)
print(predictor.predict(payload))

```

#### Passing the payload in JSON format 

```py
payload = {"data": ["F", 0.515, 0.425, 0.14, 0.766, 0.304, 0.1725, 0.255]}

predictor = RealTimePredictor(
    endpoint=endpoint_name,
    sagemaker_session=sess,
    serializer=json_serializer,
    content_type=CONTENT_TYPE_JSON,
    accept=CONTENT_TYPE_CSV,
)

print(predictor.predict(payload))

```

#### Passing the payload with both schema and the data

```py
payload = {
    "schema": {
        "input": [
            {"name": "length", "type": "double"},
            {"name": "sex", "type": "string"},
            {"name": "diameter", "type": "double"},
            {"name": "height", "type": "double"},
            {"name": "whole_weight", "type": "double"},
            {"name": "shucked_weight", "type": "double"},
            {"name": "viscera_weight", "type": "double"},
            {"name": "shell_weight", "type": "double"},
        ],
        "output": {"name": "prediction", "type": "double"},
    },
    "data": [0.515, "F", 0.425, 0.14, 0.766, 0.304, 0.1725, 0.255],
}

predictor = RealTimePredictor(
    endpoint=endpoint_name,
    sagemaker_session=sess,
    serializer=json_serializer,
    content_type=CONTENT_TYPE_JSON,
    accept=CONTENT_TYPE_CSV,
)

print(predictor.predict(payload))

```

Next we will delete the endpoint so that you do not incur the cost of keeping it running.

```py
boto_session = sess.boto_session
sm_client = boto_session.client("sagemaker")
sm_client.delete_endpoint(EndpointName=endpoint_name)

```

<a href="https://github.com/aws/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/sparkml_serving_emr_mleap_abalone"> Reference </a>
