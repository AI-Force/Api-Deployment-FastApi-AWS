# Deploying Machine Learning Models with FastAPI and Docker on AWS

## Introduction

Many blog posts and tutorials focus on the training of machine learning and deep learning models. However, there is not a lot of information about how to create and deploy these models so they can be easily used as part of a website or another system.

This blog post describes approaches for creating web APIs for these models, and their deployment to AWS.

Although the focus of this post is on [TensorFlow 2](https://www.tensorflow.org/), with a few modifications the code can also support other frameworks such as [PyTorch](https://pytorch.org/) or [LightGBM](https://lightgbm.readthedocs.io/en/latest/).

## About Me

My name is Adam and I develop machine learning / deep learning systems and prototypes at the AI Technology R&D Division of Proto Solution.

Most of my work focuses on the development of deep learning models using TensorFlow, and using these to develop APIs or batch applications that can be deployed onto Amazon Web Services (AWS).

## Approaches

There are a number of AWS services that can be used to deploy machine learning models, for example;

- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [AWS Lambda](https://aws.amazon.com/lambda/)
- [AWS Elastic Beanstalk](https://aws.amazon.com/elasticbeanstalk/)
- [AWS Batch](https://aws.amazon.com/batch/)
- [AWS Elastic Container Service (ECS)](https://aws.amazon.com/ecs/)

Each of these services vary in cost, complexity and features.

For this blog we will focus on deploying to **AWS Lambda** and **AWS Elastic Beanstalk**; I find these services the easiest to use and most of my prototypes and applications are deployed onto them.

I hope to focus on other services in a future blog.

## API

An, API &mdash; Application Programming Interface &mdash; describes how a user or another system can communicate with the application without having to know its internal details.

A web API allows an application to expose its functionality over the internet using HTTP or other protocols.

In this tutorial we will create a web API using the FastAPI framework and use Docker and AWS services to deploy it. 

We will use a Tensorflow [MobileNetV2](https://arxiv.org/abs/1801.04381) model that has been trained on the [ImageNet](https://www.image-net.org/) dataset as a example.

## FastAPI

[FastAPI](https://fastapi.tiangolo.com/) is a web-framework for building web APIs in Python.

Although other frameworks, such as [Flask](https://flask.palletsprojects.com/en/2.0.x/), are also very popular I believe that FastAPI has a number of advantages over these, in particular:

- It is really fast and lightweight.
- Builtin data serialization and validation with [Pydantic](https://pydantic-docs.helpmanual.io/).
- Builtin Open API documentation generation.

## Docker

[Docker](https://www.docker.com/) has become the defacto way to deploy applications. 

It allows developers to package the application and all its dependencies, including the operating system, into what is known as a Docker image.

These images can be run on a variety of different platforms that support the Docker engine. 

Images that are run on the Docker engine are known as *containers*.

Docker provides a number of different commands for managing containers, but for this tutorial we will be just be using the [build](https://docs.docker.com/engine/reference/commandline/build/) and [run](https://docs.docker.com/engine/reference/commandline/run/) commands.

## Tutorial

In this section we will introduce and describe each part of the API. The full code will be available at the end of this section.

### Install Dependencies

Our first step is to create a new virtual environment &mdash;an isolated, working copy of Python. 

For this tutorial, I will use [Conda](https://docs.conda.io/en/latest/) to create a new Python 3.7 virtual environment and activate it.

```bash
conda create -n ApiDeployment python=3.7
conda activate ApiDeployment
```

We can then install the dependencies, listed in the `requirements.txt` file into the virtual environment:

```bash
pip install -r requirements.txt
```

Let's have a look at some of the dependent packages that we will be using:

```bash
Pillow==8.2.0
tensorflow==2.4.2
numpy==1.19.5
fastapi==0.65.2
pydantic==1.8.2
aiohttp==3.7.3
uvloop==0.14.0
uvicorn[standard]==0.14.0
gunicorn==20.1.0
aiofiles==0.7.0
mangum==0.12.2
```

### Imports and Logging

The first thing we need to do is import the libraries that the API will use.

I will explain their usage as we go through the code.

```python
import argparse
import base64
import io
import os
import logging
import sys

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

from urllib.parse import urlparse

from aiohttp.client import ClientSession
from asyncio import wait_for, gather, Semaphore

from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel, validator

import numpy as np

from PIL import Image

from mangum import Mangum
```

The API will be configured using environment variables; these can be easily passed to Docker on the command line.

```python
THREAD_COUNT = int(os.environ.get('THREAD_COUNT', 5))
"""The number of threads used to download and process image content."""

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
"""The number of images to process in a batch."""

TIMEOUT = int(os.environ.get('TIMEOUT', 30))
"""The timeout to use when downloading files."""
```

We will also create a `logging` object to write log messages. 

```python
logger = logging.getLogger(__name__)
```

### Data Model

Our next step is to create the data structures that define the interface to our API by using the [Pydantic](https://pydantic-docs.helpmanual.io/) package.

Detailed tutorials for creating the data structures can be found [here](https://pydantic-docs.helpmanual.io/usage/models/). 

```python
class HealthCheck(BaseModel):
    """
    Represents an image to be predicted.
    """
    message: Optional[str] = 'OK'


class ImageInput(BaseModel):
    """
    Represents an image to be predicted.
    """
    url: Optional[str] = None
    data: Optional[str] = None


class ImageOutput(BaseModel):
    """
    Represents the result of a prediction
    """
    score: Optional[float] = 0.0
    category: Optional[str] = None
    name: Optional[str] = None

    @validator('score')
    def result_check(cls, v):
        return round(v, 4)


class PredictRequest(BaseModel):
    """
    Represents a request to process
    """
    images: List[ImageInput] = []


class PredictResponse(BaseModel):
    """
    Represents a request to process
    """
    images: List[ImageOutput] = []
```

The `PredictRequest` object represents the data that is passed into our API; that is, the images that we want to process.

The `PredictResponse` object defines how the prediction results are returned back to the caller of the API.

By defining these structures, we can automatically deserialize and validate [JSON](https://www.json.org/) requests containing image URLs or Base64 encoded image data. For example:

```json
{
  "images": [
    {
      "url": "https://localhost/test.jpg"
    }
  ]
}
```

We can also serialize a JSON response containing the ImageNet category prediction, and confidence score for each image. For example:

```json
{
    "images": [
        {
            "score": 0.508,
            "category": "n03770679",
            "name": "minivan"
        }
    ]
}
```

### FastAPI Application

The next step is to instantiate the FastAPI application object; this allows us to use a number of annotations described in the next steps.

```python
app = FastAPI()
```

### Exception Handlers

The following exception handlers are used generate error messages, that will be returned to the caller, if an error occurs during processing. 

```python
class ImageNotDownloadedException(Exception):
    pass


@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for all other errors.
    """
    return JSONResponse(status_code=500, content={'message': 'Internal error.'})


@app.exception_handler(ImageNotDownloadedException)
async def client_exception_handler(request: Request, exc: ImageNotDownloadedException):
    """
    Called when the image could not be downloaded.
    """
    return JSONResponse(status_code=400, content={'message': 'One or more images could not be downloaded.'})
```

### Image Classifier

The following class defines and downloads a pretrained MobileNetV2 model.

By calling the `predict` function we can use the model to predict the ImageNet categories for a list of images. 

Images are pre-processed and resized to fit the input of the model (224 pixels x 224 pixels).

```python
class ImageClassifier:
    """
    Classifies images according to ImageNet categories.
    """
    def __init__(self):
        """
        Prepares the model used by the application for use.
        """
        self.model = MobileNetV2()
        _, height, width, channels = self.model.input_shape
        self.input_width = width
        self.input_height = height
        self.input_channels = channels

    def _prepare_images(self, images):
        """
        Prepares the images for prediction.

        :param images: The list of images to prepare for prediction in Pillow Image format.

        :return: A list of processed images.
        """
        batch = np.zeros((len(images), self.input_height, self.input_width, self.input_channels), dtype=np.float32)
        for i, image in enumerate(images):
            x = image.resize((self.input_width, self.input_height), Image.BILINEAR)
            batch[i, :] = np.array(x, dtype=np.float32)
        batch = preprocess_input(batch)
        return batch

    def predict(self, images, batch_size):
        """
        Predicts the category of each image.

        :param images: A list of images to classify.
        :param batch_size: The number of images to process at once.

        :return: A list containing the predicted category and confidence score for each image.
        """
        batch = self._prepare_images(images)
        scores = self.model.predict(batch, batch_size)
        results = decode_predictions(scores, top=1)
        return results
```

### Configure Logging

Next, we will create a function to configure the logger for the application.

All messages will be output to `stdout`. This will allow messages to be easily logged by AWS Lambda and Elastic Beanstalk later on.
 
```python
def configure_logging(logging_level=logging.INFO):
    """
    Configures logging for the application.
    """
    root = logging.getLogger()
    root.handlers.clear()
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    root.setLevel(logging_level)
    root.addHandler(stream_handler)
```


### Load Model

This step will create and load the `ImageClassifier` object when the application is started.

It will also configure the logging for the application using the function defined in the previous section.

```python
@app.on_event('startup')
def load_model():
    """
    Loads the model prior to the first request.
    """
    configure_logging()
    logger.info('Loading model...')
    app.state.model = ImageClassifier()
```

We will store the `ImageClassifier` object in the application `state`, but it is also possible to store the model as a global variable instead.

### Image Processing

In this section we define a number of functions to download images from a URL or decode Base64 image data stored in the request message.

We will use the `aiohttp` package to perform the download and convert the data into `Pillow` images. 

```python
def get_url_scheme(url, default_scheme='unknown'):
    """
    Returns the scheme of the specified URL or 'unknown' if it could not be determined.
    """
    result = urlparse(url, scheme=default_scheme)
    return result.scheme


async def retrieve_content(entry, sess, sem):
    """
    Retrieves the image content for the specified entry.
    """
    raw_data = None
    if entry.data is not None:
        raw_data = base64.b64decode(entry.data)
    elif entry.url is not None:
        source_uri = entry.url
        scheme = get_url_scheme(source_uri)
        if scheme in ('http', 'https'):
            raw_data = await download(source_uri, sess, sem)
        else:
            raise ValueError('Invalid scheme: %s' % scheme)
    if raw_data is not None:
        image = Image.open(io.BytesIO(raw_data))
        image = image.convert('RGB')
        return image
    return None


async def retrieve_images(entries):
    """
    Retrieves the images for processing.

    :param entries: The entries to process.

    :return: The retrieved data.
    """
    tasks = list()
    sem = Semaphore(THREAD_COUNT)
    async with ClientSession() as sess:
        for entry in entries:
            tasks.append(
                wait_for(
                    retrieve_content(entry, sess, sem),
                    timeout=TIMEOUT,
                )
            )
        return await gather(*tasks)


async def download(url, sess, sem):
    """
    Downloads an image from the specified URL.

    :param url: The URL to download the image from.
    :param sess: The session to use to retrieve the data.
    :param sem: Used to limit concurrency.

    :return: The file's data.
    """
    async with sem, sess.get(url) as res:
        logger.info('Downloading %s' % url)
        content = await res.read()
        logger.info('Finished downloading %s' % url)
    if res.status != 200:
        raise ImageNotDownloadedException('Could not download image.')
    return content
```

### Image Prediction

The last two methods tie all of the above code together.

`predict_images` predicts the ImageNet categories for a list of `Pillow` formatted images and returns the results as a list of `ImageOutput` objects. 

The `process` function downloads images from a list of URLs, processes them and returns a response.

The `@app.post` annotation indicates that we want `FastAPI` to expose this function using HTTP and that the request and response structure is defined by `PredictRequest` and `PredictResponse` respectively. 

```python
def predict_images(images):
    """
    Predicts the image's category and transforms the results into the output format.

    :param images: The Pillow Images to predict.

    :return: The prediction results.
    """
    response = list()
    results = app.state.model.predict(images, BATCH_SIZE)
    for top_n in results:
        category, name, score = top_n[0]
        response.append(ImageOutput(category=category, name=name, score=score))
    return response


@app.post('/v1/predict', response_model=PredictResponse)
async def process(req: PredictRequest):
    """
    Predicts the category of the images contained in the request.

    :param req: The request object containing the image data to predict.

    :return: The prediction results.
    """
    logger.info('Processing request.')
    logger.debug(req.json())
    logger.info('Downloading images.')
    images = await retrieve_images(req.images)
    logger.info('Performing prediction.')
    results = predict_images(images)
    logger.info('Transaction complete.')
    return PredictResponse(images=results)
```

### Health Check

We will also expose a `HTTP GET` endpoint that can be called by load balancers to check the health of the application.

```python
@app.get('/health')
def test():
    """
    Can be called by load balancers as a health check.
    """
    return HealthCheck()
```

### Handler (Lambda)

In order to use `FastAPI` on Lambda we need to wrap the `FastAPI` `app` object with `Mangum`. 

```python
handler = Mangum(app)
```

*Note: When deploying the application locally or on AWS Elastic Beanstalk, this handler will not be used.*

### Server setup (Local development only)

Finally we will add an embedded `uvicorn` server that will allow us to run the API from the commandline for testing purposes.

```python
if __name__ == '__main__':

    import uvicorn

    parser = argparse.ArgumentParser(description='Runs the API locally.')
    parser.add_argument('--port',
                        help='The port to listen for requests on.',
                        type=int,
                        default=8080)
    args = parser.parse_args()
    configure_logging()
    uvicorn.run(app, host='0.0.0.0', port=args.port)
```

### Run the API

You can run the application locally by executing the command below:

```bash
python main.py
```

This runs the application on an embedded `uvicorn` server.

## Lambda Deployment

Deploying the API to Lambda has a number of benefits, such as:
- No management of servers or the underlying infrastructure.
- Easy to scale the number of API instances to handle fluctuations in traffic.

However it also has a few drawbacks, in particular:
- The first time the Lambda is called there will be a small delay due to the container starting up.
- Only memory (and proportional CPU amount) can be adjusted to improve performance.
- Timeout is limited to 29 seconds when using it with an API Gateway.
- Docker images are limited to 10 GB (but this should be more than enough for most APIs).
- Pricing is based on both request count and processing time, which can become expensive for large numbers of longer running processes.
- Not suitable for applications that require local storage; Lambda only provides 512MB of additional storage in the `/tmp` directory.

### Docker Image

AWS provides a number of base images for programming languages supported by Lambda. 

For this tutorial we are going to be using the Python 3.7 base image.

*Note that the name of the handler matches the name of the `Mangum` handler defined in the source code.* 

```bash
FROM public.ecr.aws/lambda/python:3.7

# Copy function code
COPY main.py ${LAMBDA_TASK_ROOT}/app.py

# Install the function's dependencies using file requirements.txt

COPY requirements.txt  .
RUN  pip3 install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]
```

Next we build and tag the Lambda image using the `docker build` command:

```bash
docker build -t imagenet-lambda -f Dockerfile.lambda .
```

After the command completes you should see output similar to below:

```bash
Step 5/5 : CMD [ "app.handler" ]
 ---> Running in 2257d01b05b0
Removing intermediate container 2257d01b05b0
 ---> 14755a1c4440
Successfully built 14755a1c4440
Successfully tagged imagenet-lambda:latest
```

Before we can use the image in Lambda we also need to upload it to [AWS Elastic Container Registry (ECR)](https://aws.amazon.com/ecr/).

First, navigate to the AWS ECR console and create a new repository. 

Next log into the private ECR repository you just created, by replacing the `<Region Name>` and `<Account ID>` placeholders with the region of the ECR repository and the account number of your AWS account, and executing the command below:

```bash
aws ecr get-login-password --region <Region Name> | docker login --username AWS --password-stdin <Account ID>.dkr.ecr.<Region Name>.amazonaws.com
```

Note: to run the command above, the `awscli` package needs to have been installed. Installation instructions can be found [here](https://github.com/aws/aws-cli).


Finally we will tag the image and push it (upload it) to ECR.

```bash
docker tag imagenet-lambda:latest <AccountID>.dkr.ecr.<Region>.amazonaws.com/<RepositoryName>:latest
docker push  <AccountID>.dkr.ecr.<Region>.amazonaws.com/<RepositoryName>:latest
```

### CloudFormation

Now that we have created the Docker image, the next step is to create the Lambda function that will use it. 

To deploy the function I have prepared a CloudFormation template that creates a Lambda, an API Gateway, and sets up the permissions required to call the function.

1. Go to: https://console.aws.amazon.com/cloudformation/home
2. Select the region that you want to deploy to.
3. Select 'Create Stack'->'With new resources (standard).'
4. Select 'Upload template'.
5. Enter the path to [deploy/lambda.yaml](deploy/lambda.yaml). Click next.
6. Enter the required parameters for the template, in particular:
   - Stack Name. The name of the stack. For example: 'ImageNetTest'. 
   - ImageUri. The URI of Docker image uploaded in the previous step; In the format `<Account ID>.dkr.ecr.<Region>.amazonaws.com/<Repository Name>:latest`.
7. Click next.
8. At the bottom of the screen, check all of the check boxes. 
9. Click 'Create Stack'. 

The API has now been deployed and can be invoked by calling it through the API Gateway.

## Elastic Beanstalk Deployment

Similar to Lambda, deploying the API to Elastic Beanstalk also has a number of benefits, such as:
- A lot more flexibility in configuration. For example, you can choose from a wide variety of instance types, including GPU instances.
- No warmup time.
- Cost is based on the time that an instance is running, rather than per a request. This maybe cheaper for some APIs.

However it also has a few drawbacks, in particular:
- Instance startup and scaling is slower.
- You need to pay for the instance even if it is idle.
- The increase in flexibility also makes it more complicated to use.

### Docker Image

Similar to Lambda we will create a `Dockerfile` to create a Docker image, but the contents are a little bit different.

Instead of using `Mangum` and Lambda to call the handler directly we will deploy the application using [uvicorn](https://www.uvicorn.org/) and [gunicorn](https://gunicorn.org/).

To simplify the creation of the template, and setup of the server, we will reuse the `uvicorn-gunicorn-docker` [template](https://github.com/tiangolo/uvicorn-gunicorn-docker).

To build the Elastic Beanstalk image execute the following command:

```bash
docker build -t imagenet-eb -f Dockerfile  .
```

Once the build is complete, you should see output similar to below:

```bash
Removing intermediate container 91a51bb166d4
 ---> 0bb1a7031a5e
Successfully built 0bb1a7031a5e
Successfully tagged imagenet-eb:latest
```

Before we can use the image in Elastic Beanstalk we also need to upload it to [AWS Elastic Container Registry (ECR)](https://aws.amazon.com/ecr/).

First, navigate to the AWS ECR console and create a new repository. 

Next, we will tag the image and push it to the ECR repository. Please replace the `<Region Name>`, `<Repository Name>` and `<Account ID>` placeholders with the region and name of the ECR repository and the account number of your AWS account:

```
docker tag imagenet-eb:latest <Account ID>.dkr.ecr.<Region>.amazonaws.com/<Repository Name>:latest
docker push  <Account ID>.dkr.ecr.<Region>.amazonaws.com/<Repository Name>:latest
```

### CloudFormation

Now that we have created the Docker image, the next step is to create the Elastic Beanstalk application to deploy it to. 

To create this I have prepared a CloudFormation template that creates and sets up an Elastic Beanstalk application and environment with the required permissions.

1. Go to: https://console.aws.amazon.com/cloudformation/home
2. Select the region that you want to deploy to.
3. Select 'Create Stack'->'With new resources (standard).'
4. Select 'Upload template'.
5. Enter the path to [deploy/elastic-beanstalk.yaml](deploy/elastic-beanstalk.yaml). Click next.
6. Enter the required parameters for the template, in particular:
   - Stack Name. The name of the stack. For example: 'ImageNetTestEb'. 
   - InstanceType. The instance type to instantiate. 
7. Click next.
8. At the bottom of the screen, check all of the check boxes. 
9. Click 'Create Stack'. 

### API Deployment

After CloudFormation has completed creating the Elastic Beanstalk application, you can deploy the API to it.

Elastic Beanstalk provides two methods to deploy your application: 
1. Upload a Zip file. You upload a zip file containing the API's files. A file named `Dockerrun.aws.json` must exist and define port mappings between the host system and Docker, but does not need to specify the Docker image URI. If a `Dockerfile` exists in the zip file the API container will be built locally without the need to upload it to ECR. 
2. Upload `Dockerrun.aws.json` only. This file describes the Docker image path and port mappings between the host system and Docker, but needs to specify the URI of the Docker image.   

For this example we will use the second approach.

Create a new file called `Dockerrun.aws.json` with the content below:

```json
{
  "AWSEBDockerrunVersion": "1",
  "Image": {
    "Name": "<Account ID>.dkr.ecr.<Region>.amazonaws.com/<Repository Name>:latest"
  },
  "Ports": [
    {
      "ContainerPort": "80"
    }
  ]
}
```

Replace the Image URI with with the URI of the image you uploaded previously. 

Go to the Elastic Beanstalk console and upload the `Dockerrun.aws.json` file using the 'Upload and deploy' button.

Deployment is now complete.

## Conclusion

In this tutorial I have described an approach to create an API for a TensorFlow model using FastAPI. 

We have also deployed the API onto two different AWS services using the same code. 

Although it can sometimes be difficult to choose which AWS service to use for machine learning deployment, I'd recommend that you try AWS Lambda first because of its simplicity.

 If you require a GPU for faster inference, or you have other requirements that make AWS Lambda unsuitable, then I would recommend AWS Elastic Beanstalk instead.

 ## Files

- [main.py](main.py) - API Source code
- [Dockerfile](Dockerfile) - Dockerfile for Elastic Beanstalk deployment.
- [Dockerfile.lambda](Dockerfile.lambda) - Dockerfile for Lambda deployment.
- [Dockerrun.aws.json](Dockerrun.aws.json) - Elastic Beanstalk container configuration file.
- [requirements.txt](requirements.txt) - Lists the requirements for the API.
- [deploy/elastic-beanstalk.yaml](deploy/elastic-beanstalk.yaml) - Elastic Beanstalk deployment CloudFormation template.
- [deploy/lambda.yaml](deploy/lambda.yaml) - Lambda deployment Cloudformation template.