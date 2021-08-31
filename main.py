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


THREAD_COUNT = int(os.environ.get('THREAD_COUNT', 5))
"""The number of threads used to download and process image content."""

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
"""The number of images to process in a batch."""

TIMEOUT = int(os.environ.get('TIMEOUT', 30))
"""The timeout to use when downloading files."""


logger = logging.getLogger(__name__)


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


app = FastAPI()


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


@app.on_event('startup')
def load_model():
    """
    Loads the model prior to the first request.
    """
    configure_logging()
    logger.info('Loading models...')
    app.state.model = ImageClassifier()


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


@app.get('/health')
def test():
    """
    Can be called by load balancers as a health check.
    """
    return HealthCheck()


handler = Mangum(app)

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
