import numpy as np
import requests

from PIL import Image
from io import BytesIO
from imageai.Detection import ObjectDetection


def fetch_image(url) -> Image:
    return Image.open(BytesIO(requests.get(url).content))


class ObjectOnlyCompressor:

    def __init__(self, model_path, api_key):
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(model_path)
        self.detector.loadModel()
        self.api_key = api_key

    def compress(self, input_path, output_path):
        image = Image.open(input_path).convert('RGB')
        image.thumbnail((255, 255))
        items = self.detector.detectObjectsFromImage(
            input_image=np.asarray(image),
            input_type='array',
            output_image_path='debug.jpg',
        )

        file_content = bytearray()
        for item in items:
            file_content.extend(item['name'].encode('ascii'))
            file_content.append(0)
            file_content.extend(item['box_points'])

        with open(output_path, 'wb') as file:
            file.write(file_content)

    def decompress(self, input_path, output_path):
        with open(input_path, 'rb') as input_file:
            input_file_contents = input_file.read()

        output_image = Image.new('RGB', (255, 255), (255, 255, 255))

        items = []
        while input_file_contents:
            delimiter_index = input_file_contents.index(0)

            name = input_file_contents[:delimiter_index].decode('ascii')
            box_points = [int(input_file_contents[delimiter_index + offset]) for offset in range(1, 5)]

            items.append((name, box_points))

            input_file_contents = input_file_contents[delimiter_index+5:]

        for item in items:
            name, box_points = item

            print(name, box_points)

            response = requests.post(
                "https://api.deepai.org/api/text2img",
                data={'text': name},
                headers={'api-key': self.api_key}
            )
            url = response.json()['output_url']

            image = fetch_image(url)
            image = image.resize((
                box_points[2] - box_points[0],
                box_points[3] - box_points[1],
            ))

            output_image.paste(image, (box_points[0], box_points[1]))

        output_image.save(output_path)
