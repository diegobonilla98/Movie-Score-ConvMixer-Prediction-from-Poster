import cv2
import matplotlib.pyplot as plt
import pandas as pd
import python_filmaffinity
import tqdm
from PIL import Image
import requests
from io import BytesIO
import utils
import uuid
import csv
import numpy as np
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('movies.csv')
service = python_filmaffinity.FilmAffinity(lang='en')
f = open('/media/bonilla/My Book/MoviePosterScore/movies_info.csv', 'w')
writer = csv.writer(f)
header = ['name', 'score', 'image_id', 'year']
writer.writerow(header)

with tqdm.tqdm(total=len(data['title'])) as pbar:
    for movie_name in data['title']:
        try:
            year = int(movie_name[-5:-1])
            title = movie_name[:-7]
            movie = service.search(top=1, title=title, from_year=year - 1, to_yeat=year + 1)[0]
            poster_url = utils.get_movie_poster(title)
            response = requests.get(poster_url)
            img = np.array(Image.open(BytesIO(response.content)))
            rating = float(movie['rating'])
            image_name = str(uuid.uuid4())
            image_path = '/media/bonilla/My Book/MoviePosterScore/movie_posters/' + image_name + '.jpg'
            cv2.imwrite(image_path, img[:, :, ::-1])
            image_path = image_name
            writer.writerow([title, rating, image_path, year])
            f.flush()
        except Exception as e:
            print(e)
            pass
        finally:
            pbar.update()

f.close()
