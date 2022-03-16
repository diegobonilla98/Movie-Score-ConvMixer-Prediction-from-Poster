import pandas as pd
import python_filmaffinity
from PIL import Image
import requests
from io import BytesIO
import utils
import random
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def nothing(x):
    pass


def new_movie():
    errors = True
    m = i = s = None
    while errors:
        try:
            errors = False
            title_org = random.choice(data['title'])
            year = int(title_org[-5:-1])
            title = title_org[:-7]
            m = service.search(top=1, title=title, from_year=year - 1, to_yeat=year + 1)[0]
            poster_url = utils.get_movie_poster(title)
            response = requests.get(poster_url)
            i = np.array(Image.open(BytesIO(response.content)))
            s = int(float(m['rating']) * 10)
        except Exception:
            errors = True
    return m, i, s


cv2.namedWindow('Output')
cv2.createTrackbar('r', 'Output', 0, 100, nothing)

data = pd.read_csv('movies.csv')
service = python_filmaffinity.FilmAffinity(lang='en')

best_score = np.load('best_score.npy')
movie, img, real_rating = new_movie()
total_score = 0
best_of = 10
movie_counter = 1
print(f"Movie {movie_counter}/{best_of}")

while True:
    cv2.imshow('Output', img[:, :, ::-1])
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord(' '):
        diff = abs(int(cv2.getTrackbarPos('r', 'Output')) / 10. - real_rating / 10.)
        if 5 > diff > 3:
            total_score += 0
        elif diff > 5:
            total_score -= 10
        elif diff == 0:
            total_score += 75
        else:
            total_score += int(1. / diff * 5)
        print(f"You put {int(cv2.getTrackbarPos('r', 'Output')) / 10.} and the real score is {real_rating / 10.}\t[Your score is: {total_score}]\n")
        movie, img, real_rating = new_movie()
        movie_counter += 1
        if movie_counter == best_of + 1:
            print("Game finished")
            break
        print(f"Movie {movie_counter}/{best_of}")

cv2.destroyAllWindows()
if best_of not in best_score[:, 0]:
    np.save('best_score.npy', np.concatenate([best_score, np.array([[best_of, total_score]])], axis=0))
    print("NEW RECORD!")
else:
    for idx in range(best_score.shape[0]):
        if best_score[idx][0] == best_of:
            if best_score[idx][1] < total_score:
                print("NEW RECORD!")
                best_score[idx] = total_score
                break
    np.save('best_score.npy', best_score)
print(f"Your score is {total_score} after {best_of} rounds.")
