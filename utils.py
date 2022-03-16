import imdb
import re

ia = imdb.IMDb()


def get_movie_poster(title, factor=16):
    movie = ia.search_movie(title)[0]
    cover = movie.get('cover url')
    s = re.search("V1_U[a-zA-Z][0-9][0-9]_.*,[0-9][0-9],[0-9][0-9]_AL_", cover)
    first_num_str = re.match("V1_U[a-zA-Z][0-9][0-9]_", s[0])
    first_num = int(first_num_str[0][5:-1])
    second_num = s[0].split(",")
    third_num = int(second_num[-1].split("_")[0])
    second_num = int(second_num[2])
    new_s = s[0].replace(str(first_num), str(first_num * factor))
    new_s = new_s.replace(str(second_num), str(second_num * factor))
    new_s = new_s.replace(str(third_num), str(third_num * factor))
    cover = cover.replace(s[0], new_s)
    return cover


def create_movie_dict(title):
    movie = ia.search_movie(title)[0]

    movieID = movie.movieID
    movie = ia.get_movie(movieID)

    score_met = ia.get_movie_critic_reviews(movieID)['data']
    if score_met:
        score_met = score_met.get('metascore')
        if score_met is None:
            return None
    else:
        return None

    title = movie.get('title')
    year = movie.get('year')
    genres = movie.get('genre')
    plot = movie.get('plot')
    if plot:
        plot = plot[0]
    cast = movie.get('cast')
    cover = movie.get('cover url')
    director = movie.get('director')
    url = ia.get_imdbURL(movie)

    return {'title': title, 'year': year, 'director': director, 'score': score_met, 'genres': genres,
            'cast': cast, 'cover': cover, 'plot': plot, 'url': url}
