from google_images_search import GoogleImagesSearch
import os
import curses  # Теперь будет работать

gis = GoogleImagesSearch(
    'AIzaSyB9hYDV5bBRwrsFSr3YFC3zjGLnLWfLnbI',
    '051ece63b505a4d69'
    )
_search_params = {
    'q': 'street, people, city street',  # что искать
    'num': 10,                   # количество фото
    'fileType': 'jpg|png',
}

gis.search(search_params=_search_params, path_to_dir='/home/user/diploma/share/images')
