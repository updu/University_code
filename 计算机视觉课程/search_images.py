#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import errno
import requests
from PIL import Image
from io import BytesIO
import os
import grequests


def exception_handler(request, exception):
    print("--Error--\n")  # + request.url)



query_string = '林心如'
request_url = 'http://huaban.com/search/'

headers = {
    "Host": 'huaban.com',
    "Referer": 'http://huaban.com/favorite/beauty/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    "Accept": "application/json",
    "X-Request": "JSON",
    "X-Requested-With": "XMLHttpRequest",
}

params = {
    "max": "1072369228",
    "limit": "20",
    "wfl": "1",
}

requests_params = {
    'q': query_string,
    'type': 'pins',
    'page': 1,
    'per_page': 20,
    'wfl': 1,
}

session = requests.session()
session.headers.update(headers)


# send request to get more pinid which is in json format
def more_pin_ids(last_pin):
    params["max"] = last_pin
    resp = session.get(request_url, params=params, headers=headers)
    data = resp.json()["pins"]
    pin_ids = [x["pin_id"] for x in data]
    return pin_ids


def json_to_pic_urls(json):
    pic_keys = [x['file']['key'] for x in json['pins']]
    pic_urls = ['http://img.hb.aicdn.com/' + x  for x in pic_keys]
    return pic_urls


def more_pics():
    resp = session.get(request_url, params=requests_params, headers=headers)
    requests_params['page'] += 1
    json = resp.json()
    return json_to_pic_urls(json)


# 创建一个路径
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def download(the_urls):
    n = 0
    while len(the_urls) > 0:
        urls = [the_urls.pop() for i in range(50) if len(the_urls) > 0 ]
        rs = (grequests.get(u, timeout=20) for u in urls)
        imgs = grequests.map(rs, exception_handler=exception_handler)
        for img in imgs:
            if not img:
                continue
            n = n + 1
            im = Image.open(BytesIO(img.content))
            mkdir_p('jpg')
            im.save("jpg\\" + str(n) + ".png")
            print('complete {} image'.format(n))



if __name__ == '__main__':
    urls_to_download = set()
    while len(urls_to_download) < 1000:
        urls_to_download.update(more_pics())
    download(urls_to_download)
