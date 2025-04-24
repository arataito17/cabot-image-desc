#!/usr/bin/env python3

# Copyright (c) 2024  Carnegie Mellon University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import base64
import os
import argparse
import datetime
from io import BytesIO

from openai import OpenAI
from pymongo import MongoClient
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from PIL.TiffImagePlugin import IFDRational
import hashlib
import sys
import json
from bson import ObjectId

IMAGE_DESCRIPTION_PROMPT = """
# 概要
- 画像に写っているものをリストで表示。リストのみ
- フォーマットは以下の通りで```は含まない
```
- ＜名称＞:<詳細>
```

# リストに含めないもの
- 画像に写っているものなどの説明
- 天気に関するもの、表現
- 人
- 乗り物
"""
IMAGE_DESCRIPTION_PROMPT_1 = """
# 概要
- 画像に写っているものをリストで表示。リストのみ
- それぞれの写っているものについて視覚的な特徴を記述
- フォーマットは以下の通りで```は含まない
```
- ＜名称＞：＜詳細＞
```

# リストに含めないもの
- 画像に写っているものなどの説明
- 天気に関するもの、表現
- 人
- 乗り物
"""
if os.getenv('OPENAI_API_KEY'):
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def transcribe_image_query(filepath):
    def resize_with_aspect_ratio(image, target_size):
        original_width, original_height = image.size
        if original_width > original_height:
            new_width = target_size
            new_height = int((target_size / original_width) * original_height)
        else:
            new_height = target_size
            new_width = int((target_size / original_height) * original_width)
        return image.resize((new_width, new_height))

    def parse_exif(exif_data):
        def convert_to_degrees(value):
            """Helper function to convert GPS coordinates to degrees."""
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
        gps_info = {}
        exif = {}

        def normalize(value):
            if isinstance(value, IFDRational):
                float(value)
            elif isinstance(value, (int, str)):
                return value
            elif isinstance(value, bytes):
                # convert bytes to hex string
                return f"0x{value.hex()}"
            elif isinstance(value, tuple):
                return [float(e) for e in value]
            else:
                print([tag_name, type(value), value], file=sys.stderr)
                return None

        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'GPSInfo':
                exif[tag_name] = {}
                gps_latitude = None
                gps_longitude = None
                gps_direction = None
                for gps_tag in value:
                    sub_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    exif[tag_name][sub_tag_name] = normalize(value[gps_tag])
                    if sub_tag_name == 'GPSLatitude':
                        gps_latitude = convert_to_degrees(value[gps_tag])
                    elif sub_tag_name == 'GPSLatitudeRef':
                        if value[gps_tag] != 'N' and gps_latitude is not None:
                            gps_latitude = -gps_latitude
                    elif sub_tag_name == 'GPSLongitude':
                        gps_longitude = convert_to_degrees(value[gps_tag])
                    elif sub_tag_name == 'GPSLongitudeRef':
                        if value[gps_tag] != 'E' and gps_longitude is not None:
                            gps_longitude = -gps_longitude
                    elif sub_tag_name == 'GPSImgDirection':
                        gps_direction = value[gps_tag]
                if gps_latitude is not None and gps_longitude is not None:
                    gps_info['Latitude'] = gps_latitude
                    gps_info['Longitude'] = gps_longitude
                if gps_direction is not None:
                    gps_info['Direction'] = gps_direction
            else:
                if normalize(value):
                    exif[tag_name] = normalize(value)
        return gps_info, exif

    def get_md5_hash(filepath):
        # Create an MD5 hash object
        md5_hash = hashlib.md5()

        # Open the file in binary mode
        with open(filepath, 'rb') as file:
            # Read the file in chunks to avoid memory issues with large files
            for chunk in iter(lambda: file.read(4096), b''):
                md5_hash.update(chunk)

        # Return the hexadecimal digest of the MD5 hash
        return md5_hash.hexdigest()

    def linux_time_from_exif(exif):
        # TODO (consider time zone)
        datestr, timestr = str(exif['DateTime']).split(" ")
        items = datestr.split(":")
        items.extend(timestr.split(":"))
        items = [int(v) for v in items]
        return datetime.datetime(*items).timestamp()

    image = Image.open(filepath)
    image_hash = get_md5_hash(filepath)
    # Extract EXIF data
    gps_info, exif = parse_exif(image._getexif())
    image = resize_with_aspect_ratio(image, 512)
    with BytesIO() as buffer:
        image.convert('RGB').save(buffer, format='JPEG')
        jpeg_image_bytes = buffer.getvalue()
    jpeg_filepath = f'{os.path.splitext(filepath)[0]}_shrunk.jpeg'
    with open(jpeg_filepath, 'wb') as f:
        f.write(jpeg_image_bytes)
    base64_image = base64.b64encode(jpeg_image_bytes).decode('utf-8')
    image_uri = f'data:image/jpeg;base64,{base64_image}'
    filename = os.path.basename(filepath)

    return ({
        'model': 'gpt-4o-2024-08-06',
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': IMAGE_DESCRIPTION_PROMPT
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': image_uri
                        },
                    },
                ],
            }
        ]}, {
            'filename': filename,
            'image_hash': image_hash,
            'location': {
                'type': 'Point',
                'coordinates': [gps_info['Longitude'], gps_info['Latitude']]
            },
            'direction': float(gps_info['Direction']),
            'image': image_uri,
            'linuxtime': linux_time_from_exif(exif),
            'exif': exif
    })


def post_query(query):
    return openai_client.chat.completions.create(**query)


def get_result(response, metadata):
    description = response.choices[0].message.content

    metadata.update({
        'description': description
    })
    return metadata


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)


def pretty_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False, cls=JSONEncoder))


def pretty_print(data, depth=1, exclude=None, prefix=''):
    if isinstance(data, dict):
        for key in data:
            if exclude and key in exclude:
                continue
            print(F"{'  ' * depth}{prefix}{key}:")
            pretty_print(data[key], depth=depth + 1)
    elif isinstance(data, list):
        for item in data:
            pretty_print(item, depth=depth, prefix='- ')
    elif isinstance(data, str):
        # print escaped str
        escaped = data.replace('\n', '\\n').replace('\r', '\\r')
        # truncate long strings at 200 characters
        if len(escaped) > 100:
            escaped = escaped[:100] + '...'
        print(F"{'  ' * depth}{prefix}{escaped}")
    else:
        print(F"{'  ' * depth}{prefix}{data}")


# Configure MongoDB connection
mongodb_host = os.getenv('MONGODB_HOST', 'mongodb://mongo:27017/')
mongodb_name = os.getenv('MONGODB_NAME', 'geo_image_db')
client = MongoClient(mongodb_host)
db = client[mongodb_name]
collection = db['images']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an image file to transcribe its content.')
    parser.add_argument('-f', '--file', help='Path to the image file')
    parser.add_argument('-p', '--prompt', help='Prompt to use for the image description')
    parser.add_argument('-F', '--floor', type=int, help='Floor number of the building')
    parser.add_argument('-e', '--exif', action='store_true', help='Check the exif data of the image')
    parser.add_argument('-t', '--tag', action='append', help='Tags to associate with the image (can be used multiple times)')
    parser.add_argument('-T', '--removetag', action='append', help='Tags to remove from the image')
    parser.add_argument('-c', '--clear-tags', action='store_true', help='Clear all tags from the image')
    parser.add_argument('-n', '--dryrun', action='store_true', help='Perform a dry run without modifying the database')
    parser.add_argument('-r', '--retry', action='store_true', help='Descrtibe image even if the entry has already description')
    parser.add_argument('-l', '--list', action='store_true', help='List all images in the database')
    parser.add_argument('-j', '--json', action='store_true', help='Output the json data of the file')
    parser.add_argument('-J', '--jsonid', type=str, help='Output the json data of the ID')
    parser.add_argument('-R', '--remove', type=str, help='Remove the entry with the given ID')
    args = parser.parse_args()

    print(args)

    if args.list:
        for entry in collection.find():
            print(F"ID: {entry['_id']} Filename: {entry['filename']}")
        sys.exit(0)

    if args.jsonid:
        entry = collection.find_one({'_id': ObjectId(args.jsonid)})
        if entry:
            pretty_json(entry)
        else:
            print(f"No entry found with ID: {args.jsonid}")
        sys.exit(0)
    if args.remove:
        result = collection.delete_one({'_id': ObjectId(args.remove)})
        print(result)
        sys.exit(0)

    filepath = args.file
    query, metadata = transcribe_image_query(filepath)
    if args.floor is not None:
        metadata['floor'] = args.floor

    if args.exif:
        pretty_print(metadata['exif'])
        sys.exit(0)

    if args.prompt:
        with open(args.prompt, 'r') as f:
            IMAGE_DESCRIPTION_PROMPT = f.readlines()
            print("Custom prompt is used:")
            print(IMAGE_DESCRIPTION_PROMPT)

    existing_entry = collection.find_one({
        'filename': metadata['filename'],
        'image_hash': metadata['image_hash']
    })

    if args.json:
        pretty_json(existing_entry)
        sys.exit(0)

    if existing_entry:
        update = {}
        for key in metadata:
            if key not in existing_entry or existing_entry[key] != metadata[key]:
                update[key] = metadata[key]

        if args.clear_tags:
            update["tags"] = []
        else:
            if args.tag:
                update["tags"] = list(set(args.tag + existing_entry["tags"]))
            if args.removetag:
                update["tags"] = list(set(existing_entry["tags"]) - set(args.removetag))

        if args.dryrun:
            print(F"Dry run: The image is already transcribed ({existing_entry['_id']}):")
            pretty_print(existing_entry['description'])
            if update:
                print("The entry will be updated with:")
                pretty_print(update, depth=1)
            if args.retry:
                print("Will retry description with:")
                pretty_print(query)
        else:
            if args.retry:
                response = post_query(query)
                print("Received data:")
                pretty_print(response)
                data = get_result(response, metadata)
                for key in data:
                    if key not in existing_entry or existing_entry[key] != data[key]:
                        update[key] = data[key]

            print(F"The image is already transcribed ({existing_entry['_id']}):")
            pretty_print(existing_entry['description'])
            if update:
                collection.update_one(
                    {'_id': existing_entry['_id']},
                    {'$set': update}
                )
                print("The entry has been updated with:")
                pretty_print(update, depth=1)
    else:
        if args.dryrun:
            print(f"Dry run: Would post query and insert data for image: {filepath}")
            pretty_print(query)
        else:
            print("Posting query")
            response = post_query(query)
            print("Received data:")
            pretty_print(response)
            data = get_result(response, metadata)
            if args.tag:
                data['tags'] = args.tag
            if 'floor' not in data:
                data['floor'] = 0
            print("Insert data into the DB")
            collection.insert_one(data)
            print("Description:")
            pretty_print(data, exclude=['image', 'exif'])