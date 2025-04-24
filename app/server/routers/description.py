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
import datetime
import json
import logging
import math
import os
import time
from fastapi import APIRouter, Depends, Query, Request, HTTPException
from typing import Optional
from transformers import AutoModelForCausalLM, AutoProcessor

# Import required functions/classes from openai_agent and auth
from ..openai.openai_agent import GPTAgent
from ..openai.openai_agent import TranslatedDescription, construct_prompt_for_image_description
from ..openai.openai_agent import StopReason, construct_prompt_for_stop_reason
from .auth import verify_api_key_or_cookie
from ..db import get_description_by_lat_lng

router = APIRouter()
gpt_agent = GPTAgent()

logger = logging.getLogger(__name__)


def getOrientation(rotation, direction):
    direction = -direction * math.pi / 180
    diff = direction - rotation
    while diff < -math.pi:
        diff += math.pi * 2
    while diff > math.pi:
        diff -= math.pi * 2
    return diff


def get_relative_coordinates(lat1, lng1, lat2, lng2, rotation):
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)
    R = 6371000
    d_lat = lat2_rad - lat1_rad
    d_lng = lng2_rad - lng1_rad
    x = R * d_lng * math.cos((lat1_rad + lat2_rad) / 2)
    y = R * d_lat
    relative_x = x * math.cos(rotation) + y * math.sin(rotation)
    relative_y = -x * math.sin(rotation) + y * math.cos(rotation)
    return relative_x, relative_y


def classify_direction(radian):
    front_threshold = 30
    degree = radian * 180 / math.pi
    if abs(degree) < front_threshold:
        return "front"
    elif abs(degree) > (180 - front_threshold):
        return "back"
    elif degree > 0:
        return "left"
    else:
        return "right"


def preprocess_descriptions(locations, rotation, lat, lng, max_distance):
    dummy_object = {"distance": 9999, "description": ""}
    tags_to_use = {"sign", "poi", "highpriority"}
    location_per_directions = {"front": dummy_object.copy(), "left": dummy_object.copy(), "right": dummy_object.copy()}
    for location in locations:
        tag = location.get("tags", [])
        if len(set(tag).intersection(tags_to_use)) == 0:
            continue
        if "sign" in tag:
            location["description"] = "これはこの方向にある看板に関する追加の説明文章です。" + location["description"]
        elif "highpriority" in tag:
            location["description"] = "【重要！】" + location["description"]
        elif "poi" in tag:
            location["description"] = "これはこの方向にある施設・設備に関する追加の説明文章です。" + location["description"]
        direction = location['direction']
        location['relative_direction'] = getOrientation(rotation, direction)
        loc_lat = location['location']['coordinates'][1]
        loc_lng = location['location']['coordinates'][0]
        relative_x, relative_y = get_relative_coordinates(lat, lng, loc_lat, loc_lng, rotation)
        location['relative_coordinates'] = {'x': relative_x, 'y': relative_y}
        location['distance'] = math.sqrt(relative_x ** 2 + relative_y ** 2)
        direction = classify_direction(location["relative_direction"])
        distance = math.sqrt(relative_x ** 2 + relative_y ** 2)
        if direction == "back":
            continue
        if direction == "front":
            location["description"] = "前：" + location["description"]
        elif direction == "left":
            location["description"] = "左：" + location["description"]
        elif direction == "right":
            location["description"] = "右：" + location["description"]
        if distance < location_per_directions[direction]["distance"]:
            location_per_directions[direction] = location
    past_explanations = ""
    for past_description in gpt_agent.past_descriptions.copy():
        rel_x, rel_y = get_relative_coordinates(lat, lng, past_description["location"]["lat"], past_description["location"]["lng"], rotation)
        if math.sqrt(rel_x ** 2 + rel_y ** 2) < max_distance:
            past_explanations += past_description["description"] + "\n"
        else:
            gpt_agent.past_descriptions.remove(past_description)
    return location_per_directions, past_explanations


def parsed_value(result, key):
    try:
        return getattr(result.choices[0].message.parsed, key)
    except Exception as e:
        if not hasattr(result, "error"):
            setattr(result, "error", str(e))
            result.obj["error"] = str(e)
        return f"Error: No {key}"


class BaseMLLM:
        def __init__(self):
            pass
            
        def __call__(self):
            print("NO MODEL\n")
            print("please select a model")
            return None
            
        def sara2_woi(self, lat, lng, floor, rotation, max_distance, max_count, sentence_length, lang):
            st = time.time()        
            self.model_path = "sbintuitions/sarashina2-vision-8b"
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            #####ここから
            self.locations = get_description_by_lat_lng(lat, lng, floor, max_distance, max_count=5)

            location_per_directions, past_explanations = preprocess_descriptions(self.locations, rotation, lat, lng, max_distance)
            #####ここまでは全部のモデルで共通してるからclassの外に出してもいいかも
            self.prompt = construct_prompt_for_image_description(sentence_length=sentence_length,
                                                front=location_per_directions["front"]["description"],
                                                right=location_per_directions["right"]["description"],
                                                left=location_per_directions["left"]["description"],
                                                past_explanations=past_explanations,
                                                lang=lang,
                                                )
            message = [{"role": "user", "content": self.prompt}]
            text_prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = self.processor(
            text=[text_prompt],
            padding=True,
            return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            stopping_criteria = self.processor.get_stopping_criteria(["\n###"])
            
            # Inference: Generation of the output
            output_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.6,
            do_sample=True,
            stopping_criteria=stopping_criteria,
            )
            generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            elapsed_time = time.time() - st

            logger.info("Time taken: %s", elapsed_time)
            logger.info("Generated description: %s", output_text[0])
            return output_text[0], self.prompt, elapsed_time, self.locations, self.model_path        
        ###
        #def sara2(self, lat, lng, floor, rotation, max_distance, max_count, sentence_length, tags, lang):
                

        
             

        async def gpt4o(self, lat, lng, floor,rotation, max_distance, max_count, sentence_length, tags, lang):
            ####ここから
            logger.info("description get")
            locations = get_description_by_lat_lng(lat, lng, floor, max_distance, max_count)

            location_per_directions, past_explanations = preprocess_descriptions(locations, rotation, lat, lng, max_distance)
            ####ここまでsarashina2-vision-8bと共通してるからclassの外に出してもいいかも
            prompt = construct_prompt_for_image_description(sentence_length=sentence_length,
                                                            front=location_per_directions["front"]["description"],
                                                            right=location_per_directions["right"]["description"],
                                                            left=location_per_directions["left"]["description"],
                                                            past_explanations=past_explanations,
                                                            lang=lang,
                                                            )

            st = time.time()
            (original_result, query) = await gpt_agent.query_with_images(prompt=prompt, response_format=TranslatedDescription)
            elapsed_time = time.time() - st
            description = parsed_value(original_result, "description")
            translated = parsed_value(original_result, "translated")
            lang = parsed_value(original_result, "lang")
            # log
            date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
            if hasattr(original_result, "error"):
                raise HTTPException(status_code=400, detail=original_result.error)
            
            return description, translated, lang, query, date, locations, self.model_path, self.processor, self.model, elapsed_time, lang






@router.get('/description', dependencies=[Depends(verify_api_key_or_cookie)])
def read_description_by_lat_lng(lat: float = Query(...),
                               lng: float = Query(...),
                               floor: int = Query(0),
                               rotation: float = Query(...),
                               max_count: Optional[int] = Query(10),
                               max_distance: Optional[float] = Query(100),
                               lang: Optional[str] = Query("ja"),
                               sentence_length: Optional[int] = Query(3),
                               ):
    logger.info(f"Handling description request: lat={lat}, lng={lng}, floor={floor}, rotation={rotation}")
    base_mllm = BaseMLLM()

    # モデル呼び出し
    output_text, prompt, elapsed_time, locations, model_path = base_mllm.sara2_woi(
        lat, lng, floor, rotation, max_distance, max_count, sentence_length, lang=lang
    )
    
    logger.info(f"Model response: elapsed_time={elapsed_time:.2f}s, model={model_path}")
    
    # 出力の検証
    if not output_text or output_text.startswith("エラー"):
        logger.error(f"Model error: {output_text}")
        output_text = "説明文の生成中にエラーが発生しました。"
    else:
        logger.info(f"Generated description: {output_text}")
    
    # ログ記録（変更なし）
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    log_json(directory=date, name="params", data={
        "lat": lat,
        "lng": lng,
        "rotation": rotation,
        "max_count": max_count,
        "max_distance": max_distance,
        "sentence_length": sentence_length,
        "prompt": prompt,
        "lang": lang,
        "time_taken": elapsed_time,
        "model_path": model_path,
    })
    
    # モデル出力をログに記録
    log_json(directory=date, name="locations", data=locations)
    log_text(directory=date, name="model-output", data=output_text)
    log_text(directory=date, name="prompt", data=prompt)
    
    # フロントエンド用のレスポンス形式を確認
    return {
        'locations': locations,
        'elapsed_time': elapsed_time,
        'description': output_text,  # output_text[0] ではなく output_text に
        'lang': lang,
        'model': model_path,
    }

# TODO: upload a live image and describe the image, using nearby data
@router.post('/description_with_live_image', dependencies=[Depends(verify_api_key_or_cookie)])
async def read_description_by_lat_lng_with_image(request: Request,
                                                 lat: float = Query(...),
                                                 lng: float = Query(...),
                                                 floor: int = Query(0),
                                                 rotation: float = Query(...),
                                                 max_count: Optional[int] = Query(10),
                                                 max_distance: Optional[float] = Query(100),
                                                 lang: Optional[str] = Query("ja"),
                                                 sentence_length: Optional[int] = Query(3),
                                                 use_live_image_only: Optional[bool] = Query(False),
                                                 ):
    logger.info("description_with_live_image post")
    locations = []
    if not use_live_image_only:
        logger.info("you won't use live image only")
        locations = get_description_by_lat_lng(lat, lng, floor, max_distance, max_count)
        if not locations:
            raise HTTPException(status_code=400, detail="No locations found")
        logger.info("locations: %s", locations)
    else:
        logger.info("using live image only")

    images = await request.json()

    location_per_directions, past_explanations = preprocess_descriptions(locations, rotation, lat, lng, max_distance)

    count = 1
    tags = ""
    for image in images:
        position = image['position']
        tags += f"{count}枚目: {position}\n"
        count += 1

    prompt = construct_prompt_for_image_description(sentence_length=sentence_length,
                                                    front=location_per_directions["front"]["description"],
                                                    right=location_per_directions["right"]["description"],
                                                    left=location_per_directions["left"]["description"],
                                                    past_explanations=past_explanations,
                                                    image_tags=tags,
                                                    lang=lang,
                                                    )

    st = time.time()
    (original_result, query) = await gpt_agent.query_with_images(prompt=prompt, images=images, response_format=TranslatedDescription)
    elapsed_time = time.time() - st
    description = parsed_value(original_result, "description")
    translated = parsed_value(original_result, "translated")
    lang = parsed_value(original_result, "lang")
    logger.info("Time taken: %s", elapsed_time)
    logger.info("Generated description: %s", description)
    logger.info("Translated description: %s", translated)
    logger.info("Language: %s", lang)

    # log
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    log_json(directory=date, name="images", data=images)
    log_json(directory=date, name="params", data={
        "lat": lat,
        "lng": lng,
        "rotation": rotation,
        "max_count": max_count,
        "max_distance": max_distance,
        "sentence_length": sentence_length,
        "use_live_image_only": use_live_image_only,
        "prompt": prompt,
        "lang": lang,
    })
    log_json(directory=date, name="openai-query", data=query)
    log_json(directory=date, name="openai-prompt", data=prompt)
    log_json(directory=date, name="locations", data=locations)
    log_json(directory=date, name="openai-response", data=json.loads(original_result.model_dump_json()))

    if hasattr(original_result, "error"):
        raise HTTPException(status_code=400, detail=original_result.error)

    return {
        'locations': locations,
        'elapsed_time': elapsed_time,
        'description': description,
        'lang': lang,
        'translated': translated,
    }


@router.post("/stop_reason", dependencies=[Depends(verify_api_key_or_cookie)])
async def stop_reason(request: Request,
                      lang: Optional[str] = Query("ja"),
                      ):
    logger.info("stop_reason post")

    images = await request.json()

    temp = []
    for image in images:
        position = image["position"]
        if "front" != position:
            continue
        temp.append(image)

    prompt = construct_prompt_for_stop_reason(lang=lang)

    st = time.time()
    (original_result, query) = await gpt_agent.query_with_images(prompt=prompt, images=temp, response_format=StopReason)
    elapsed_time = time.time() - st
    description = parsed_value(original_result, "message")
    translated = parsed_value(original_result, "translated")
    lang = parsed_value(original_result, "lang")
    #queryを表示
    logger.info("Query: %s", query)
    logger.info("Time taken: %s", elapsed_time)
    logger.info("Generated description: %s", description)
    logger.info("Translated description: %s", translated)
    logger.info("Language: %s", lang)

    # log
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    log_json(directory=date, name="images", data=temp)
    log_json(
        directory=date,
        name="params",
        data={
            "mode": "stop-reason",
            "prompt": prompt,
            "lang": lang,
        },
    )
    log_json(directory=date, name="openai-query", data=query)
    log_text(directory=date, name="openai-prompt", data=prompt)
    log_json(
        directory=date, name="openai-response", data=json.loads(original_result.model_dump_json())
    )
    log_image(directory=date, position="front", images=temp)

    if hasattr(original_result, "error"):
        raise HTTPException(status_code=400, detail=original_result.error)

    return {
        "elapsed_time": elapsed_time,
        "description": description,
        "translated": translated,
        "lang": lang,
    }


def log_json(directory, name, data):
    basepath = f"/logs/{directory}"
    os.makedirs(basepath, exist_ok=True)
    with open(f"{basepath}/{name}.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_text(directory, name, data):
    basepath = f"/logs/{directory}"
    os.makedirs(basepath, exist_ok=True)
    with open(f"{basepath}/{name}.txt", "w") as f:
        print(data, file=f)


def log_image(directory, position, images):
    image = list(filter(lambda x: x["position"] == position, images))
    if image:
        image_uri = image[0]["image_uri"]

    basepath = f"/logs/{directory}"
    os.makedirs(basepath, exist_ok=True)
    with open(f"{basepath}/{position}.jpg", "wb") as f:
        f.write(base64.b64decode(image_uri.split(",")[1]))
