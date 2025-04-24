import base64
import datetime
import json
import logging
import math
import os
import time
from fastapi import APIRouter, Depends, Query, Request, HTTPException
from typing import Optional
import requests

# Import required functions/classes from openai_agent and auth
from ..openai.openai_agent import GPTAgent
from ..openai.openai_agent import TranslatedDescription, construct_prompt_for_image_description
from ..openai.openai_agent import StopReason, construct_prompt_for_stop_reason
from .auth import verify_api_key_or_cookie
from ..db import get_description_by_lat_lng
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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

@router.get('/description', dependencies=[Depends(verify_api_key_or_cookie)])
async def read_description_by_lat_lng(lat: float = Query(...),
                                      lng: float = Query(...),
                                      floor: int = Query(0),
                                      rotation: float = Query(...),
                                      max_count: Optional[int] = Query(10),
                                      max_distance: Optional[float] = Query(100),
                                      lang: Optional[str] = Query("ja"),
                                      sentence_length: Optional[int] = Query(3),
                                      ):
    logger.info("no live image")
    logger.info("description get")
    #近くの地点を取得
    locations = get_description_by_lat_lng(lat, lng, floor, max_distance, max_count)
    #地点を方向で振り分け
    location_per_directions, past_explanations = preprocess_descriptions(locations, rotation, lat, lng, max_distance)
    #振り分けた方向ごとの地点の情報をプロンプトに組みこむ
    prompt = construct_prompt_for_image_description(sentence_length=sentence_length,
                                                    front=location_per_directions["front"]["description"],
                                                    right=location_per_directions["right"]["description"],
                                                    left=location_per_directions["left"]["description"],
                                                    past_explanations=past_explanations,
                                                    lang=lang,
                                                    )
    #プロンプトをもとに画像の説明を生成
    #モデルの指定
    model_path = "sbintuitions/sarashina2-vision-8b"
    #モデルの読み込み
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    )
    st= time.time()
    text_prompt = processor.apply_chat_template(prompt, add_generation_prompt=True)
    inputs = processor(
    text=[text_prompt],
    padding=True,
    return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    stopping_criteria = processor.get_stopping_criteria(["\n###"])
    # Inference: Generation of the output
    output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=0.0,
    do_sample=False,
    stopping_criteria=stopping_criteria,
    )
    generated_ids = [
    output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    elapsed_time = time.time() - st
    #st = time.time()
    #(original_result, query) = await gpt_agent.query_with_images(prompt=prompt, response_format=TranslatedDescription)
    #elapsed_time = time.time() - st
    #description = parsed_value(original_result, "description")
    #translated = parsed_value(original_result, "translated")
    #lang = parsed_value(original_result, "lang")
    #logger.info("Time taken: %s", elapsed_time)
    #logger.info("Generated description: %s", description)
    #logger.info("Translated description: %s", translated)
    #logger.info("Language: %s", lang)

    # log
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    log_json(directory=date, name="params", data={
        "lat": lat,
        "lng": lng,
        "rotation": rotation,
        "max_count": max_count,
        "max_distance": max_distance,
        "sentence_length": sentence_length,
        "prompt": prompt,
        "lang": "ja",
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
        'description': output_text[0],
        'lang': "ja",
        'model': model_path,
    }

