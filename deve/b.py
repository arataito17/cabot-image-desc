import base64
import datetime
import json
import logging
import math
import os
import time
from fastapi import APIRouter, Depends, Query, Request, HTTPException
from typing import Optional

# Import required functions/classes from openai_agent and auth
from ..openai.openai_agent import GPTAgent
from ..openai.openai_agent import TranslatedDescription, construct_prompt_for_image_description
from ..openai.openai_agent import StopReason, construct_prompt_for_stop_reason
from .auth import verify_api_key_or_cookie
from ..db import get_description_by_lat_lng
from transformers import AutoModelForCausalLM, AutoProcessor


router = APIRouter()
gpt_agent = GPTAgent()

logger = logging.getLogger(__name__)
    
class BaseMLLM:
        def __init__(self):
            pass
            
        def __call__(self):
            print("NO MODEL\n")
            print("please select a model")
            return None
            
        def sara2_woi(self, lat, lng, floor, rotation, max_distance, max_count, sentence_length, tags, lang):
            self.model_path = "sbintuitions/sarashina2-vision-8b"
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            #####ここから
            self.locations = get_description_by_lat_lng(lat, lng, floor, max_distance, max_count)

            location_per_directions, past_explanations = preprocess_descriptions(self.locations, rotation, lat, lng, max_distance)
            #####ここまでは全部のモデルで共通してるからclassの外に出してもいいかも
            self.prompt = construct_prompt_for_image_description_sara2_woi(sentence_length=sentence_length,
                                                front=location_per_directions["front"]["description"],
                                                right=location_per_directions["right"]["description"],
                                                left=location_per_directions["left"]["description"],
                                                past_explanations=past_explanations,
                                                image_tags=tags,
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
            temperature=0.0,
            do_sample=False,
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
            date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
            return output_text[0], self.prompt, date, self.locations, self.model_path, self.processor, self.model
        
        ###
        #def sara2(self, lat, lng, floor, rotation, max_distance, max_count, sentence_length, tags, lang):
                

        
             

        async def gpt4o(self, lat, lng, floor,rotation, max_distance, max_count, sentence_length, tags, lang):
            ####ここから
            logger.info("description get")
            locations = get_description_by_lat_lng(lat, lng, floor, max_distance, max_count)

            location_per_directions, past_explanations = preprocess_descriptions(locations, rotation, lat, lng, max_distance)
            ####ここまでsarashina2-vision-8bと共通してるからclassの外に出してもいいかも
            prompt = construct_prompt_for_image_description_gpt(sentence_length=sentence_length,
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




