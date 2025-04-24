FROM python:3.9

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U transformers==4.47.0 torch torchvision pillow protobuf sentencepiece accelerate


COPY --from=src ./app /app
COPY --from=src ./static /static
COPY --from=src ./templates /templates
RUN mkdir -p /static_js_lib/ && curl https://cdn.jsdelivr.net/npm/ol@v7.3.0/dist/ol.js -o /static_js_lib/ol.js
RUN curl -L https://cdnjs.cloudflare.com/ajax/libs/exif-js/2.3.0/exif.min.js -o /static_js_lib/exif.min.js

WORKDIR /app

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
