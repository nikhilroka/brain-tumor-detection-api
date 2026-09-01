FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt

RUN pip install --no-cache-dir -r ./api/requirements.txt

COPY api ./api

COPY models/brain_tumor_resnet50.pth ./models/brain_tumor_resnet50.pth

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]