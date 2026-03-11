# Dockerfile - minimal image, just what Python needs
FROM python:3.11-slim

# set working directory
WORKDIR /app

# install dependencies first (layer caching — only reruns if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY . .

# expose default ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# start FastAPI
# host 0.0.0.0 lets the contained listen to requests from outside it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]