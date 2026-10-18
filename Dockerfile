FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only PyTorch keeps the image small; everything else comes from requirements.txt
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml README.md LICENSE config.yaml ./
COPY src ./src
COPY app ./app
COPY .streamlit ./.streamlit
RUN pip install --no-deps -e .

# data/ and reports/ are mounted from the host, see README
# Web app:  docker run -p 8501:8501 --entrypoint streamlit <image> run app/streamlit_app.py --server.address 0.0.0.0
EXPOSE 8501
ENTRYPOINT ["python", "-m", "peatland.cli"]
CMD ["--help"]
