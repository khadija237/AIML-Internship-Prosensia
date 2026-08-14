# production dockerfile - day 26
# python:3.11-slim base image = lightweight, fast

FROM python:3.11-slim

WORKDIR /app

# copy requirements first (layer caching)
COPY requirements.txt .

# install without cache to keep image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy app files
COPY main.py .
COPY loan_approval_model.pkl .

# non-root user for security
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# model loads ONCE on startup (not inside route)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
