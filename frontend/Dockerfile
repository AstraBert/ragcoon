FROM python:3.11.9-slim-bookworm

WORKDIR /frontend/
COPY ./ /frontend/

RUN pip cache purge
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["gunicorn", "--bind", "0.0.0.0:8001", "frontend:me"]