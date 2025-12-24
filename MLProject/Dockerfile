FROM python:3.9

WORKDIR /app

COPY MLProject/ /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 5000
EXPOSE 8000

CMD ["python", "prometheus_exporter.py"]