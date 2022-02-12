From python:3.9-bullseye
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY api_app/hotel.py .
ENV FLASK_APP=hotel
CMD ["python3", "flask", "run", "--host=0.0.0.0"]