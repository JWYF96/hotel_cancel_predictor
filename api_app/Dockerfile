From python:3.9-bullseye
COPY . .
RUN pip3 install -r requirements.txt
ENV FLASK_APP=hotel
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]