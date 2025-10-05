FROM python:3.5-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install requirements.txt
COPY  . /app
EXPOSE 3000
CMD [ "python","src/app.py" ]