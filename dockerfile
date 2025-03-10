FROM python:3.12-slim

# sets the working directory
WORKDIR /application

COPY local_app.py requirements.txt .

RUN pip3 install -r requirements.txt

RUN mkdir /application/icon
COPY icon /application/icon

# listen to Streamlit’s (default) port: 8501
EXPOSE 8501

# configure a container that will run as an executable
ENTRYPOINT ["streamlit", "run", "local_app.py", "--server.port=8501"]
