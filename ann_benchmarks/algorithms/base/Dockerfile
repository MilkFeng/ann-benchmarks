FROM ubuntu:25.04

RUN apt-get update
RUN apt-get install -y python3-pip python-is-python3 build-essential git
RUN apt-get install -y cmake ninja-build

RUN python --version | grep 'Python 3.13'
RUN g++ --version | grep '14.'

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip install -r requirements.txt --break-system-packages

ENTRYPOINT ["python", "-u", "run_algorithm.py"]
