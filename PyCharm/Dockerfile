FROM python:3.8

WORKDIR /capstone

COPY . .

RUN pip install flask
RUN pip install tensorflow
RUN pip install tensorflow_hub
RUN pip install pillow
RUN pip install numpy

ENTRYPOINT ["python"]
CMD ["capstone.py"]