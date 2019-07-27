FROM python:alpine3.7
COPY . /AgeGender
WORKDIR /AgeGender
RUN pip install -r requirements.txt
CMD python ./AgeGender.py.py