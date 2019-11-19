FROM python:3.7.4

RUN mkdir -p /src/app
WORKDIR /src/app

COPY ./ /src/app

# RUN pip3 install -qr requirements.txt 

RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install pythainlp
RUN pip3 install emoji
RUN pip3 install scikit-learn
RUN pip3 install fastai
RUN pip3 install tqdm
RUN pip3 install dataclasses
RUN pip3 install matplotlib
RUN pip3 install fastprogress
RUN pip3 install flask
RUN pip3 install seaborn

# RUN pip3 install pythainlp

EXPOSE 5000

RUN python3 Sentiment_Kiite.py
# CMD ["python3", "test.py"]