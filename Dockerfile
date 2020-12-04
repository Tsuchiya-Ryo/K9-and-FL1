FROM frolvlad/alpine-miniconda3:latest

COPY . /home
WORKDIR /home

RUN pip install --no-cache dash\
    pip install --no-cache dash-bootstrap-components\
    pip install --no-cache numpy\
    pip install --no-cache dash-daq\
    pip install --no-cache efficientnet_pytorch\
    pip install --no-cache torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

CMD python app.py -h 0.0.0.0 -p $PORT