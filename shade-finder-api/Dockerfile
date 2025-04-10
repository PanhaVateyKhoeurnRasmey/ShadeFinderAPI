FROM public.ecr.aws/lambda/python:3.11
COPY requirements.txt .
COPY skin_tone_classifier.pkl .
COPY lambda_function.py .
RUN yum update -y
#used to get rid of Error ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN yum install mesa-libGL -y
RUN yum install libglvnd-glx -y
RUN yum install gcc openssl-devel wget tar -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD [ "lambda_function.lambda_handler" ]