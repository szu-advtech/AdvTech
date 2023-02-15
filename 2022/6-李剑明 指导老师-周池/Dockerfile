FROM docker.io/library/python:3.10-slim as base
FROM base as builder

RUN mkdir /install
WORKDIR /install

COPY requirements.txt /requirements.txt

RUN pip install --prefix=/install --no-warn-script-location -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

FROM base

COPY --from=builder /install /usr/local
COPY src /app
WORKDIR /app

CMD ["python", "-u", "scheduler.py"]
