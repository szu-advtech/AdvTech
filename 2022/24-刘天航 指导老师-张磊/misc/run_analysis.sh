#!/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PORT=30000
HTTPVERSION=1.1
HOST=127.0.0.1
URLPATH=/1048576
QLOGDIR=/tmp/qlog
CONN_FLOW_CONTROL=15728640
STREAM_FLOW_CONTROL=6291456

mkdir -p ${BASEDIR}/local
mkdir -p ${QLOGDIR}/client
mkdir -p ${QLOGDIR}/server
rm -rfv ${QLOGDIR}/client/*
rm -rfv ${QLOGDIR}/server/*

# Set network condition
sudo tcdel lo --all
sudo tcset lo --rate 100mbps --delay 5ms --direction outgoing

# Remove container
docker stop mvfst
docker rm mvfst

# Start docker container
docker run \
-d \
--name mvfst \
-v ${QLOGDIR}/server:/logs \
-v /certs:/certs \
--entrypoint /proxygen/_build/proxygen/bin/hq \
-p ${PORT}:${PORT}/udp \
-p ${HOST}:${PORT}:${PORT}/tcp \
lnicco/mvfst-qns:latest \
--mode=server \
--cert=/certs/leaf_cert.pem \
--key=/certs/leaf_cert.key \
--port=${PORT} \
--httpversion=${HTTPVERSION} \
--h2port=${PORT} \
--qlogger_path=/logs \
--host=0.0.0.0 \
--congestion=bbr \
--pacing=true \
--v=0

# Run clients

# Proxygen
/quic/proxygen/proxygen/_build/proxygen/httpserver/hq \
--log_response=false \
--mode=client \
--stream_flow_control=${STREAM_FLOW_CONTROL} \
--conn_flow_control=${CONN_FLOW_CONTROL} \
--use_draft=true \
--draft-version=29 \
--httpversion=${HTTPVERSION} \
--qlogger_path=${QLOGDIR}/client \
--host=${HOST} \
--port=${PORT} \
--path=${URLPATH} \
--v=0

sleep 2

mkdir -p ${BASEDIR}/local/proxygen_h3/client
mkdir -p ${BASEDIR}/local/proxygen_h3/server
mv /tmp/qlog/server/* ${BASEDIR}/local/proxygen_h3/server/
mv /tmp/qlog/client/* ${BASEDIR}/local/proxygen_h3/client/

# ngtpc2
/quic/ngtcp2/examples/client \
--quiet \
--exit-on-all-streams-close \
--max-data=${CONN_FLOW_CONTROL} \
--max-stream-data-uni=${STREAM_FLOW_CONTROL} \
--max-stream-data-bidi-local=${STREAM_FLOW_CONTROL} \
--group=X25519 \
--qlog-dir=${QLOGDIR}/client \
${HOST} \
${PORT} \
https://${HOST}:${PORT}${URLPATH}

sleep 2

mkdir -p ${BASEDIR}/local/ngtcp2_h3/client
mkdir -p ${BASEDIR}/local/ngtcp2_h3/server
mv /tmp/qlog/server/* ${BASEDIR}/local/ngtcp2_h3/server/
mv /tmp/qlog/client/* ${BASEDIR}/local/ngtcp2_h3/client/

# chrome h3
$BASEDIR/connect_h3.sh ${HOST} ${PORT} ${URLPATH}

sleep 2

mkdir -p ${BASEDIR}/local/chrome_h3/client
mkdir -p ${BASEDIR}/local/chrome_h3/server
mv /tmp/qlog/server/* ${BASEDIR}/local/chrome_h3/server/
mv /tmp/netlog/chrome_h3.json ${BASEDIR}/local/chrome_h3/client/chrome_h3.json

# chrome h2
$BASEDIR/connect_h2.sh ${HOST} ${PORT} ${URLPATH}

sleep 2

mkdir -p ${BASEDIR}/local/chrome_h2/client
mkdir -p ${BASEDIR}/local/chrome_h2/server
mv /tmp/netlog/chrome_h2.json ${BASEDIR}/local/chrome_h2/client/chrome_h2.json


docker run \
-d \
--name quiche \
-v ${QLOGDIR}/server:/logs \
-v /certs:/certs \
-v /www:/www \
--entrypoint /usr/local/bin/quiche-server \
-p ${PORT}:${PORT}/udp \
-e QLOGDIR=/logs \
cloudflare/quiche \
--listen "[::]:${PORT}" \
--cert /certs/leaf_cert.pem \
--key /certs/leaf_cert.key \
--root /www \
--max-data 107374182 \
--max-stream-data 107374182 \
--no-retry