#!/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PORT=443
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

# Run clients

# Proxygen
/home/sky/proxygen/proxygen/_build/proxygen/httpserver/hq \
--log_response=false \
--mode=client \
--stream_flow_control=${STREAM_FLOW_CONTROL} \
--conn_flow_control=${CONN_FLOW_CONTROL} \
--quic_version=1 \
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
/home/sky/ngtcp2/ngtcp2/examples/client \
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


