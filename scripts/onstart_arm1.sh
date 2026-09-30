#!/usr/bin/env bash
# vast.ai onstart for Arm 1: fetch AmalgaMatch, clone the pre-registered branch, launch box_arm1.sh detached.
cd /root
(command -v git && command -v wget) >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git wget; }
nohup bash -c 'for i in 1 2 3; do wget -c -q -O /root/AmalgaMatch_Dataset.zip https://fordatis.fraunhofer.de/bitstream/fordatis/478/1/AmalgaMatch_Dataset.zip && break; done' > /root/wget.log 2>&1 &
[ -d cma ] || git clone -q -b triage-ext https://github.com/fronkt/correlative-microscopy-alignment.git cma
nohup bash /root/cma/scripts/box_arm1.sh > /root/arm1.log 2>&1 &
