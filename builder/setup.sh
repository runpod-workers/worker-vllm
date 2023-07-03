#!/bin/bash

apt-get update
apt-get upgrade -y

apt-get install -y software-properties-common

apt-get autoremove -y
apt-get clean -y
rm -rf /var/lib/apt/lists/*
