#!/bin/bash
yum update -y
amazon-linux-extras install nginx1
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/nginx/cert.key -out /etc/nginx/cert.crt
