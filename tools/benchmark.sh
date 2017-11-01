#!/bin/bash
killall stemcl
mkdir -p logs
rm logs/*
rm -rf sample/*.stemcl sample/progress.pgm sample/transmissions

for i in $(seq $2)
do
	i=$(expr $i - 1)
	stemcl "$1" "$i" sample > "logs/$1-$i.txt" &
done
