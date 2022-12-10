#!usr/bin/env bash

file="xraynames.txt"

while read -r line; do
    cp $line . 
echo -e "$line\n"
done <$file 
