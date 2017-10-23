#!/bin/bash 
IN=$1 
NAME=${IN%.cl} 
OUT=$NAME.h

echo "const char *"$NAME"_ocl =" >$OUT 
sed -e 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN >>$OUT 
echo ";" >>$OUT 
