#!/bin/bash
sed 's/mpr = MPRester("MAPI_KEY")/mpr = MPRester(\"'"$1"'\")/g' ingrained/construct.py >> ingrained/temp.py
rm -r ingrained/construct.py ;
mv ingrained/temp.py ingrained/construct.py;