#!/bin/bash
sed 's/os.environ.*/os.environ["MAPI_KEY"] = \"'"$1"'\"/' ingrained/structure.py >> ingrained/temp.py ;
rm -r ingrained/structure.py ;
mv ingrained/temp.py ingrained/structure.py;
