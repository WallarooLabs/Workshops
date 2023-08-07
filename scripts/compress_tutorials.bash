#!/opt/homebrew/bin/bash

# This relies on Bash 4 and above
# Meant to be run from the root of this project folder

declare -A tutorials

tutorials=( 
    ["CV-Retail"]="Computer\ Vision/CV-Retail" 
    )

currentDirectory=$PWD

for zip in "${!tutorials[@]}"; 
    do (cd ${tutorials[$zip]}/..;zip -r $currentDirectory/compress_tutorials/$zip.zip $zip);
done