#!/bin/bash

limit='-y'
# limit="$limit -ss 60 -to 180"
# limit="$limit -to 00:15:00"

dirUser="/home/sameer"
dirProj="$dirUser/Shared/Sync/Private/Work/Projects/qadrai"
dirRnn="$dirProj/rnn"

cores_count=$(nproc --all)
threads_count=$((cores_count - 2))
ffmpeg="/usr/bin/ffmpeg -nostdin -loglevel error -stats -threads $threads_count"
echo "cores_count:$cores_count | threads_count:$threads_count | $ffmpeg"
    
filePath=$1
# get the base name without extension
fileNom="${filePath%.*}"

# $ffmpeg -i $fileNom.mp3 $limit $fileNom.pl.mp3
# sleep 2

$ffmpeg -i $fileNom.mp3 -af "arnndn=m=$dirRnn/cb.rnnn" $limit $fileNom.cb.mp3
sleep 2

# $ffmpeg -i $fileNom.cb.mp3 -af "arnndn=m=$dirRnn/sh.rnnn" -y $fileNom.sh.mp3
# sleep 2

$ffmpeg -i $fileNom.cb.mp3 -af "arnndn=m=$dirRnn/bd.rnnn" -y $fileNom.clean.mp3
sleep 2

# $ffmpeg -i input.mp3 -af "agate=threshold=-30dB:ratio=10:attack=1:release=100" output.mp3

rm -f $fileNom.pl.mp3
rm -f $fileNom.bd.mp3
rm -f $fileNom.sh.mp3
rm -f $fileNom.cb.mp3
	