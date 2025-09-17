# Audio vocal comparison

Python V3.11

## Installation
```
conda create -n qadrai python=3.11 pip
conda activate qadrai

pip install -r ./requirements.txt
```
## Audio data
https://everyayah.com/
```
wget https://thenoblequran.s3.amazonaws.com/recitations/khalifa/001002.mp3

wget https://everyayah.com/data/AbdulSamad_64kbps_QuranExplorer.Com/001002.mp3
wget https://everyayah.com/data/Alafasy_64kbps/001002.mp3

wget https://download.quranicaudio.com/qdc/mishari_al_afasy/murattal/1.mp3
```

## Run
```
python src/cosine_compare.py  > output/cosine_compare.txt

python src/compare_audio.py  > output/compare_audio.txt

python src/compare_vocals.py  > output/compare_vocals.txt
```