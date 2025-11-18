apt-get update
apt-get install tmux unzip

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

mkdir data

mv train2017.zip data/train2017.zip
mv val2017.zip data/val2017.zip
mv test2017.zip data/test2017.zip
mv annotations_trainval2017.zip data/annotations_trainval2017.zip

cd data/

unzip train2017.zip
unzip val2017.zip 
unzip test2017.zip 
unzip annotations_trainval2017.zip

