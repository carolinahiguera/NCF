wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qSYorxAarrjD5lae5H6HNf8yxc0rOd9B' -O NCF_demo_train_objects.tar.xz
mv NCF_demo_train_objects.tar.xz ./data_collection/objects/train/
cd ./data_collection/objects/train/
tar -xf NCF_demo_train_objects.tar.xz
rm NCF_demo_train_objects.tar.xz
mv NCF_demo_train_objects/* .
rm -r NCF_demo_train_objects
echo "Objects assets for NCF train demo downloaded"