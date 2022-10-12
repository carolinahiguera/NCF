wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TKeJjqhDMMQS28MTPNtiC1Bez8teVKLN' -O NCF_demo_test_objects.tar.xz
mv NCF_demo_test_objects.tar.xz ./data_collection/objects/test/
cd ./data_collection/objects/test/
tar -xf NCF_demo_test_objects.tar.xz
rm NCF_demo_test_objects.tar.xz
mv NCF_demo_test_objects/* .
rm -r NCF_demo_test_objects
echo "Objects assets for NCF test demo downloaded"

