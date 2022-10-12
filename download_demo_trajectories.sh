wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jcaBxLc0wqJ-PeIee2aJaMFy9L9k3dXi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jcaBxLc0wqJ-PeIee2aJaMFy9L9k3dXi" -O NCF_demo_trajectories.tar.xz && rm -rf /tmp/cookies.txt
mv NCF_demo_trajectories.tar.xz ./data_collection/test_data/
cd ./data_collection/test_data/
tar -xf NCF_demo_trajectories.tar.xz
rm NCF_demo_trajectories.tar.xz
mv NCF_demo_trajectories/* .
rm -r NCF_demo_trajectories
echo "Trajectories for NCF demo downloaded"