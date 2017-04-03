apt-get update
apt-get install -y git
cd /usr/local/
git clone -q https://github.com/GPflow/GPflow.git
cd GPflow
python setup.py develop
rm /notebooks/*
cd /notebooks
git clone -q https://github.com/ShibataLabPrivate/GPyWorkshop.git
