set -e
set -x

git pull origin map
cd environment/Forge/eclipse
rm -rf tasks maps
tar -zxf tasks.tar.gz
