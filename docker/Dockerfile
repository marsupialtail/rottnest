sudo apt update
sudo apt install unzip
sudo apt install parallel
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir /data
sudo mount /dev/nvme1n1 /data
sudo chmod 777 /data

cd /data
cat <<EOF > generate_urls.py
import sys
batch = int(sys.argv[1])
for n in range(batch * 25, batch * 25 + 25):
    i = n // 50
    j = n % 50
    print(f"https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2024-10/00{i}_000{str(j).zfill(2)}.parquet")
EOF

python3 generate_urls.py $1 > files
cat files | parallel wget -q {} > down.log

sudo apt-get -y install python3-pip
pip3 install --break-system-packages rottnest==1.4.0

mkdir index
ls *parquet > files
cat <<EOF > index.py
import rottnest
from tqdm import tqdm

files = open("files").readlines()
for i, file in tqdm(enumerate(files)):
    rottnest.index_file_bm25(file.strip(), "text", f"index/{file.split('.')[0]}")
EOF

cd /data
cat <<EOF > merge.py
import rottnest
files = open("files").readlines()
indices = ["index/" + f.replace(".parquet\n", "") for f in files]
rottnest.merge_index_bm25("merged_index", indices)
import polars
polars.read_parquet("merged_index.meta").with_columns(polars.concat_str(polars.lit("https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2024-10/"), polars.col("file_path")).alias("file_path")).write_parquet("merged_index.meta")
EOF

python3 merge.py

aws configure set aws_access_key_id 
aws configure set aws_secret_access_key 
aws configure set region us-west-2
aws configure set output json

aws s3 cp merged_index.meta s3://fineweb-index/$1.meta
aws s3 cp merged_index.lava s3://fineweb-index/$1.lava