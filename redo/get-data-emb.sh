set -e
# lg=$1  # input language

# data path
MAIN_PATH=$PWD
DATA_PATH=$PWD/tools

# Embedding
EMB_DIR=$DATA_PATH/emb
# tools
mkdir -p $DATA_PATH

#
# Download and install tools
#

cd $DATA_PATH
# Download Embeddings

if [ ! -d "EMB_DIR"]; then
	mkdir -p EMB_DIR


echo "*** Downloading word embeddings from fastText ***"
# validation and test sets
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
  echo "*** Downloading word embeddings from fastText for $lg ***"

  BIN-link = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc."
  BIN-ext = ".300.bin.gz"# extension
  TXT-link = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc."
  TXT-ext = ".300.vec.gz"
  
  BINurl = "$BIN-link$lg$BIN-ext"
  TXTurl = "$TXT-link$lg$TXT-ext"

  wget $BINurl
  wget $TXTurl
  # decompress?
done
