#!/usr/bin/env bash

set -e

MAIN_PATH = $PWD
DATA_PATH = $PWD/data
EMBEDDING_PATH = $PWD/data/embedding

lgs = "ar en bg de el es fr hi ru sw th tr ur vi zh"
for lg in $lgs; do
   mkdir $EMBEDDING_PATH/lg
   TEXT_LINK = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc." + $lg + ".300.vec.gz"
   BIN_LINK = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc." + $lg + ".300.bin.gz"

   wget -c $TEXT_LINK -P $EMBEDDING_PATH
   wget -c $BIN_LINK -P $EMBEDDING_PATH
done

