set -e

# data paths
MAIN_PATH=$PWD
OUTPATH=$PWD/data/xnli
XNLI_PATH=$PWD/data/xnli/XNLI-1.0
PROCESSED_PATH=$PWD/data/processed/XLM15
#CODES_PATH=$MAIN_PATH/codes_xnli_15
VOCAB_PATH=$MAIN_PATH/vocab_xnli_15

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
#FASTBPE=$TOOLS_PATH/fastBPE/fast

# install tools
./install-tools.sh

# create directories
mkdir -p $OUTPATH

# download data
if [ ! -d $OUTPATH/XNLI-MT-1.0 ]; then
  if [ ! -f $OUTPATH/XNLI-MT-1.0.zip ]; then
    wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip -P $OUTPATH
  fi
  unzip $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
fi
if [ ! -d $OUTPATH/XNLI-1.0 ]; then
  if [ ! -f $OUTPATH/XNLI-1.0.zip ]; then
    wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip -P $OUTPATH
  fi
  unzip $OUTPATH/XNLI-1.0.zip -d $OUTPATH
fi

