set -e

# data paths
MAIN_PATH=$PWD
OUTPATH=$PWD/data/xnli
XNLI_PATH=$PWD/data/xnli/XNLI-1.0
PROCESSED_PATH=$PWD/data/processed/XLM15
#CODES_PATH=$MAIN_PATH/codes_xnli_15
#VOCAB_PATH=$MAIN_PATH/vocab_xnli_15

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

# English train set
echo "*** Preparing English train set FOR ENCODER/CLASSIFIER ****"
cat $OUTPATH/XNLI-MT-1.0/multinli/multinli.train.en.tsv | sed 's/\tcontradictory/\tcontradiction/g' > $XNLI_PATH/en.train

for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do
	echo "*** Preparing other lg train set FOR ALIGNING ENCODER ****"
	cat $OUTPATH/XNLI-MT-1.0/multinli/multinli.train.$lg.tsv | sed 's/\tcontradictory/\tcontradiction/g' > $XNLI_PATH/$lg.train
done


# validation and test sets
for lg in ar bg de el en es fr hi ru sw th tr ur vi zh; do

  echo "*** Preparing validation and test sets in $lg ***"
  echo -e "premise\thypo\tlabel" > $XNLI_PATH/$lg.valid
  echo -e "premise\thypo\tlabel" > $XNLI_PATH/$lg.test

  # label
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.dev.tsv  | cut -f2 > $XNLI_PATH/dev.f2
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.test.tsv | cut -f2 > $XNLI_PATH/test.f2

  # premise/hypothesis
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.dev.tsv  | cut -f7 | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/dev.f7
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.dev.tsv  | cut -f8 | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/dev.f8
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.test.tsv | cut -f7 | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/test.f7
  awk -v lg=$lg '$1==lg' $XNLI_PATH/xnli.test.tsv | cut -f8 | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $XNLI_PATH/test.f8

  paste $XNLI_PATH/dev.f7  $XNLI_PATH/dev.f8  $XNLI_PATH/dev.f2  >> $XNLI_PATH/$lg.valid
  paste $XNLI_PATH/test.f7 $XNLI_PATH/test.f8 $XNLI_PATH/test.f2 >> $XNLI_PATH/$lg.test

  rm $XNLI_PATH/*.f2 $XNLI_PATH/*.f7 $XNLI_PATH/*.f8
done

#rm -rf $PROCESSED_PATH/eval/XNLI
#mkdir -p $PROCESSED_PATH/eval/XNLI


