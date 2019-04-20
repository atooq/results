export ROOT=$rootdir
OLD_PWD=$PWD

DIR=v0.5.0/nvidia/submission/code/rnn_translator

export DATASET_DIR=$rootdir/$DIR/data
export PATH=$PATH:$HOME/.local/bin

git clone https://github.com/nvidia/apex --depth=1 && \
  cd apex && \
  pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --user .

cd .. && \
    pip install --user -r requirements.txt && \
    python setup.py install --user

cd $ROOT/pipeline && pip install --force-reinstall .

cd $OLD_PWD

git checkout pipeline

bash run_gnmt16.sh
