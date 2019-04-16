ROOT=$rootdir

DIR=results/v0.5.0/nvidia/submission/code/rnn_translator

export DATASET_DIR=$WORKSPACE/$DIR/data

git clone https://github.com/nvidia/apex --depth=1 && \
  cd apex && \
  pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --user .

cd .. && \
    pip install --user -r requirements.txt && \
    python setup.py install --user

./run_and_time.sh
