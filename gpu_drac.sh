#!/bin/bash
#SBATCH --account=def-pbranco
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=62GB
#SBATCH --time=12:00:00
#SBATCH --signal=SIGUSR1@90

# GPU NODE SETTINGS
# -- 10GB Mig --- 15GB --- 2 Cores
# -- 20GB Mig --- 31GB --- 4 Cores
# -- 40GB Mig --- 62GB --- 8 Cores
# -- 80GB H100 -- 124GB -- 16 Cores
#s#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#s#SBATCH --gpus=h100:1
#s#SBATCH --array=1-5%1

module load StdEnv/2023
module load gcc arrow/22.0.0 python/3.12.4 cmake/3.27.7 cuda/12.6 scipy-stack


HOME=$PWD
export HOME
export _JAVA_OPTS="-Xmx30G -Xms4G"
export TOKENIZERS_PARALLELISM="true"
export TRANSFORMERS_VERBOSITY="error"
export JOERN_CLI="$SLURM_TMPDIR/joern/joern-cli/"
export HF_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RORQUAL_DIR="/home/adefu020/links/projects/def-pbranco/adefu020"
PROJECTS_DIR="/home/adefu020/projects/def-pbranco/adefu020"
if [ -d $RORQUAL_DIR ]; then
    PROJECTS_DIR=$RORQUAL_DIR
fi
export HF_HOME="$PROJECTS_DIR/.cache"
export HF_HUB_CACHE="$PROJECTS_DIR/.cache"



cp $PROJECTS_DIR/TimeWillTell/pyproject.toml $SLURM_TMPDIR/pyproject.toml
cd $SLURM_TMPDIR
python -m venv venv
source venv/bin/activate

echo "Copying over dependencies"
cp -r $PROJECTS_DIR/TimeWillTell/deps/ $SLURM_TMPDIR/deps

python --version
python -m pip install --no-index $SLURM_TMPDIR/deps/*.whl
python -m pip install --no-index  .

echo "Done Installing Python Packages"

NEED_LLAMA=0
for arg in "$@"; do
    [[ "$arg" == --model=*gguf* ]] && NEED_LLAMA=1 && break
done

module list
if [[ $NEED_LLAMA -eq 1 ]]; then
    LLAMA_SRC="$PROJECTS_DIR/TimeWillTell/deps/llama.cpp"
    if [[ ! -d "$LLAMA_SRC" ]]; then
        echo "ERROR: llama.cpp source not found at $LLAMA_SRC"
        exit 1
    fi
    cp -r "$LLAMA_SRC" "$SLURM_TMPDIR/llama.cpp"
    cd "$SLURM_TMPDIR/llama.cpp"
    rm -rf build
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release -j 8
    export LLAMABUILDDIR="$SLURM_TMPDIR/llama.cpp/build"
    echo "llama-server built at $LLAMABUILDDIR/bin/llama-server"
    export PATH="$SLURM_TMPDIR/llama.cpp/build/bin:$PATH"
fi

nvidia-smi
echo "PWD: $(pwd)"
echo "Changing Dir to $HOME/TimeWillTell"
cd $HOME/TimeWillTell
echo "Changed to $(pwd)"

source $SLURM_TMPDIR/venv/bin/activate
which pip
which python
which pip3
which python3
ls -la
echo "********** Launching **********"
python main.py --dataset=megavul $@ && scancel $SLURM_ARRAY_JOB_ID

