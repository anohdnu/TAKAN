DATASET=CUB
N_CLASS=200

# save directory
OUTDIR=/outputs/${DATASET}/CUB5_linear

# hard coded inputs
GPUID='0'
CONFIG=configs/cub_5tasks.yaml
REPEAT=5
OVERWRITE=1

# process inputs
mkdir -p $OUTDIR


# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ${OUTDIR}/coda-p


###############################################################

# save directory
OUTDIR=/outputs/${DATASET}/CUB5_takan

# hard coded inputs
GPUID='0'
CONFIG=configs/cub_5tasks_kac.yaml
REPEAT=5
OVERWRITE=1

# process inputs
mkdir -p $OUTDIR


# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ${OUTDIR}/coda-p