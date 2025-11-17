```bash
DATA_INFO="/home/tx-deepocean/data2/jxq/data/mmedagent/src/heart/processed/data_info_zh_new.json"
CONFIG_FILE="experts/heart/config_heart.md"
MAX_WORKERS=20
TEST_FRAC=1
OUT_FILEPREFIX="/home/tx-deepocean/data2/jxq/data/mmedagent/processed/heart/expert_heart_conv_v0"
python experts/heart/main_heart_multhreads.py --data_info ${DATA_INFO} --config_file ${CONFIG_FILE} --max_workers ${MAX_WORKERS} --out_fileprefix ${OUT_FILEPREFIX} --test_frac ${TEST_FRAC}
```