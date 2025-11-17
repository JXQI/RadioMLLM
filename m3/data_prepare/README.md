## 心脏
```bash
DATA_INFO="/home/tx-deepocean/data2/jxq/data/mmedagent/src/heart/processed/data_info_zh_new.json"
CONFIG_FILE="experts/heart/config_heart.md"
MAX_WORKERS=20
TEST_FRAC=1
OUT_FILEPREFIX="/home/tx-deepocean/data2/jxq/data/mmedagent/processed/heart/expert_heart_conv_v0"
python experts/heart/main_heart_multhreads.py --data_info ${DATA_INFO} --config_file ${CONFIG_FILE} --max_workers ${MAX_WORKERS} --out_fileprefix ${OUT_FILEPREFIX} --test_frac ${TEST_FRAC}
```

## 胸肺
```bash
DATA_INFO="/home/tx-deepocean/data2/jxq/data/mmedagent/src/chest/processed/data_info_zh_new.json"
CONFIG_FILE="experts/chest/config_chest.md"
KNOWLOGE_FILE="/home/tx-deepocean/data1/jxq/code/ragflow/example/dataset/2025.1-NCCN.json"
MAX_WORKERS=20
TEST_FRAC=1
OUT_FILEPREFIX="/home/tx-deepocean/data2/jxq/data/mmedagent/processed/chest/expert_chest_conv_v0"
python experts/chest/main_chest_multhreads.py --data_info ${DATA_INFO} --config_file ${CONFIG_FILE} --max_workers ${MAX_WORKERS} --knowloge_file ${KNOWLOGE_FILE} --out_fileprefix ${OUT_FILEPREFIX} --test_frac ${TEST_FRAC}
```