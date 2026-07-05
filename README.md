# Multi-Task Semantic Communication

## 环境

```bash
sudo apt update && sudo apt install -y aria2 unzip

conda create -n mtsc python=3.9 -y
conda activate mtsc

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install numpy==1.23.5 transformers==4.28.1 speechbrain==0.5.15 \
  omegaconf==2.3.0 sentencepiece python_speech_features scikit-learn scipy \
  pandas pillow tqdm h5py einops -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## MM-IMDb

```bash
mkdir -p data/mmimdb
aria2c -c -d data/mmimdb -o mmimdb.tar.gz -x16 -s64 -k1M \
  https://archive.org/download/mmimdb/mmimdb.tar.gz
tar -xzf data/mmimdb/mmimdb.tar.gz -C data/mmimdb
```

## VQAv2 验证集

```bash
mkdir -p data/vqav2_fastrnn/data/img

aria2c -c -d data/vqav2_fastrnn/data \
  https://nlp.cs.unc.edu/data/lxmert_data/vqa/minival.json
aria2c -c -d data/vqav2_fastrnn/data \
  https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_ans2label.json
aria2c -c -d data/vqav2_fastrnn/data \
  https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json
aria2c -c -d data/vqav2_fastrnn -o val2014_obj36.zip -x16 -s64 -k1M \
  https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip

unzip -j data/vqav2_fastrnn/val2014_obj36.zip '*val2014_obj36.tsv' \
  -d data/vqav2_fastrnn/data/img
```

生成 TSV 索引：

```bash
python - <<'PY'
src = 'data/vqav2_fastrnn/data/img/val2014_obj36.tsv'
dst = 'data/vqav2_fastrnn/data/img/val2014_offset.txt'
with open(src) as f, open(dst, 'w') as out:
    while True:
        offset = f.tell()
        line = f.readline()
        if not line:
            break
        out.write(f'{line.split(chr(9), 1)[0]}\t{offset}\n')
PY
```
