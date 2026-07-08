# Multi-Task Semantic Communication

## 环境

```bash
sudo apt update && sudo apt install -y aria2 unzip

conda create -n mtsc python=3.9 -y
conda activate mtsc

pip install torch==1.13.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install numpy==1.23.5 transformers==4.31.0 omegaconf==2.3.0 \
  tqdm boto3 requests -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 下载 BERT 预训练权重和 tokenizer

模型和 tokenizer 固定从项目内的 `pretrained/bert-base-uncased` 加载，不会联网回退下载。使用 Hugging Face 镜像下载：

```bash
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download google-bert/bert-base-uncased \
  config.json pytorch_model.bin vocab.txt tokenizer.json tokenizer_config.json \
  --local-dir ./pretrained/bert-base-uncased
```

下载完成后的必要文件如下：

```text
pretrained/bert-base-uncased/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

`pretrained/` 已加入 `.gitignore`，模型权重不会提交到 Git。

## 统计 Transceiver 参数量

在 `mtsc` 环境中按当前代码构造模型并统计：

```bash
conda run -n mtsc python count_transceiver_params.py
```

也可以直接统计训练代码保存的完整模型或 state dict 检查点：

```bash
conda run -n mtsc python count_transceiver_params.py \
  --checkpoint checkpoint/epoch_0.pth
```

脚本使用 PyTorch 原生的 `numel()`，输出总参数量、可训练/冻结参数量、参数存储量以及一级组件明细。直接构造模型时会从 `pretrained/bert-base-uncased` 加载预训练权重。

## VQAv2 验证集

```bash
mkdir -p data/vqav2/data/img

aria2c -c -d data/vqav2/data \
  https://nlp.cs.unc.edu/data/lxmert_data/vqa/minival.json
aria2c -c -d data/vqav2/data \
  https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_ans2label.json
aria2c -c -d data/vqav2/data \
  https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json
aria2c -c -d data/vqav2 -o val2014_obj36.zip -x16 -s64 -k1M \
  https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip

unzip -j data/vqav2/val2014_obj36.zip '*val2014_obj36.tsv' \
  -d data/vqav2/data/img
```

生成 TSV 索引：

```bash
python - <<'PY'
src = 'data/vqav2/data/img/val2014_obj36.tsv'
dst = 'data/vqav2/data/img/val2014_offset.txt'
with open(src) as f, open(dst, 'w') as out:
    while True:
        offset = f.tell()
        line = f.readline()
        if not line:
            break
        out.write(f'{line.split(chr(9), 1)[0]}\t{offset}\n')
PY
```

## 推理测试

下载已训练好的模型权重：

```bash
mkdir -p checkpoints

aria2c -c -d checkpoints -o sommsc.pth \
  https://github.com/VegetableBird10086/Multi-Task-Semantic-Communication/releases/download/model/sommsc.pth
```

下载完成后，使用测试脚本进行推理测试：

```bash
bash dist_test.sh checkpoints/sommsc.pth
```

其中 `checkpoints/sommsc.pth` 为模型权重路径，也可以替换为其他本地 checkpoint 路径，例如：

```bash
bash dist_test.sh checkpoints/xxxx.pth
```
