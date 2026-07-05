# python eval.py --datasets  glue_sst2 glue_cola europarl --bs 32 32 32 --load_from checkpoint_mmdb_fusion/epoch_5.pth --device_id 5
python eval.py --datasets mmdb --bs 32 32 32 --load_from checkpoint_mmdb_fusion/epoch_43.pth  --device_id 5
