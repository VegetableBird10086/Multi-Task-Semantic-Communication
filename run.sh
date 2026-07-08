#  python main.py --datasets  libri_sb --bs 16 --lr 1e-4 --optims Adam --save_path checkpoint_libri --device_id 6 --epochs 6
# python main.py --datasets  mmdb --bs 32 --lr 1e-4 --optims Adam --save_path checkpoint_mmdb --device_id 7 --epochs 6


# python main.py --datasets  VQAv2 --bs 32 32 32 --lr 1e-4 1e-4 1e-4 --optims BertAdam Adam Adam --save_path checkpoint --device_id 5 --epochs 20
# python main.py --datasets  mmdb --bs 32 32 32 --lr 1e-4 1e-4 1e-4 --optims Adam Adam Adam --save_path checkpoint_mmdb_not_fusion --device_id 2 --epochs 6


# python main.py --datasets  mmdb --bs 8 32 32 --lr 1e-4 1e-4 1e-4 --optims BertAdam Adam Adam --save_path checkpoint_mmdb_fusion --device_id 4 --epochs 50


# python main.py --datasets  mmhs --bs 32 32 32 --lr 1e-2 1e-4 1e-4 --optims Adam Adam Adam --save_path checkpoint_mmdb_fusion --device_id 5 --epochs 6




# python main.py --datasets  twitter_sarcasm --bs 32 --lr 1e-4 --optims BertAdam --save_path checkpoint_mmdb_fusion --device_id 4 --epochs 8


python main.py --datasets  hmdb51   --bs 8 32 32 --lr 1e-4 1e-4 1e-4 --optims Adam Adam Adam --save_path checkpoint_video --device_id 6 --epochs 5