import torch.nn as nn 
import torch
from .image_encoder import resnet50
from .text_encoder import BertModel
from .video_encoder import VivitModel
from .mask_cnn import MaskVGG
from transformers.models.auto.configuration_auto import AutoConfig
from omegaconf import OmegaConf
from .audio_encoder import TransformerASR
from .shared_block3 import VBFeatureExtraction as VBE
from .timesformer import TimesformerModel
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from .audio_backbone import AudioBackbone
from speechbrain.lobes.augment import SpecAugment



class SemanticEncoder(nn.Module):
    def __init__(
        self,
        args,
    ):
        super(SemanticEncoder, self).__init__()
        
        # image encoder 
        self.image_encoder = resnet50(True)
        self.image_proj = nn.Conv2d(2048, args.d_model, kernel_size=1)
        
        # text encoder 
        bert_config = OmegaConf.create({'bert_model_name': args.bert_name})
        
        
        params_config = {'config': AutoConfig.from_pretrained(args.bert_name, 
                                **OmegaConf.to_container(bert_config))}
        
        
        self.text_encoder = BertModel.from_pretrained(args.bert_name, **params_config)
        
        args.bert_config = self.text_encoder.config
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, args.d_model)
        
        # audio encoder 
        self.audio_extractor = Fbank(n_mels=80)
        self.audio_normalize = InputNormalization(update_until_epoch=4)
        
        self.audio_aug = SpecAugment(
            time_warp=False,
            time_warp_window=5,
            time_warp_mode='bicubic',
            freq_mask=True,
            n_freq_mask=4,
            time_mask=True,
            n_time_mask=4,
            replace_with_zero=False,
            freq_mask_width=15,
            time_mask_width=20,
        )

        
        self.audio_backbone = AudioBackbone(
            input_shape=(8, 10, 80),
            num_blocks=3,
            num_layers_per_block=1,
            out_channels=(64, 64, 64),
            kernel_sizes=(5, 5, 1),
            strides=(2, 2, 1),
            residuals=(False, False, True),
        )
        self.audio_encoder = TransformerASR(
            input_size=1280,
            tgt_vocab=5000,
            d_model=512,
            nhead=4,
            num_encoder_layers=12,
            num_decoder_layers=6,
            d_ffn=2048,
            dropout=0.1,
            activation=torch.nn.GELU,
            encoder_module='transformer',
            attention_type='regularMHA',
            normalize_before=True,
            causal=False,
        )
        
        
        # video encoder 
        timesformer_config = OmegaConf.create({'timesformer_config': 'facebook/timesformer-base-finetuned-k400'})
        timesformer_params_config = {'config': AutoConfig.from_pretrained('facebook/timesformer-base-finetuned-k400', **OmegaConf.to_container(timesformer_config))}
        self.video_encoder = TimesformerModel.from_pretrained('facebook/timesformer-base-finetuned-k400', **timesformer_params_config)
        # self.video_encoder = VivitModel.from_pretrained(args.vivit_name, **vivit_params_config)
        self.video_proj = nn.Linear(self.video_encoder.config.hidden_size, args.d_model)

        
        # vqa proj 
        self.vqa_proj = nn.Linear(self.text_encoder.config.hidden_size, args.d_model)
        
        self.shared_block = nn.ModuleList([
            VBE.from_pretrained("bert-base-uncased") for _ in range(4)
        ])
        
        self.img_proj = nn.Linear(2048, 768)
        
        
        # twitter 
        self.twitter = nn.Linear(2048, 768)
        
    
        
    def forward(self, data):
        task_type = data['task_type']
        
        if task_type in ['cifar10', 'cifar10_recovery']:
            features = self.image_encoder(data['image'])
            feature = features[-1]
            feature = self.image_proj(feature)
            data['image'] = feature.flatten(1)   
            
        
        elif task_type in ['glue_sst2', 'glue_cola', 'europarl']:
            text = data['text']
            token_type_ids = data['segment_ids']
            attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            device = text.device
            position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            attention_mask = (1.0 - attention_mask) * -1e5
            
            text_embedding = self.text_encoder.embeddings(
                        input_ids=text,
                        position_ids=position_ids,
                        token_type_ids=token_type_ids,
                    )          
            
            head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            encoder_outputs = self.text_encoder.encoder(
                text_embedding,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )  
            encoder_outputs = self.text_proj(encoder_outputs[0])
            data['text'] = encoder_outputs
            
        if task_type == 'VQAv2':
            # method 1
            # bs = data['image'].shape[0] 
            # features = []
            # for i in range(bs):
            #     cropped_imgs = data['image'][i] 
            #     features.append(self.image_encoder(cropped_imgs)[-1].unsqueeze(0))
            
            # data['image'] = torch.cat(features, dim=0).flatten(2)
            
           
            # print(data['image'].shape)
            
            # text = data['text']
            # token_type_ids = data['segment_ids']
            # attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            # device = text.device
            # position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            # attention_mask = (1.0 - attention_mask) * -1e5
            
            # text_embedding = self.text_encoder.embeddings(
            #             input_ids=text,
            #             position_ids=position_ids,
            #             token_type_ids=token_type_ids,
            #         )          
            
            # head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            # encoder_outputs = self.text_encoder.encoder(
            #     text_embedding,
            #     attention_mask=attention_mask,
            #     head_mask=head_mask,
            # )  
            # encoder_outputs = encoder_outputs[0] 
            # data['text'] = encoder_outputs
            
            
            # method 2
            # device = data['text'].device
            # bs = data['text'].shape[0]
            
            # visual_segment_ids = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # v_mask = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # data['memories'] = self.shared_block(input_ids = data['text'], 
            #                            token_type_ids =data['segment_ids'], 
            #                            attention_mask =data['text_mask'],
            #                            visual_feats = data['image'],
            #                            visual_token_type_ids=visual_segment_ids,
            #                            visual_attention_mask=v_mask)
       

            # data['memories'] = self.vqa_proj(data['memories'])
            # del data['image'], data['text']
            # data['memories'] = torch.cat([data['image'], data['text']], dim=1)  
            # torch.cuda.empty_cache()
            # 32 x 2048 x 1 x 1
            # 32 x 128 x 756
            
            # method 3 
            text = data['text']
            token_type_ids = data['segment_ids']
            attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            device = text.device
            position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            attention_mask = (1.0 - attention_mask) * -1e5
            
            text_embedding = self.text_encoder.embeddings(
                        input_ids=text,
                        position_ids=position_ids,
                        token_type_ids=token_type_ids,
                    )          
            
            head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            encoder_outputs = self.text_encoder.encoder(
                text_embedding,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )  
            encoder_outputs = encoder_outputs[0] 
            # data['text'] = encoder_outputs     

            image = data['image']  
            image = self.img_proj(image) 
            
            data['memories'] = torch.cat([image, encoder_outputs], dim=1)
            
            

        
        elif task_type == 'libri':
            feature, length = self.audio_backbone(data['audio'], data['length'])
            feature, length = self.audio_encoder(feature, length)
            data['audio'] = feature 
            data['length'] = length
            print(feature.shape)
            

        elif task_type == 'libri_sb':
            audio = data['audio']
            audio_length = data['audio_length']
            fbank_feats = self.audio_extractor(audio)
            cur_epoch = data['epoch']
            fbank_feats = self.audio_normalize(fbank_feats, audio_length, epoch=cur_epoch)
            fbank_feats = self.audio_aug(fbank_feats)
            label_bos = data['label_bos']
            cnn_feats = self.audio_backbone(fbank_feats)
            enc_out, pred = self.audio_encoder(
                cnn_feats, label_bos, audio_length, pad_idx=0,
            )
            data['audio'] = enc_out 
            data['audio_pred'] = pred


        
        elif task_type == 'hmdb51':
            feature = self.video_encoder(data['video'])[0][:, 0, :]
            data['video'] = self.video_proj(feature)
            
        
        elif task_type == 'twitter_sarcasm':

            # Method 1 not fusion            
            # img_features = self.image_encoder(data['image'])[-1]
            # img_features = img_features.flatten(1)
            # img_features = self.twitter(img_features)
            # img_features = img_features.unsqueeze(1)
            
            
            # text = data['text']
            # token_type_ids = data['segment_ids']
            # attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            # device = text.device
            # position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            # attention_mask = (1.0 - attention_mask) * -1e5
            
            # text_embedding = self.text_encoder.embeddings(
            #             input_ids=text,
            #             position_ids=position_ids,
            #             token_type_ids=token_type_ids,
            #         )          
            
            # head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            # encoder_outputs = self.text_encoder.encoder(
            #     text_embedding,
            #     attention_mask=attention_mask,
            #     head_mask=head_mask,
            # )  
            # encoder_outputs = encoder_outputs[0] 
            
            # data['memories'] = torch.cat((encoder_outputs, img_features), dim=1)
            
            # Method 2 fusion      
            
            # device = data['text'].device
            # bs = data['text'].shape[0]
            
            # img_features = self.image_encoder(data['image'])[-1]
            # img_features = img_features.flatten(1)
            # img_features = img_features.unsqueeze(1)
            # data['image'] = img_features
            
            
            # visual_segment_ids = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # v_mask = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # data['memories'] = self.shared_block(input_ids = data['text'], 
            #                            token_type_ids =data['segment_ids'], 
            #                            attention_mask =data['text_mask'],
            #                            visual_feats = data['image'],
            #                            visual_token_type_ids=visual_segment_ids,
            #                            visual_attention_mask=v_mask)


            # Method 3 late fusion      
            
            # device = data['text'].device
            # bs = data['text'].shape[0]
            

            # text = data['text']
            # token_type_ids = data['segment_ids']
            # attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            # device = text.device
            # position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            # attention_mask = (1.0 - attention_mask) * -1e5
            
            # text_embedding = self.text_encoder.embeddings(
            #             input_ids=text,
            #             position_ids=position_ids,
            #             token_type_ids=token_type_ids,
            #         )          
            
            # head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            # encoder_outputs = self.text_encoder.encoder(
            #     text_embedding,
            #     attention_mask=attention_mask,
            #     head_mask=head_mask,
            # )  
            # data['text'] = encoder_outputs[0]


            
            # img_features = self.image_encoder(data['image'])[-1]
            # img_features = img_features.flatten(1)
            # img_features = img_features.unsqueeze(1)
            # data['image'] = img_features
            
            
            # visual_segment_ids = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # v_mask = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # data['memories'] = self.shared_block(input_ids = data['text'], 
            #                            token_type_ids =data['segment_ids'], 
            #                            attention_mask =data['text_mask'],
            #                            visual_feats = data['image'],
            #                            visual_token_type_ids=visual_segment_ids,
            #                            visual_attention_mask=v_mask)
            
            
            # method 4 middle fusion 
            device = data['text'].device
            bs = data['text'].shape[0]
            

            text = data['text']
            token_type_ids = data['segment_ids']
            attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            device = text.device
            position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            attention_mask = (1.0 - attention_mask) * -1e5
            
            text_embedding = self.text_encoder.embeddings(
                        input_ids=text,
                        position_ids=position_ids,
                        token_type_ids=token_type_ids,
                    )          
            
            head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            encoder_outputs = self.text_encoder.encoder(
                text_embedding,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )  
            
            all_hidden_states = encoder_outputs[1]
            img_features = self.image_encoder(data['image'])
            hidden_fusion = None
            for i in range(4):
                img_features[i] = img_features[i].flatten(1)
                img_features[i] = img_features[i].unsqueeze(1)
                visual_segment_ids = torch.ones(bs, img_features[i].shape[1],dtype=torch.long).to(device)
                v_mask = torch.ones(bs, img_features[i].shape[1],dtype=torch.long).to(device)
                if hidden_fusion is not None:
                    fusion_mask=torch.ones(bs, hidden_fusion.shape[1],dtype=torch.long).to(device)
                else:
                    fusion_mask = None
            
                new_hidden_fusion = self.shared_block[i](
                    input_ids = all_hidden_states[i], 
                    token_type_ids = data['segment_ids'], 
                    attention_mask = data['text_mask'],
                    visual_feats = img_features[i],
                    visual_token_type_ids=visual_segment_ids,
                    visual_attention_mask=v_mask,
                    fusion_feat=hidden_fusion,
                    fusion_mask=fusion_mask,
                )
                if hidden_fusion is None:
                    hidden_fusion = new_hidden_fusion
                else:
                    new_hidden_fusion = torch.cat((hidden_fusion, new_hidden_fusion), dim=1)
            # print(hidden_fusion.shape)
            #sdf
            data['memories'] = hidden_fusion
            
           
            
            
            # visual_segment_ids = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # v_mask = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # data['memories'] = self.shared_block(input_ids = data['text'], 
            #                            token_type_ids =data['segment_ids'], 
            #                            attention_mask =data['text_mask'],
            #                            visual_feats = data['image'],
            #                            visual_token_type_ids=visual_segment_ids,
            #                            visual_attention_mask=v_mask)

        elif task_type == 'flickr30':

            # Method 1 not fusion            
            # img_features = self.image_encoder(data['image'])[-1]
            # img_features = img_features.flatten(1)
            # img_features = self.twitter(img_features)
            # img_features = img_features.unsqueeze(1)
            
            
            # text = data['text']
            # token_type_ids = data['segment_ids']
            # attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            # device = text.device
            # position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            # attention_mask = (1.0 - attention_mask) * -1e5
            
            # text_embedding = self.text_encoder.embeddings(
            #             input_ids=text,
            #             position_ids=position_ids,
            #             token_type_ids=token_type_ids,
            #         )          
            
            # head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            # encoder_outputs = self.text_encoder.encoder(
            #     text_embedding,
            #     attention_mask=attention_mask,
            #     head_mask=head_mask,
            # )  
            # encoder_outputs = encoder_outputs[0] 
            
            # data['memories'] = torch.cat((encoder_outputs, img_features), dim=1)
            # print(data['memories'].shape) 
            # print(data['label'])
            
            # Method 2 fusion      
            
            device = data['text'].device
            bs = data['text'].shape[0]
            
            img_features = self.image_encoder(data['image'])[-1]
            img_features = img_features.flatten(1)
            img_features = img_features.unsqueeze(1)
            data['image'] = img_features
            
            
            visual_segment_ids = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            v_mask = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            data['memories'] = self.shared_block(input_ids = data['text'], 
                                       token_type_ids =data['segment_ids'], 
                                       attention_mask =data['text_mask'],
                                       visual_feats = data['image'],
                                       visual_token_type_ids=visual_segment_ids,
                                       visual_attention_mask=v_mask)
        
        elif task_type == 'mmdb':
            # features = self.image_encoder(data['image'])
            # feature = features[-1]
            # feature = self.image_proj(feature)
            # data['image'] = feature.flatten(1)  
            
            # method 1
            # img_features = self.image_encoder(data['image'])[-1]
            # img_features = img_features.flatten(1)
            # img_features = self.twitter(img_features)
            # img_features = img_features.unsqueeze(1)
            
            
            # text = data['text']
            # token_type_ids = data['segment_ids']
            # attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            # device = text.device
            # position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            # attention_mask = (1.0 - attention_mask) * -1e5
            
            # text_embedding = self.text_encoder.embeddings(
            #             input_ids=text,
            #             position_ids=position_ids,
            #             token_type_ids=token_type_ids,
            #         )          
            
            # head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            # encoder_outputs = self.text_encoder.encoder(
            #     text_embedding,
            #     attention_mask=attention_mask,
            #     head_mask=head_mask,
            # )  
            # encoder_outputs = encoder_outputs[0] 
            
            # data['memories'] = torch.cat((encoder_outputs, img_features), dim=1)
            
            
            # device = data['text'].device
            # bs = data['text'].shape[0]
            
            # img_features = self.image_encoder(data['image'])[-1]
            # img_features = img_features.flatten(1)
            # img_features = img_features.unsqueeze(1)
            # data['image'] = img_features
            
            
            # visual_segment_ids = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # v_mask = torch.ones(bs, data['image'].shape[1],dtype=torch.long).to(device)
            # data['memories'] = self.shared_block(input_ids = data['text'], 
            #                            token_type_ids =data['segment_ids'], 
            #                            attention_mask =data['text_mask'],
            #                            visual_feats = data['image'],
            #                            visual_token_type_ids=visual_segment_ids,
            #                            visual_attention_mask=v_mask) 
            
            # method 3 middle fusion
            
            device = data['text'].device
            bs = data['text'].shape[0]
            

            text = data['text']
            token_type_ids = data['segment_ids']
            attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            device = text.device
            position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            attention_mask = (1.0 - attention_mask) * -1e5
            
            text_embedding = self.text_encoder.embeddings(
                        input_ids=text,
                        position_ids=position_ids,
                        token_type_ids=token_type_ids,
                    )          
            
            head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            encoder_outputs = self.text_encoder.encoder(
                text_embedding,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )  
            
            all_hidden_states = encoder_outputs[1]
            img_features = self.image_encoder(data['image'])
            hidden_fusion = None
            for i in range(4):
                img_features[i] = img_features[i].flatten(1)
                img_features[i] = img_features[i].unsqueeze(1)
                visual_segment_ids = torch.ones(bs, img_features[i].shape[1],dtype=torch.long).to(device)
                v_mask = torch.ones(bs, img_features[i].shape[1],dtype=torch.long).to(device)
                if hidden_fusion is not None:
                    fusion_mask=torch.ones(bs, hidden_fusion.shape[1],dtype=torch.long).to(device)
                else:
                    fusion_mask = None
            
                hidden_fusion = self.shared_block[i](
                    input_ids = all_hidden_states[i], 
                    token_type_ids = data['segment_ids'], 
                    attention_mask = data['text_mask'],
                    visual_feats = img_features[i],
                    visual_token_type_ids=visual_segment_ids,
                    visual_attention_mask=v_mask,
                    fusion_feat=hidden_fusion,
                    fusion_mask=fusion_mask,
                )
                # if hidden_fusion is None:
                #     hidden_fusion = new_hidden_fusion
                # else:
                #     new_hidden_fusion = torch.cat((hidden_fusion, new_hidden_fusion), dim=1)
            # print(hidden_fusion.shape)
            #sdf
            data['memories'] = hidden_fusion         



        elif task_type == 'mmhs':

            
            # method 1
            # img_features = self.image_encoder(data['image'])[-1]
            # img_features = img_features.flatten(1)
            # img_features = self.twitter(img_features)
            # img_features = img_features.unsqueeze(1)
            
            
            # text = data['text']
            # token_type_ids = data['segment_ids']
            # attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            # device = text.device
            # position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            # attention_mask = (1.0 - attention_mask) * -1e5
            
            # text_embedding = self.text_encoder.embeddings(
            #             input_ids=text,
            #             position_ids=position_ids,
            #             token_type_ids=token_type_ids,
            #         )          
            
            # head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            # encoder_outputs = self.text_encoder.encoder(
            #     text_embedding,
            #     attention_mask=attention_mask,
            #     head_mask=head_mask,
            # )  
            # encoder_outputs = encoder_outputs[0] 
            
            # data['memories'] = torch.cat((encoder_outputs, img_features), dim=1)
            
            # method 2 middle fusion
            device = data['text'].device
            bs = data['text'].shape[0]
            

            text = data['text']
            token_type_ids = data['segment_ids']
            attention_mask = data['text_mask'].unsqueeze(1).unsqueeze(2)
            device = text.device
            position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
    
            attention_mask = (1.0 - attention_mask) * -1e5
            
            text_embedding = self.text_encoder.embeddings(
                        input_ids=text,
                        position_ids=position_ids,
                        token_type_ids=token_type_ids,
                    )          
            
            head_mask = [None for _ in range(self.text_encoder.config.num_hidden_layers)]
            
            encoder_outputs = self.text_encoder.encoder(
                text_embedding,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )  
            
            all_hidden_states = encoder_outputs[1]
            img_features = self.image_encoder(data['image'])
            hidden_fusion = None
            for i in range(4):
                img_features[i] = img_features[i].flatten(1)
                img_features[i] = img_features[i].unsqueeze(1)
                visual_segment_ids = torch.ones(bs, img_features[i].shape[1],dtype=torch.long).to(device)
                v_mask = torch.ones(bs, img_features[i].shape[1],dtype=torch.long).to(device)
                if hidden_fusion is not None:
                    fusion_mask=torch.ones(bs, hidden_fusion.shape[1],dtype=torch.long).to(device)
                else:
                    fusion_mask = None
            
                hidden_fusion = self.shared_block[i](
                    input_ids = all_hidden_states[i], 
                    token_type_ids = data['segment_ids'], 
                    attention_mask = data['text_mask'],
                    visual_feats = img_features[i],
                    visual_token_type_ids=visual_segment_ids,
                    visual_attention_mask=v_mask,
                    fusion_feat=hidden_fusion,
                    fusion_mask=fusion_mask,
                )
                # if hidden_fusion is None:
                #     hidden_fusion = new_hidden_fusion
                # else:
                #     new_hidden_fusion = torch.cat((hidden_fusion, new_hidden_fusion), dim=1)
            # print(hidden_fusion.shape)
            #sdf
            data['memories'] = hidden_fusion        

            

   
        return data




