from torch import nn
import pytorch_lightning as pl
import torch
from .unet import Unet
from prodigyopt import Prodigy
from kornia.filters import filter2d
import torch.nn.functional as F
from exp_utils.metric_util import gkern
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .model_extractors import *
#from nem_config import TRAIN_BATCH_SIZE, EPOCHS, MIXED_PRECISION_TRAINING
import math, os
import time
from .unet import Unet3D          
from .model_extractors import DenseNet3DExtractor, UNetEnc3DExtractor 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def subsetoperator(scores, k, tau=1, revert=False, training=False, cumsum_ranking=False,gumbel_inference=False, epoch = 10, samples = 6):
    score_shape = scores.shape
    scores = scores.view(scores.size(0), -1)
    scores = scores  - scores.logsumexp(dim=1).unsqueeze(1)
    if training or gumbel_inference:
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g
    val_sorted, val_indices = torch.sort(scores)     
    
    mask_k = torch.zeros_like(val_sorted)
    if training:
        size = score_shape[-1] * score_shape[-2]
        bin_size = size//samples
        sizes= torch.tensor([int(size - bin_size * i)  for i in range(1, samples + 1)]) 
        bin_add = (torch.rand(len(sizes)) * bin_size)
        k_values = (bin_add + sizes).type(torch.int)
       # k_values = torch.randint(low=1,  high=int(val_sorted.size(1))  , size=( mask_k.size(0),))
        for i in range(mask_k.size(0)):
                j = i//int(mask_k.size(0)/samples)
                # Resetting at the k value
                val_sorted[i] = val_sorted[i] -  val_sorted[i, -k_values[j]]
                mask_k[i, -k_values[j]:] = 1   
    else:
        mask_k[:, -k:] = 1
    # Given we have resetted at the k value, using the sigmoid is approximating a thresholding function
    val_sorted = torch.sigmoid(val_sorted)
    mask = val_sorted - val_sorted.detach()
    
    hardmask = mask + mask_k
    indices = val_indices + (torch.arange(scores.size(0)).view(-1, 1) * scores.size(1)).to(val_sorted.device)
    
    new_mask_soft = torch.zeros_like(val_sorted,dtype=val_sorted.dtype)
    new_mask_soft.view(-1)[indices.view(-1)] = val_sorted.view(-1)
    soft_mask = new_mask_soft.view(score_shape)
    
    new_mask_hard = torch.zeros_like(val_sorted)
    new_mask_hard.view(-1)[indices.view(-1)] = hardmask.view(-1)
    hard_mask = new_mask_hard.view(score_shape)
    
        
    return hard_mask, soft_mask

def cosine_mean(x1,x2 ):
    return  (1 - torch.mean(F.cosine_similarity(x1, x2, dim=-1)))/2 # 1 is close 0 is far so say 1 - sim


class EQ_dist(torch.nn.Module):
    def __init__(self):
        super(EQ_dist, self).__init__()

    def forward(self, target, pred_mask,pred, sep = False):
        pred = F.softmax(pred, dim=-1)
        pred_mask = F.softmax(pred_mask, dim=-1)
        if sep:
            return (pred_mask[:,target] - pred[:,target] + 1e-8).pow(2)
        return  (pred_mask[:,target] - pred[:,target] + 1e-8).pow(2).mean()


class MAX_dist(torch.nn.Module):
    def __init__(self):
        super(MAX_dist, self).__init__()

    def forward(self, target, pred_mask,pred, sep = False):
        pred_mask = F.softmax(pred_mask, dim=-1)
        if sep:
            return (1 - pred_mask[:,target])
        return  (1 - pred_mask[:,target]).mean()



def max_target_dist(target, pred_mask,pred):
    return (1 - torch.softmax(pred_mask, dim=-1)[:, target]).mean()

def cross_entropy(target, pred_mask,pred):
    return F.cross_entropy(pred_mask, pred.softmax(dim=-1)).mean()


class Masking_Loss(nn.Module):
    def __init__(self,constrastive = False, supervised = False,inversed = False, use_ranking = False):
        super(Masking_Loss, self, ).__init__()
        self.constrastive = constrastive
        self.supervised = supervised
        
        self.n_cor = 1e-6
        self.inverse = inversed
        self.supervised_ratio = 50
        self.unsupervised_ratio = 1
        self.use_ranking = use_ranking

        self.measure_emb = cosine_mean
        self.measure_output = MAX_dist()
        #print("loss using ranking", use_ranking)
        
    def unsupervised_loss(self, masks, true_vector_emb, masked_vector_emb, neg_masked_vector_emb):
        masking_ratio = torch.mean((masks))
        pdist_emb = self.measure_emb(true_vector_emb, masked_vector_emb)
        if self.use_ranking:
            loss = pdist_emb
            if self.inverse:
                loss = 2 - loss
            return loss, pdist_emb, masking_ratio
        
        if self.constrastive:
            ndist_emb = self.measure_emb(masked_vector_emb,neg_masked_vector_emb)
            loss =  masking_ratio + (ndist_emb - 1.5) * pdist_emb
        else: 
            loss = 4* pdist_emb + 1/2 *  masking_ratio

        return   loss, pdist_emb, masking_ratio
    
    def supervised_loss(self, masks,  true_vector_output, masked_vector_output, target, sep = False):
        if target is None:
            target = true_vector_output.argmax(dim=-1)
        
        masking_ratio = torch.mean((masks))
        pdist = self.measure_output(target, masked_vector_output, true_vector_output, sep = sep)
        if self.use_ranking:
            if sep:
                loss = pdist.mean()
            else:
                    loss = pdist
            masking_ratio = torch.mean(torch.sigmoid(masks))
            if self.inverse:
                loss = 1 - loss
            return loss, pdist, masking_ratio
        
               
        loss =  masking_ratio  + self.supervised_ratio * pdist
        
        if self.inverse:
            loss = 2 - loss     
        return loss, pdist, masking_ratio
    
    def forward(self, masks, 
                true_vector_emb, true_vector_output,
                masked_vector_emb, masked_vector_output, 
                neg_masked_vector_emb, neg_masked_vector_output, target = None, sep = False):
        

        if  self.supervised:
            return self.supervised_loss(masks=masks, 
                                         true_vector_output=true_vector_output, 
                                         masked_vector_output=masked_vector_output, 
                                        target = target, sep = sep)
        return self.unsupervised_loss(masks, true_vector_emb, masked_vector_emb, neg_masked_vector_emb)

    
class masking_network(pl.LightningModule):
    def __init__(self, epochs, batch_size,
                   lr = 1,img_size = (224,224), 
                   partition = 1, 
                   noise_mask = False, blur_mask = False, blur = False, constrastive=False, 
                   variational = False , supervised = False, use_real_target = False, 
                   inverse = False, use_random_filter = False, use_reinmax = False, use_ranking = False, use_scale_space_partition = False,
                   cumsum_ranking = False, complex_reduction = None):
        super().__init__()
        assert not (use_reinmax & use_ranking), "Cannot use Reinmax and Ranking at the same time"
        assert not (use_ranking & constrastive), "Cannot use Ranking and Constrastive at the same time"
        assert not (use_ranking & inverse), "Cannot use Ranking and Inverse at the same time"
        
        self.noise_mask = noise_mask
        self.blur_mask = blur_mask
        self.blur_kernel = torch.Tensor(gkern(11,5))
        self.negative = constrastive
        self.patcher = partition is not None
        self.blur = blur
        self.partition = partition
        self.use_scale_space_partition = use_scale_space_partition


        self.samples = 6 if use_ranking else 1
        self.tau_start = 3 if use_ranking else 5
        self.tau_min = 0.1 if use_ranking else 1
        self.tau_reduction = 0.33 if use_ranking else 0.05
        sample_reduction = 1 if use_scale_space_partition or partition == None else partition  
        self.top_k_val = int(0.05 * 224 * 224/sample_reduction)   if use_ranking else 0
        self.topk_revert = False
        self.tau = self.tau_start
        self.rand_kernel_size = 21
        self.cumsum_ranking = cumsum_ranking
        self.complex_reduction = complex_reduction
        self.gumbel_inference = False

        assert complex_reduction in [None, "hist_max", "hist_last_peak", "mean","comb","trimmed_mean","trim_hist_max", "gumbel_95" ], "Complex reduction not implemented"

        
        self.supervised = supervised
        self.use_reinmax = use_reinmax
        self.use_ranking = use_ranking
        self.supervised_use_real_target = use_real_target & supervised

        self.use_random_filter = use_random_filter    
        self.loss_func       = Masking_Loss(
            constrastive=constrastive,
            supervised=supervised,
            inversed=inverse,
            use_ranking=use_ranking
            )
        self.learning_rate    =   lr 
        self.epochs = epochs
        
        self.img_size = img_size
        self.batch_size = batch_size
        if self.patcher:
            self.patching_layer = nn.Conv2d(1,  1, kernel_size=partition, stride=partition, padding=0)
        self.masking_network = None
        self.frozen_network  = None
        self.global_pool = None

    def gen_mask(self,x, mask = None):
        # GENERATE MASK LOGITS
        if mask is None:
            mask =  self.masking_network(x)
        if self.patcher and not self.use_scale_space_partition:
            mask = self.patching_layer(mask)  
        
        ### TRANSFORM MASK LOGITS
        if self.supervised & self.use_random_filter:
            rand_kernel = torch.rand(size=(1,self.rand_kernel_size,self.rand_kernel_size))  
            mask = filter2d(mask, rand_kernel, normalized=True,border_type='circular')

        ### CREATE MULTIPLE SAMPLES FOR RANKING
        if self.training and self.use_ranking:
            mask = mask.repeat(self.samples,1,1,1).reshape(-1, *mask.shape[1:])
            x = x.repeat(self.samples,1,1,1).reshape(-1, *x.shape[1:])


        ### CREATE APPLIED MASK

        if self.use_ranking:
            applied_mask, soft_mask = subsetoperator(mask.permute(0,2,3,1).squeeze(3),
                                              self.top_k_val, tau=self.tau,
                                              training=self.training, 
                                              revert=self.topk_revert,gumbel_inference=self.gumbel_inference,
                                              cumsum_ranking=self.cumsum_ranking,epoch=self.current_epoch, samples=self.samples)
            applied_mask = applied_mask.unsqueeze(3).permute(0,3,1,2)
        else:
            mask = torch.sigmoid(mask)
            applied_mask = mask

          

        ### APPLY MASK
        if self.noise_mask:
            device = next(self.masking_network.parameters()).device
            noise = torch.normal(0,1,size=x.shape).to( device)
            data_mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device =  device)[None, :, None, None]
            data_std = torch.tensor(IMAGENET_DEFAULT_STD, device =  device)[None, :, None, None]
            perturbation = noise*data_std+data_mean
        elif self.blur:
                perturbation =  torch.nn.functional.conv2d(x, self.blur_kernel.type(x.dtype).to(x.device), padding=11//2)
        else:
            perturbation = 0 * x

        x_masked =  x * applied_mask + (1 - applied_mask) * perturbation
        if self.negative:
            self.negative_img = x * (1 - applied_mask) + (applied_mask) * perturbation
        return mask, x_masked, applied_mask
    
    def gen_representations(self,x, x_masked, pred_emb = None, pred = None):
        if pred_emb is None:
            pred_emb =  self.frozen_network.get_embeddings(x)
        pred_masked_emb =  self.frozen_network.get_embeddings(x_masked)
        pred_neg_emb = self.frozen_network.get_embeddings(self.negative_img) if self.negative else None
        if self.supervised:
            if pred is None:
                pred = self.frozen_network.get_output_from_embeddings(pred_emb)        
            pred_neg = self.frozen_network.get_output_from_embeddings(pred_neg_emb) if self.negative else None
            pred_masked = self.frozen_network.get_output_from_embeddings(pred_masked_emb)
        else:
            pred = pred_emb
            pred_neg = pred_neg_emb
            pred_masked = pred_masked_emb
            
        if self.training and self.use_ranking:
            if self.supervised:
                pred = pred.repeat(self.samples,1,1).reshape(-1, *pred.shape[1:])
            else:
                pred_emb = pred
                pred_masked_emb = pred_masked
                pred_emb = pred_emb.repeat(self.samples,1,1).reshape(-1, *pred_emb.shape[1:])
            

        return pred_emb, pred, pred_masked_emb, pred_masked , pred_neg_emb, pred_neg
    
    def run_step(self,x,target = None):
        mask, x_masked, applied_mask = self.gen_mask(x)
        pred_emb, pred, pred_masked_emb, pred_masked,pred_neg_emb,  pred_neg = self.gen_representations(x=x,x_masked=x_masked)
        loss, pdist, masking_ratio  = self.loss_func(
            mask, pred_emb, pred, pred_masked_emb, pred_masked, pred_neg_emb, pred_neg, target = target)
        return loss, pdist, masking_ratio, pred, pred_masked 

    def training_step(self, batch, batch_idx):
        if self.use_ranking:
            masking_choices = ["remove", "blur", "noise"]
            masking = masking_choices[torch.randint(0,3,(1,)).item()]
            self.noise_mask = masking == "noise"
            self.blur = masking == "blur"

        if len(batch) == 2:
            x, y = batch
        else:
            x = batch

        if self.masking_network.freeze_backbone:
            self.masking_network.encoder.eval()
        
        if self.use_ranking and self.training and self.supervised:
            y = y.repeat(self.samples,1,1,1).reshape(-1, *y.shape[1:])

        target= y if self.supervised_use_real_target else None
        loss, pdist, masking_ratio, _, _ = self.run_step(x,target=target)
        self.log("train_loss", loss, batch_size=self.batch_size)
        self.log("train_mask_norm",masking_ratio, batch_size=self.batch_size)
        self.log("train_dist",pdist, batch_size=self.batch_size)

        return loss 
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
        loss, pdist, masking_ratio, _ , _ = self.run_step(x,target= y if self.supervised else None)
        self.log("val_loss", loss, batch_size=self.batch_size)
        self.log("val_mask_norm",masking_ratio, batch_size=self.batch_size)
        self.log("val_dist",pdist, batch_size=self.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, pdist, masking_ratio, _ , _ = self.run_step(x,target= y if self.supervised else None)
        self.log("test_loss", loss, batch_size=self.batch_size)
        self.log("test_mask_norm",masking_ratio, batch_size=self.batch_size)
        self.log("test_dist",pdist, batch_size=self.batch_size)
        return loss
    
    def forward(self, x):
        return self.gen_mask(x)
    
    def configure_optimizers(self):
    
        optim_spec = lambda params: Prodigy(
                params, weight_decay=1e-4)
        
        # Only optimize decoder if backbone is frozen
        if self.masking_network.freeze_backbone:
            if self.patcher:
                optimizer =  optim_spec(
                    list(self.masking_network.decoder.parameters()) + 
                    list(self.patching_layer.parameters())
                    )                      
            else:
                optimizer =  optim_spec(self.masking_network.decoder.parameters())
        else:
            optimizer =  optim_spec(self.masking_network.parameters())
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-6,),
            'interval': 'epoch',  # Adjust learning rate every epoch
            'frequency': 1,       # How often to apply the scheduler
        }
        return [optimizer], [scheduler]


class resnet50_nem(masking_network):
    def __init__(self, explained_model,  epochs, batch_size, lr=1., center=False,
                 partition=None, noise_mask=False, constrastive=False, 
                 inverse=False, use_random_filter=False, supervised=True, 
                 use_real_target=False, use_reinmax=False, use_ranking=False, use_scale_space_partition=False, cumsum_ranking=False):
        
        super().__init__(epochs=epochs, lr=lr, batch_size=batch_size, 
                         partition=partition, constrastive=constrastive, 
                         noise_mask=noise_mask, supervised=supervised, 
                         inverse=inverse, use_random_filter=use_random_filter, 
                         use_real_target=use_real_target, use_reinmax=use_reinmax, 
                         use_ranking=use_ranking, use_scale_space_partition=use_scale_space_partition, cumsum_ranking=cumsum_ranking) 
        
        backbone = resnet50_img_extractor(explained_model)
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32, 16)
        self.masking_network = Unet(
            num_classes=1, backbone=backbone, 
            freeze_backbone=True, center=center,encoder_channels=encoder_channels,
            decoder_channels=decoder_channels, use_img_space=backbone.use_img_space)
        self.frozen_network = backbone
        #print(f"Running {supervised} resnet50")


class vgg16_nem(masking_network):
    def __init__(self, explained_model,  epochs, batch_size, lr=1., center=False,
                 partition=None, noise_mask=False, constrastive=False, 
                 inverse=False, use_random_filter=False, supervised=True, 
                 use_real_target=False, use_reinmax=False, use_ranking=False, use_scale_space_partition=False, cumsum_ranking=False):
        
        super().__init__(epochs=epochs, lr=lr, batch_size=batch_size, 
                         partition=partition, constrastive=constrastive, 
                         noise_mask=noise_mask, supervised=supervised, 
                         inverse=inverse, use_random_filter=use_random_filter, 
                         use_real_target=use_real_target, use_reinmax=use_reinmax, 
                         use_ranking=use_ranking, use_scale_space_partition=use_scale_space_partition, cumsum_ranking=cumsum_ranking) 
        
        backbone = vgg_img_extractor(explained_model)
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32, 16)
        self.masking_network = Unet(
            num_classes=1, backbone=backbone, 
            freeze_backbone=True, center=center,encoder_channels=encoder_channels,
            decoder_channels=decoder_channels, use_img_space=backbone.use_img_space)
        self.frozen_network = backbone
        #print(f"Running {supervised} vgg16")

class convnext_nem(masking_network):
    def __init__(self, explained_model,  epochs, batch_size, lr=1., center=False,
                 partition=None, noise_mask=False, constrastive=False, 
                 inverse=False, use_random_filter=False, supervised=True, 
                 use_real_target=False, use_reinmax=False, use_ranking=False, use_scale_space_partition=False, cumsum_ranking=False):
        
        super().__init__(epochs=epochs, lr=lr, batch_size=batch_size, 
                         partition=partition, constrastive=constrastive, 
                         noise_mask=noise_mask, supervised=supervised, 
                         inverse=inverse, use_random_filter=use_random_filter, 
                         use_real_target=use_real_target, use_reinmax=use_reinmax, 
                         use_ranking=use_ranking, use_scale_space_partition=use_scale_space_partition, cumsum_ranking=cumsum_ranking) 
        
        backbone = convnext_extractor(explained_model)
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32)
        self.masking_network = Unet(
            num_classes=1, backbone=backbone, 
            freeze_backbone=True, center=center,encoder_channels=encoder_channels,
            decoder_channels=decoder_channels, use_img_space=backbone.use_img_space)
        self.frozen_network = backbone
        #print(f"Running {supervised} convnext")

class vit_nem(masking_network):
    def __init__(self, explained_model,  epochs, batch_size, lr=1., center=False,
                 partition=None, noise_mask=False, constrastive=False, 
                 inverse=False, use_random_filter=False, supervised=True, 
                 use_real_target=False, use_reinmax=False, use_ranking=False, use_scale_space_partition=False, cumsum_ranking=False):
        
        super().__init__(epochs=epochs, lr=lr, batch_size=batch_size, 
                         partition=partition, constrastive=constrastive, 
                         noise_mask=noise_mask, supervised=supervised, 
                         inverse=inverse, use_random_filter=use_random_filter, 
                         use_real_target=use_real_target, use_reinmax=use_reinmax, 
                         use_ranking=use_ranking, use_scale_space_partition=use_scale_space_partition, cumsum_ranking=cumsum_ranking) 
        
        backbone = vit_feature_extractor(explained_model)
        self.reshaper = backbone.output_shape
        encoder_channels = backbone.channels
        decoder_channels = (256, 128, 64, 32, 16)
        self.masking_network = Unet(
            num_classes=1, backbone=backbone, 
            freeze_backbone=True, center=center,encoder_channels=encoder_channels,
            decoder_channels=decoder_channels, use_img_space=backbone.use_img_space)
        self.frozen_network = backbone
        #print(f"Running {supervised} vit")

def select_nem(model):
    if model.__class__.__name__ == "ResNet":
        return resnet50_nem
    elif model.__class__.__name__ == "VGG":
        return vgg16_nem
    elif model.__class__.__name__ == "ConvNeXt":
        return convnext_nem
    elif model.__class__.__name__ == "VisionTransformer":
        return vit_nem
    else:
        raise ValueError(f"Unsupported model {model.__class__.__name__}")



def train_nem(nem, data,log_path = None):
    trainer = pl.Trainer(
            max_epochs=EPOCHS,
            devices="auto",
            accelerator="gpu" if torch.cuda.is_available() else "cpu", 
            precision=16 if MIXED_PRECISION_TRAINING else 32,
            default_root_dir=log_path,)
    trainer.fit(nem, data)
        


def load_ranem(model,train_data):
    
    nem_constructor = select_nem(model)
    log_path = f"attrs/nem_utils/logs/ranem/{model.__class__.__name__}"
    if os.path.exists(log_path):
         checkpoint_path = f"{log_path}/lightning_logs/version_0/checkpoints/"
         checkpoint_path = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
         ranem = nem_constructor.load_from_checkpoint(
            checkpoint_path, epochs = EPOCHS, batch_size=TRAIN_BATCH_SIZE, explained_model=model,use_ranking =True,strict=False)
    else:
        ranem = nem_constructor(model, EPOCHS, batch_size=TRAIN_BATCH_SIZE, use_ranking=True)
        start_time = time.time()
        print("Training")
        print(f"Storing data at {log_path}")
        train_nem(ranem, train_data, log_path)
        total_time = time.time() - start_time
        print(f"Training time {total_time}")
        np.save(f"{log_path}/training_time.npy", total_time)
    return ranem


def load_nemt(model,train_data):
    
    nem_constructor = select_nem(model)
    log_path = f"attrs/nem_utils/logs/nemt/{model.__class__.__name__}"
    if os.path.exists(log_path):
         checkpoint_path = f"{log_path}/lightning_logs/version_0/checkpoints/"
         checkpoint_path = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
         nemt = nem_constructor.load_from_checkpoint(
            checkpoint_path, epochs = EPOCHS, batch_size=TRAIN_BATCH_SIZE, explained_model=model,inverse=True,use_random_filter=True,strict=False)
    else:
        nemt = nem_constructor(model, EPOCHS, batch_size=TRAIN_BATCH_SIZE, inverse=True, use_random_filter=True)
        start_time = time.time()
        print("Training")
        print(f"Storing data at {log_path}")
        train_nem(nemt, train_data, log_path)
        total_time = time.time() - start_time
        print(f"Training time {total_time}")
        np.save(f"{log_path}/training_time.npy", total_time)
    return nemt

# 3D NEM (no ranking, no noise/blur)
def _limit_batches_env(key, default=1.0):
    """
    NEM_LIMIT_TRAIN / NEM_LIMIT_VAL helper:
      - int  -> that many batches
      - float in (0,1] -> fraction of epoch
    """
    v = os.environ.get(key, "")
    if not v:
        return default
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return default


def _fast_dev_run_env():
    return os.environ.get("NEM_FAST_DEV", "").lower() in ("1", "true", "yes")

def make_gaussian_kernel_3d(kernel_size=5, sigma=1.0, device="cpu", dtype=torch.float32):
    """
    Returns a 3D Gaussian kernel of shape [1, 1, k, k, k] normalized to sum to 1.
    """
    assert kernel_size % 2 == 1, "kernel_size should be odd."
    k = kernel_size
    coords = torch.arange(k, dtype=dtype, device=device) - (k - 1) / 2.0
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    g = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    g /= g.sum()
    return g.view(1, 1, k, k, k)


def _maybe_logger(log_path: str):
    """
    If NEM_TBLOG=1, create a TensorBoard logger under <log_path>/tb.
    Otherwise disable Lightning logging.
    """
    if os.environ.get("NEM_TBLOG", "").lower() in ("1", "true", "yes"):
        return TensorBoardLogger(
            save_dir=os.path.join(log_path, "tb"),
            name="lightning_logs",
        )
    return False

class MaskingLoss3DNEMT(nn.Module):
    """
    3D NEMT-style loss with explicit semantics:

      - masks: keep probabilities in [0,1], shape [B,1,D,H,W]
      - x_masked = x * mask + baseline * (1 - mask)
      - We want:
          * mean(mask) near keep_target (e.g. 0.3)
          * large drop in true-class probability when masked

      Uses:
        keep_target  from NEM_KEEP_TARGET (default 0.3)
        lambda_del   from NEMT_LAMBDA    (default 10.0)
    """
    def __init__(self, keep_target: float | None = None, lambda_del: float | None = None):
        super().__init__()
        kt = 0.3 if keep_target is None else float(keep_target)
        ld = 10.0 if lambda_del is None else float(lambda_del)

        self.keep_target = float(os.environ.get("NEM_KEEP_TARGET", kt))
        self.lambda_del = float(os.environ.get("NEMT_LAMBDA", ld))

    def forward(self, masks, true_logits, masked_logits, target=None):
        # flatten mask to mean keep ratio
        if masks.dim() == 5:
            m = masks.flatten(1).mean(dim=1)       # [B]
        else:
            m = masks.view(masks.size(0), -1).mean(dim=1)

        # logits to shape [B]
        true_logits = true_logits.view(-1)
        masked_logits = masked_logits.view(-1)

        # choose targets
        if target is None:
            # default: use model's own prediction as target
            with torch.no_grad():
                p = torch.sigmoid(true_logits)
                target = (p >= 0.5).long()
        else:
            target = target.view(-1).long()

        # true-class probabilities
        p_orig = torch.sigmoid(true_logits)
        p_mask = torch.sigmoid(masked_logits)

        p_true_orig = torch.where(target == 1, p_orig, 1.0 - p_orig)
        p_true_mask = torch.where(target == 1, p_mask, 1.0 - p_mask)

        # deletion effect
        del_effect = (p_true_orig - p_true_mask).clamp(min=0.0)  # [B], want large

        # keep term
        keep_term = (m - self.keep_target).pow(2).mean()

        #deletion term
        del_term = - self.lambda_del * del_effect.mean()

        loss = keep_term + del_term

        # for logging
        return loss, del_effect.mean(), m.mean()

class masking_network_3d(pl.LightningModule):
    """
    Simple 3D variant:
      - ignores ranking
      - ignores noise/blur
      - optimized with 3D NEMT deletion-based loss (mask = keep)
    """
    def __init__(self, epochs, batch_size, lr=1.0, supervised=True, inverse=True):
        super().__init__()
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.supervised = supervised
        self.inverse = inverse
        self.masking_network = None     # set by subclass
        self.frozen_network = None      # set by subclass
        self.loss_func = MaskingLoss3DNEMT()

        smooth = os.environ.get("NEM_SMOOTH_MASK", "1").lower() in ("1", "true", "yes")
        self.smooth_masks = smooth
        if self.smooth_masks:
            # register as buffer so it moves with the model
            ksize = int(os.environ.get("NEM_SMOOTH_K", "5"))
            sigma = float(os.environ.get("NEM_SMOOTH_SIGMA", "1.0"))
            kernel = make_gaussian_kernel_3d(ksize, sigma)
            self.register_buffer("gauss_kernel_3d", kernel)

    def _unpack_batch(self, batch):
        """
        Normalize different batch structures into (x, y) where:
          - x: torch.Tensor [B, 1, D, H, W] (or [1, D, H, W] for B=1)
          - y: torch.Tensor [B] or None

        Handles:
          * (x, y) where x, y are already batched tensors
          * [x, y] where x, y may be tensors OR lists/tuples of samples
          * dict with 'image' / 'label' keys (MONAI style)
          * list of dicts with 'image' and optional 'label'
          * list of tensors (x only, no labels)
        """
        import torch

        # MONAI-style dict batch
        if isinstance(batch, dict):
            x = batch.get("image", None)
            y = batch.get("label", None)
            return x, y

        if isinstance(batch, (list, tuple)):
            # treat as (x, y)
            if len(batch) == 2:
                x, y = batch

                if isinstance(x, (list, tuple)):
                    if len(x) == 0:
                        raise RuntimeError("Batch 'x' is an empty list/tuple.")
                    if isinstance(x[0], torch.Tensor):
                        x = torch.stack(x, dim=0)
                    else:
                        x = torch.as_tensor(x)
                if isinstance(y, (list, tuple)):
                    if len(y) == 0:
                        y = None
                    elif isinstance(y[0], torch.Tensor):
                        y = torch.stack(y, dim=0)
                    else:
                        y = torch.as_tensor(y)

                return x, y

            # list of dicts
            if len(batch) > 0 and isinstance(batch[0], dict) and "image" in batch[0]:
                xs = [b["image"] for b in batch]

                if isinstance(xs[0], str):
                    raise RuntimeError(
                        "Batch 'image' entries are strings (filenames). "
                        "This suggests MONAI transform wasn't applied. "
                        "Check that LunaCandidates3DDataset(use_monai_pipeline=True, "
                        "transform=build_val_tf(...)) is used."
                    )

                if isinstance(xs[0], torch.Tensor):
                    x = torch.stack(xs, dim=0)
                else:
                    x = torch.as_tensor(xs)

                if "label" in batch[0]:
                    labs = [b["label"] for b in batch]
                    if isinstance(labs[0], torch.Tensor):
                        y = torch.stack(labs, dim=0)
                    else:
                        y = torch.as_tensor(labs)
                else:
                    y = None

                return x, y

            # list of tensors only -> inputs without labels
            if len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                x = torch.stack(batch, dim=0)
                y = None
                return x, y

        # already a tensor
        if torch.is_tensor(batch):
            return batch, None
        raise RuntimeError(f"_unpack_batch: unsupported batch type {type(batch)}")

    # core NEMT3D
    def gen_mask(self, x):
        mask_logits = self.masking_network(x)         # [B,1,D',H',W']

        # smooth mask logits with 3D Gaussian
        if getattr(self, "smooth_masks", False):
            k = self.gauss_kernel_3d.to(mask_logits.dtype)
            pad = k.shape[-1] // 2
            mask_logits = F.conv3d(mask_logits, k, padding=pad)

        amask = torch.sigmoid(mask_logits)            # keep ∈ [0,1]
        if amask.shape[-3:] != x.shape[-3:]:
            amask = F.interpolate(amask, size=x.shape[-3:], mode="trilinear", align_corners=False)
        amask = torch.nan_to_num(amask, nan=0.0, posinf=1.0, neginf=0.0)

        baseline = x.mean(dim=(2, 3, 4), keepdim=True)
        x_masked = x * amask + (1.0 - amask) * baseline
        return mask_logits, x_masked, amask


    def gen_representations(self, x, x_masked):
        pred_emb = self.frozen_network.get_embeddings(x)
        pred_masked_emb = self.frozen_network.get_embeddings(x_masked)
        if self.supervised:
            pred = self.frozen_network.get_output_from_embeddings(pred_emb)
            pred_masked = self.frozen_network.get_output_from_embeddings(pred_masked_emb)
            pred_neg_emb = None; pred_neg = None
        else:
            pred = pred_emb; pred_masked = pred_masked_emb
            pred_neg_emb = None; pred_neg = None
        return pred_emb, pred, pred_masked_emb, pred_masked, pred_neg_emb, pred_neg

    def run_step(self, x, target=None):
        # 1) generate mask + masked input
        mask_logits, x_masked, applied_mask = self.gen_mask(x)

        # 2) get logits from frozen network
        pred_emb, pred, pred_masked_emb, pred_masked, pred_neg_emb, pred_neg = \
            self.gen_representations(x, x_masked)

        # pred, pred_masked are logits [B,1]
        loss, pdist, masking_ratio = self.loss_func(
            applied_mask,
            true_logits=pred,
            masked_logits=pred_masked,
            target=target,
        )

        # return the mask to log its stats
        return loss, pdist, masking_ratio, pred, pred_masked, applied_mask


    # Lightning steps
    def training_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)

        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"Expected x to be a Tensor in training_step, got {type(x)}")

        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, device=x.device)

        # label handling
        pos_only = os.environ.get("NEM_POS_ONLY", "1").lower() in ("1", "true", "yes")
        max_ratio_str = os.environ.get("NEM_MAX_NEG_POS_RATIO", "").strip()
        max_ratio = float(max_ratio_str) if max_ratio_str not in ("", "None") else 0.0

        if y is not None:
            y = y.view(-1).long()
            pos_idx = (y == 1)
            neg_idx = (y == 0)

            if pos_only:
                # use positives only, skip all-neg batches
                if pos_idx.any():
                    x = x[pos_idx]
                    y = y[pos_idx]
                else:
                    skip_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                    self.log("train_skip_all_neg", 1.0, on_step=True, batch_size=self.batch_size)
                    return skip_loss
            else:
                # use positives + negatives, but optionally cap neg:pos ratio
                n_pos = int(pos_idx.sum().item())
                n_neg = int(neg_idx.sum().item())

                if max_ratio > 0 and n_pos > 0 and n_neg > 0:
                    max_neg = int(max_ratio * n_pos)
                    if n_neg > max_neg:
                        # randomly choose a subset of negatives to keep
                        neg_indices = neg_idx.nonzero(as_tuple=False).view(-1)
                        perm = torch.randperm(neg_indices.numel(), device=y.device)
                        keep_neg = neg_indices[perm[:max_neg]]

                        keep_mask = pos_idx.clone()
                        keep_mask[keep_neg] = True

                        x = x[keep_mask]
                        y = y[keep_mask]

        tgt = y if y is not None else None

        # core NEM step
        loss, pdist, masking_ratio, pred, pred_masked, amask = self.run_step(x, target=tgt)

        # Convert logits to probabilities for logging
        p_orig = torch.sigmoid(pred.view(-1)).mean()
        p_mask = torch.sigmoid(pred_masked.view(-1)).mean()

        self.log("train_loss",        loss,          batch_size=x.size(0))
        self.log("train_keep_ratio",  masking_ratio, batch_size=x.size(0))   # complexity
        self.log("train_del_effect",  pdist,         batch_size=x.size(0))   # “accuracy”
        self.log("train_p_orig",      p_orig,        batch_size=x.size(0))
        self.log("train_p_mask",      p_mask,        batch_size=x.size(0))

        # mask stats
        am = amask.detach()
        self.log("train_mask_min",  am.min(),  batch_size=x.size(0))
        self.log("train_mask_max",  am.max(),  batch_size=x.size(0))
        self.log("train_mask_mean", am.mean(), batch_size=x.size(0))
        self.log("train_mask_std",  am.std(),  batch_size=x.size(0))

        if os.environ.get("NEM_DEBUG", "0") == "1" and self.global_step % 500 == 0:
            print(
                f"[DBG step={self.global_step}] "
                f"keep={masking_ratio.item():.3f}, "
                f"del={pdist.item():.3f}, "
                f"p_orig={p_orig.item():.3f}, p_mask={p_mask.item():.3f}, "
                f"mask[min={am.min().item():.3f}, max={am.max().item():.3f}, mean={am.mean().item():.3f}]"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)

        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"Expected x to be a Tensor in validation_step, got {type(x)}")

        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, device=x.device)

        tgt = y if (self.supervised and y is not None) else None

        loss, pdist, masking_ratio, pred, pred_masked, amask = self.run_step(x, target=tgt)

        p_orig = torch.sigmoid(pred.view(-1)).mean()
        p_mask = torch.sigmoid(pred_masked.view(-1)).mean()

        self.log("val_loss",        loss,          batch_size=x.size(0))
        self.log("val_keep_ratio",  masking_ratio, batch_size=x.size(0))
        self.log("val_del_effect",  pdist,         batch_size=x.size(0))
        self.log("val_p_orig",      p_orig,        batch_size=x.size(0))
        self.log("val_p_mask",      p_mask,        batch_size=x.size(0))

        am = amask.detach()
        self.log("val_mask_mean", am.mean(), batch_size=x.size(0))
        self.log("val_mask_std",  am.std(),  batch_size=x.size(0))

        return loss

    def configure_optimizers(self):
        optim = Prodigy(self.masking_network.parameters(), weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20, eta_min=1e-6)
        return [optim], [{"scheduler": sched, "interval": "epoch", "frequency": 1}]


class densenet3d_nem(masking_network_3d):
    def __init__(self, explained_model, epochs, batch_size, lr=1.0, supervised=True, inverse=True):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, supervised=supervised, inverse=inverse)
        backbone = DenseNet3DExtractor(explained_model)
        self.masking_network = Unet3D(
            backbone=backbone, freeze_backbone=True, center=False,
            encoder_channels=backbone.channels, decoder_channels=(256,128,64,32), num_classes=1
        )
        self.frozen_network = backbone

        # warm-start mask logits so applied_mask ≈ keep_target
        KEEP_TARGET = float(os.environ.get("NEM_KEEP_TARGET", "0.3"))
        with torch.no_grad():
            final = None
            if hasattr(self.masking_network, "final_conv"):
                final = self.masking_network.final_conv
            elif hasattr(self.masking_network, "decoder") and hasattr(self.masking_network.decoder, "final_conv"):
                final = self.masking_network.decoder.final_conv
            if final is not None and getattr(final, "bias", None) is not None:
                KEEP_TARGET_CLAMP = min(max(KEEP_TARGET, 1e-6), 1.0 - 1e-6)
                bias = math.log(KEEP_TARGET_CLAMP / (1.0 - KEEP_TARGET_CLAMP))
                final.bias.fill_(bias)



class unetenc3d_nem(masking_network_3d):
    def __init__(self, explained_model, epochs, batch_size, lr=1.0, supervised=True, inverse=True):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, supervised=supervised, inverse=inverse)
        backbone = UNetEnc3DExtractor(explained_model)
        self.masking_network = Unet3D(
            backbone=backbone, freeze_backbone=True, center=False,
            encoder_channels=backbone.channels, decoder_channels=(256,128,64,32), num_classes=1
        )
        self.frozen_network = backbone

        # warm-start mask logits so applied_mask ≈ keep_target
        KEEP_TARGET = float(os.environ.get("NEM_KEEP_TARGET", "0.3"))
        with torch.no_grad():
            final = None
            if hasattr(self.masking_network, "final_conv"):
                final = self.masking_network.final_conv
            elif hasattr(self.masking_network, "decoder") and hasattr(self.masking_network.decoder, "final_conv"):
                final = self.masking_network.decoder.final_conv
            if final is not None and getattr(final, "bias", None) is not None:
                KEEP_TARGET_CLAMP = min(max(KEEP_TARGET, 1e-6), 1.0 - 1e-6)
                bias = math.log(KEEP_TARGET_CLAMP / (1.0 - KEEP_TARGET_CLAMP))
                final.bias.fill_(bias)

def _pick_3d_nem(model: nn.Module):
    name = model.__class__.__name__.lower()
    if "densenet" in name:
        return densenet3d_nem
    return unetenc3d_nem

def train_nem_3d(nem, data, log_path=None):
    os.makedirs(log_path, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=log_path,
        filename="last",           # writes <log_path>/last.ckpt
        save_last=True,
        save_top_k=0,              
        every_n_epochs=1
    )
    trainer = pl.Trainer(
        max_epochs=int(os.environ.get("NEM_EPOCHS", EPOCHS)),
        devices="auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=32,
        default_root_dir=log_path,
        limit_train_batches=_limit_batches_env("NEM_LIMIT_TRAIN", 1.0),
        num_sanity_val_steps=0,
        log_every_n_steps=int(os.environ.get("NEM_LOG_EVERY", "50")),
        fast_dev_run=_fast_dev_run_env(),
        callbacks=[ckpt_cb],
        accumulate_grad_batches=int(os.environ.get("NEM_ACCUM", "4")),
        gradient_clip_val=float(os.environ.get("NEM_CLIP", "1.0")),
        enable_checkpointing=True,
        logger=_maybe_logger(log_path),
    )
    trainer.fit(nem, train_dataloaders=data)

    with open(os.path.join(log_path, "best_ckpt.txt"), "w") as f:
        f.write(os.path.join(log_path, "last.ckpt"))

def is_3d_model(model: nn.Module) -> bool:
    return any(isinstance(m, nn.Conv3d) for m in model.modules())

def load_nemt_auto(model, train_data):
    if is_3d_model(model):
        return load_nemt_3d(model, train_data)  
    return load_nemt(model, train_data)

def _find_latest_checkpoint(log_path: str):
    import glob, os
    patt = os.path.join(log_path, "lightning_logs", "version_*", "checkpoints", "*.ckpt")
    files = glob.glob(patt)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def load_nemt_3d(model, train_data):
    """3D NEMT with single-folder 'last.ckpt' and env switches."""
    if not is_3d_model(model):
        raise ValueError("load_nemt_3d was called with a non-3D model.")

    nem_ctor = _pick_3d_nem(model)
    log_path = f"attrs/nem_utils/logs/nemt3d/{model.__class__.__name__}"
    os.makedirs(log_path, exist_ok=True)

    # Allow env to override epochs used for training / logging
    epochs_cfg = int(os.environ.get("NEM_EPOCHS", EPOCHS))

    use_ckpt    = os.environ.get("NEM_USE_CKPT", "1").lower() in ("1", "true", "yes")
    force_train = os.environ.get("NEM_FORCE_TRAIN", "0").lower() in ("1", "true", "yes")
    ckpt_path   = os.path.join(log_path, "last.ckpt")

    if use_ckpt and (not force_train) and os.path.isfile(ckpt_path):
        # load existing nem
        nemt3d = nem_ctor.load_from_checkpoint(
            ckpt_path,
            epochs=epochs_cfg,
            batch_size=TRAIN_BATCH_SIZE,
            explained_model=model,
            supervised=True,
            inverse=True,
            strict=False,
        )
        print(f"[load_nemt_3d] Loaded NEM checkpoint: {ckpt_path}")
    else:
        # train new instead
        nemt3d = nem_ctor(
            explained_model=model,
            epochs=epochs_cfg,
            batch_size=TRAIN_BATCH_SIZE,
            supervised=True,
            inverse=True,
        )
        print("[load_nemt_3d] Training (3D NEMT)")
        print(f"  -> storing single-folder artifacts at: {log_path} (last.ckpt)")
        t0 = time.time()
        train_nem_3d(nemt3d, train_data, log_path)
        dt = time.time() - t0
        print(f"[load_nemt_3d] Training time {dt:.1f} seconds")
        np.save(os.path.join(log_path, "training_time.npy"), dt)

    return nemt3d
