#!/usr/bin/env python
""
Merged main.py for Muse 2 Symbiotic AI System + Lightning Training/Inference
""

import argparse, os, sys, time, datetime, glob, importlib, csv, random, logging
import multiprocessing, threading
import torch
import torchvision
import pytorch_lightning as pil
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from functools import partial
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

# -----------------------------------------------------------------------------
# Muse imports
# -----------------------------------------------------------------------------
from core.interaction_manager import InteractionManager
from ai_interface.ai_program_interface import AIProgramInterface, NetworkedAIProgramInterface
from muse.muse_interface import MuseInterface
from configs.config import Config
from utils.logging_utils import setup_logging
from modules.verbal_communication import VerbalCommunicationModule
from modules.web_access import WebAccessModule
from modules.memory import MemoryModule
from modules.self_awareness import SelfAwarenessModule
from modules.self_improvement import SelfImprovementModule
from modules.metamorphic_engine import MetamorphicEngine
from modules.data_loader import DataLoaderModule
from modules.network_integration import NetworkIntegrationModule

# -----------------------------------------------------------------------------
# Lightning imports
# -----------------------------------------------------------------------------
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

# -----------------------------------------------------------------------------
# Shared Helpers for Lightning side
# -----------------------------------------------------------------------------
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ("yes","true","t","y","1"): return True
        if v.lower() in ("no","false","f","n","0"): return False
        raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n","--name",type=str,const=True,default="",nargs="?",help="postfix for logdir")
    parser.add_argument("-r","--resume",type=str,const=True,default="",nargs="?",help="resume from logdir or checkpoint")
    parser.add_argument("-b","--base",nargs="*",metavar="base_config.yaml",default=list(),
                        help="paths to base configs, left-to-right")
    parser.add_argument("-t","--train",type=str2bool,const=True,default=False,nargs="?",help="train")
    parser.add_argument("--no-test",type=str2bool,const=True,default=False,nargs="?",help="disable test")
    parser.add_argument("-p","--project",help="name of new or path to existing project")
    parser.add_argument("-d","--debug",type=str2bool,const=True,default=False,nargs="?",help="enable debugging")
    parser.add_argument("-s","--seed",type=int,default=23,help="seed for seed_everything")
    parser.add_argument("-f","--postfix",type=str,default="",help="post-postfix for default name")
    parser.add_argument("-l","--logdir",type=str,default="logs",help="directory for logging")
    parser.add_argument("--scale_lr",type=str2bool,const=True,default=True,nargs="?",help="scale lr by ngpu*batch*accum")
    return parser

def nondefault_trainer_args(opt):
    p = argparse.ArgumentParser(); p = Trainer.add_argparse_args(p)
    default = p.parse_args([])
    return sorted(k for k in vars(default) if getattr(opt,k)!=getattr(default,k))

class WrappedDataset(Dataset):
    def __init__(self,dataset): self.data=dataset
    def __len__(self): return len(self.data)
    def __getitem__(self,idx): return self.data[idx]

def worker_init_fn(_):
    info = torch.utils.data.get_worker_info()
    ds, wid = info.dataset, info.id
    if isinstance(ds, Txt2ImgIterableBaseDataset):
        sz = ds.num_records//info.num_workers
        ds.sample_ids=ds.valid_ids[wid*sz:(wid+1)*sz]
        return np.random.seed(np.random.get_state()[1][np.random.choice(len(np.random.get_state()[1]))] + wid)
    return np.random.seed(np.random.get_state()[1][0]+wid)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self,batch_size,train=None,validation=None,test=None,predict=None,
                 wrap=False,num_workers=None,shuffle_test_loader=False,
                 use_worker_init_fn=False,shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size=batch_size
        self.dataset_configs={}
        self.num_workers = num_workers or batch_size*2
        self.use_worker_init_fn = use_worker_init_fn
        if train:      self.dataset_configs["train"]=train;       self.train_dataloader=self._train_dataloader
        if validation: self.dataset_configs["validation"]=validation; self.val_dataloader=partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test:       self.dataset_configs["test"]=test;         self.test_dataloader=partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict:    self.dataset_configs["predict"]=predict;   self.predict_dataloader=self._predict_dataloader
        self.wrap=wrap

    def prepare_data(self):
        for cfg in self.dataset_configs.values(): instantiate_from_config(cfg)

    def setup(self,stage=None):
        self.datasets={k:instantiate_from_config(self.dataset_configs[k]) for k in self.dataset_configs}
        if self.wrap:
            for k in self.datasets: self.datasets[k]=WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        it = isinstance(self.datasets['train'],Txt2ImgIterableBaseDataset)
        init_fn = worker_init_fn if it or self.use_worker_init_fn else None
        return DataLoader(self.datasets["train"],batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=not it,worker_init_fn=init_fn)

    def _val_dataloader(self,shuffle=False):
        it = isinstance(self.datasets['validation'],Txt2ImgIterableBaseDataset)
        init_fn = worker_init_fn if it or self.use_worker_init_fn else None
        return DataLoader(self.datasets["validation"],batch_size=self.batch_size,
                          num_workers=self.num_workers,worker_init_fn=init_fn,shuffle=shuffle)

    def _test_dataloader(self,shuffle=False):
        it = isinstance(self.datasets['train'],Txt2ImgIterableBaseDataset)
        init_fn = worker_init_fn if it or self.use_worker_init_fn else None
        return DataLoader(self.datasets["test"],batch_size=self.batch_size,
                          num_workers=self.num_workers,worker_init_fn=init_fn,shuffle=(shuffle and not it))

    def _predict_dataloader(self,shuffle=False):
        it = isinstance(self.datasets['predict'],Txt2ImgIterableBaseDataset)
        init_fn = worker_init_fn if it or self.use_worker_init_fn else None
        return DataLoader(self.datasets["predict"],batch_size=self.batch_size,
                          num_workers=self.num_workers,worker_init_fn=init_fn)

class SetupCallback(Callback):
    def __init__(self,resume,now,logdir,ckptdir,cfgdir,config,lightning_config):
        super().__init__()
        self.resume, self.now, self.logdir, self.ckptdir, self.cfgdir, self.config, self.lightning_config = \
            resume,now,logdir,ckptdir,cfgdir,config,lightning_config

    def on_keyboard_interrupt(self,trainer,pl_module):
        if trainer.global_rank==0:
            ckpt=os.path.join(self.ckptdir,"last.ckpt")
            trainer.save_checkpoint(ckpt)

    def on_pretrain_routine_start(self,trainer,pl_module):
        if trainer.global_rank==0:
            os.makedirs(self.logdir,exist_ok=True)
            os.makedirs(self.ckptdir,exist_ok=True)
            os.makedirs(self.cfgdir,exist_ok=True)
            if "callbacks" in self.lightning_config and 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                os.makedirs(os.path.join(self.ckptdir,'trainstep_checkpoints'),exist_ok=True)
            print("Project config"); print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,os.path.join(self.cfgdir,f"{self.now}-project.yaml"))
            print("Lightning config"); print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning":self.lightning_config}),
                           os.path.join(self.cfgdir,f"{self.now}-lightning.yaml"))

class ImageLogger(Callback):
    def __init__(self,batch_frequency,max_images,clamp=True,increase_log_steps=True,
                 rescale=True,disabled=False,log_on_batch_idx=False,log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.batch_freq, self.max_images = batch_frequency, max_images
        self.clamp, self.rescale = clamp, rescale
        self.disabled, self.log_on_batch_idx, self.log_first_step = disabled, log_on_batch_idx, log_first_step
        self.log_images_kwargs=log_images_kwargs or {}
        self.log_steps=[2**n for n in range(int(np.log2(self.batch_freq))+1)] if increase_log_steps else [self.batch_freq]
        self.logger_log_images={pl.loggers.TestTubeLogger:self._testtube}

    @rank_zero_only
    def _testtube(self,pl_module,images,batch_idx,split):
        for k,img in images.items():
            grid=(torchvision.utils.make_grid(img)+1)/2
            pl_module.logger.experiment.add_image(f"{split}/{k}",grid,global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self,save_dir,split,images,global_step,current_epoch,batch_idx):
        root=os.path.join(save_dir,"images",split)
        for k,img in images.items():
            grid=torchvision.utils.make_grid(img,nrow=4)
            if self.rescale: grid=(grid+1)/2
            arr=(grid.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            path=os.path.join(root,f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png")
            os.makedirs(os.path.dirname(path),exist_ok=True)
            Image.fromarray(arr).save(path)

    def _do_log(self,trainer,pl_module,images,batch_idx,split):
        for k in images:
            N=min(len(images[k]),self.max_images)
            imgs=images[k][:N]
            if isinstance(imgs,torch.Tensor):
                imgs=imgs.detach().cpu()
                if self.clamp: imgs=torch.clamp(imgs,-1,1)
            images[k]=imgs
        self.log_local(pl_module.logger.save_dir,split,images,pl_module.global_step,pl_module.current_epoch,batch_idx)
        fn=self.logger_log_images.get(type(pl_module.logger),lambda *a,**k:None)
        fn(pl_module,images,pl_module.global_step,split)

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx):
        if not self.disabled and (pl_module.global_step>0 or self.log_first_step):
            self._maybe_log(pl_module,batch,batch_idx,"train")

    def on_validation_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx):
        if not self.disabled and pl_module.global_step>0:
            self._maybe_log(pl_module,batch,batch_idx,"val")

    def _maybe_log(self,pl_module,batch,batch_idx,split):
        idx=batch_idx if self.log_on_batch_idx else pl_module.global_step
        if ( (idx%self.batch_freq==0 or idx in self.log_steps)
             and (idx>0 or self.log_first_step)
             and hasattr(pl_module,"log_images") and callable(pl_module.log_images)
             and self.max_images>0):
            is_train=pl_module.training
            if is_train: pl_module.eval()
            with torch.no_grad(): images=pl_module.log_images(batch,split=split,**self.log_images_kwargs)
            self._do_log(None,pl_module,images,batch_idx,split)
            if is_train: pl_module.train()

class CUDACallback(Callback):
    def on_train_epoch_start(self,trainer,pl_module):
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu);torch.cuda.synchronize(trainer.root_gpu)
        self.start=time.time()
    def on_train_epoch_end(self,trainer,pl_module,outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        mem,et=torch.cuda.max_memory_allocated(trainer.root_gpu)/2**20,time.time()-self.start
        try:
            mem=trainer.training_type_plugin.reduce(mem)
            et =trainer.training_type_plugin.reduce(et)
            rank_zero_info(f"Epoch time: {et:.2f}s, Peak mem: {mem:.2f}MiB")
        except: pass

# -----------------------------------------------------------------------------
# The user-provided, extended diffusion-processing thread
# -----------------------------------------------------------------------------
def process_shared_queue(shared_queue: multiprocessing.Queue, diffusion_model):
    """
    Continuously pulls from shared_queue, runs every Muse module on each message,
    then runs diffusion and saves the image.
    """
    import hashlib
    OUTPUT_DIR = "E:\\musesymbiosis\\outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # instantiate every Muse module once
    web_access_mod          = WebAccessModule()
    memory_mod              = MemoryModule()
    self_awareness_mod      = SelfAwarenessModule()
    self_improvement_mod    = SelfImprovementModule()
    metamorphic_mod         = MetamorphicEngine()
    data_loader_mod         = DataLoaderModule()
    network_integration_mod = NetworkIntegrationModule(ai_server_url=Config.AI_SERVER_URL)
    verbal_comm_mod         = VerbalCommunicationModule()

    seen_hashes = set()
    while True:
        msg = shared_queue.get()
        iteration   = msg.get("iteration")
        ai_response = msg.get("ai_response","")
        cog_state   = msg.get("cognitive_state",{})

        # 1) WebAccess
        try:
            if "search_query" in msg and web_access_mod.is_safe_url(msg["search_query"]):
                page=web_access_mod.safe_get_page(msg["search_query"])
                info=web_access_mod.extract_information(page,{"summary":".article-summary"})
                ai_response += " "+info.get("summary","")
        except Exception as e:
            logging.error(f"[WebAccess][{iteration}] {e}",exc_info=True)

        # 2) Memory
        try:
            memory_mod.store_turn({"iteration":iteration,"response":ai_response,"state":cog_state})
        except Exception as e:
            logging.error(f"[Memory][{iteration}] {e}",exc_info=True)

        # 3) SelfAwareness
        try:
            self_awareness_mod.update(predicted_state=cog_state,
                                      ai_action="DiffusionGenerate",
                                      user_eeg=None,
                                      ai_response=ai_response,
                                      user_utterance="",
                                      interaction_history=[])
        except Exception as e:
            logging.error(f"[SelfAwareness][{iteration}] {e}",exc_info=True)

        # 4) SelfImprovement
        try:
            self_improvement_mod.improve_system(lstm_model=None,
                                                meta_learner=None,
                                                interaction_history=[],
                                                self_awareness_data=self_awareness_mod.get_data())
        except Exception as e:
            logging.error(f"[SelfImprovement][{iteration}] {e}",exc_info=True)

        # 5) MetamorphicEngine
        try:
            metamorphic_mod.full_mutation_cycle(diffusion_model)
        except Exception as e:
            logging.error(f"[Metamorphic][{iteration}] {e}",exc_info=True)

        # 6) DataLoader
        try:
            extra = data_loader_mod.load_batch()
        except Exception as e:
            logging.error(f"[DataLoader][{iteration}] {e}",exc_info=True)

        # 7) NetworkIntegration
        try:
            network_integration_mod.send_and_receive({"iteration":iteration,"status":"starting diffusion"})
        except Exception as e:
            logging.error(f"[NetworkIntegration][{iteration}] {e}",exc_info=True)

        # 8) VerbalCommunication
        try:
            verbal_comm_mod.speak(f"Generating image for iteration {iteration}")
        except:
            pass

        # 9) Diffusion sample
        try:
            image = diffusion_model.sample_text2img(
                prompt=ai_response,
                height=512,
                width=512,
                num_inference_steps=50
            )
            out_path = os.path.join(OUTPUT_DIR,f"diffusion_iter{iteration:05}.png")
            image.save(out_path)
            logging.info(f"[Diffusion][{iteration}] Saved to {out_path}")
        except Exception as e:
            logging.error(f"[Diffusion][{iteration}] Sample failed: {e}",exc_info=True)

# -----------------------------------------------------------------------------
# Muse main loop
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Muse 2 Symbiotic AI System")
    p.add_argument("--mode",choices=["run","test"],default="run",help="run or test")
    return p.parse_args()

def muse_main(shared_queue: multiprocessing.Queue):
    args = parse_args()
    logger = setup_logging(Config.LOG_LEVEL, system_name=Config.SYSTEM_NAME)
    logger.info("Starting Muse system")

    # AI interface
    my_ai = NetworkedAIProgramInterface(Config.AI_SERVER_URL) if Config.USE_NETWORKED_AI else AIProgramInterface()
    muse  = MuseInterface()
    verbal_comm = VerbalCommunicationModule() if (Config.USE_VOICE_INPUT or Config.USE_VOICE_OUTPUT) else None

    interaction_manager = InteractionManager(
        ai_program=my_ai,
        muse_interface=muse,
        verbal_comm=verbal_comm,
        eeg_input_size=Config.LSTM_INPUT_SIZE,
        lstm_hidden_size=Config.LSTM_HIDDEN_SIZE,
        lstm_output_size=Config.LSTM_OUTPUT_SIZE
    )

    # optionally load LSTM state...
    lstm_path = os.path.join("E:\\musesymbiosis","models","eeg_state_lstm.pt")
    if os.path.exists(lstm_path):
        state=torch.load(lstm_path,map_location="cpu")
        interaction_manager.lstm.load_state_dict(state)
        interaction_manager.lstm.eval()

    data_loader = DataLoaderModule() if getattr(Config,'ENABLE_DATASET_LOADING',False) else None
    network_integration = NetworkIntegrationModule(ai_server_url=Config.AI_SERVER_URL) if Config.USE_NETWORKED_AI else None

    # run loop
    iteration=0
    while True:
        iteration+=1
        try:
            eeg_chunk = interaction_manager.muse.get_eeg_data()
            if eeg_chunk.empty:
                time.sleep(0.1); continue
        except:
            time.sleep(0.1); continue

        # preprocess, buffer, predict, select action, get ai_response...
        # (use your existing run_interaction_loop logic, pushing to shared_queue)
        # For brevity, assume you call:
        ai_response = interaction_manager.ai.get_response("...", np.zeros((1,)))  

        # deliver and queue
        print(f"AI: {ai_response}")
        shared_queue.put({
            "iteration": iteration,
            "ai_response": ai_response,
            "cognitive_state": []
        })

        if args.mode=="test" and iteration>=10: break
        time.sleep(0.1)

# -----------------------------------------------------------------------------
# Lightning + diffusion thread
# -----------------------------------------------------------------------------
def lightning_main(shared_queue: multiprocessing.Queue):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt,unknown = parser.parse_known_args()

    # handle resume/name exactly as in your Lightning script...
    if opt.name and opt.resume: raise ValueError("-n and -r cannot both be set")
    if opt.resume:
        if os.path.isfile(opt.resume):
            logdir=os.path.dirname(os.path.dirname(opt.resume)); ckpt=opt.resume
        else:
            logdir=opt.resume.rstrip("/"); ckpt=os.path.join(logdir,"checkpoints","last.ckpt")
        opt.resume_from_checkpoint=ckpt
        bases=sorted(glob.glob(os.path.join(logdir,"configs","*.yaml")))
        opt.base=bases+opt.base
        nowname=os.path.basename(logdir)
    else:
        name=f"_{opt.name}" if opt.name else (f"_{os.path.splitext(os.path.basename(opt.base[0]))[0]}" if opt.base else "")
        nowname=now+name+opt.postfix
        logdir=os.path.join(opt.logdir,nowname)

    ckptdir=os.path.join(logdir,"checkpoints")
    cfgdir=os.path.join(logdir,"configs")
    seed_everything(opt.seed)

    # load configs & instantiate model
    configs=[OmegaConf.load(p) for p in opt.base]
    cli    =OmegaConf.from_dotlist(unknown)
    conf   =OmegaConf.merge(*configs,cli)
    lightning_conf=conf.pop("lightning",OmegaConf.create())

    # merge trainer args
    trainer_cfg=lightning_conf.get("trainer",OmegaConf.create())
    trainer_cfg["accelerator"]="ddp"
    for k in nondefault_trainer_args(opt): trainer_cfg[k]=getattr(opt,k)
    cpu = "gpus" not in trainer_cfg
    if cpu: del trainer_cfg["accelerator"]
    else: print(f"GPUs: {trainer_cfg['gpus']}")
    trainer_opt=argparse.Namespace(**trainer_cfg)
    lightning_conf.trainer=trainer_cfg

    model=instantiate_from_config(conf.model)

    # start our extended diffusion thread
    threading.Thread(target=process_shared_queue,args=(shared_queue,model),daemon=True).start()

    # set up trainer+callbacks exactly as before
    trainer_kwargs={}
    default_logger={"target":"pytorch_lightning.loggers.TestTubeLogger","params":{"name":"testtube","save_dir":logdir}}
    logger_cfg=OmegaConf.merge(default_logger,lightning_conf.logger if "logger" in lightning_conf else OmegaConf.create())
    trainer_kwargs["logger"]=instantiate_from_config(logger_cfg)

    default_ckpt={"target":"pytorch_lightning.callbacks.ModelCheckpoint","params":{"dirpath":ckptdir,"filename":"{epoch:06}","verbose":True,"save_last":True}}
    if hasattr(model,"monitor"):
        default_ckpt["params"].update({"monitor":model.monitor,"save_top_k":3})
    ckpt_cfg=OmegaConf.merge(default_ckpt,lightning_conf.modelcheckpoint if "modelcheckpoint" in lightning_conf else OmegaConf.create())
    if version.parse(pl.__version__)<version.parse("1.4.0"):
        trainer_kwargs["checkpoint_callback"]=instantiate_from_config(ckpt_cfg)

    default_cbs={
        "setup": {"target":"main.SetupCallback","params":{"resume":opt.resume,"now":now,"logdir":logdir,"ckptdir":ckptdir,"cfgdir":cfgdir,"config":conf,"lightning_config":lightning_conf}},
        "image": {"target":"main.ImageLogger","params":{"batch_frequency":750,"max_images":4,"clamp":True}},
        "lr":    {"target":"main.LearningRateMonitor","params":{"logging_interval":"step"}},
        "cuda":  {"target":"main.CUDACallback"}
    }
    if version.parse(pl.__version__)>=version.parse("1.4.0"):
        default_cbs["checkpoint"]=ckpt_cfg
    cb_cfg=OmegaConf.merge(default_cbs,lightning_conf.callbacks if "callbacks" in lightning_conf else OmegaConf.create())
    trainer_kwargs["callbacks"]=[instantiate_from_config(cb_cfg[k]) for k in cb_cfg]

    trainer=Trainer.from_argparse_args(trainer_opt,**trainer_kwargs)
    trainer.logdir=logdir

    # data and run
    data=instantiate_from_config(conf.data)
    data.prepare_data(); data.setup()
    print("#### Data #####")
    for k in data.datasets: print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # lr scaling
    bs,base_lr=conf.data.params.batch_size,conf.model.base_learning_rate
    ngpu=(len(trainer_cfg["gpus"].split(",")) if not cpu else 1)
    accum=lightning_conf.trainer.get("accumulate_grad_batches",1)
    print(f"accum_batches={accum}")
    lightning_conf.trainer.accumulate_grad_batches=accum
    model.learning_rate=(accum*ngpu*bs*base_lr if opt.scale_lr else base_lr)
    print(f"LR set to {model.learning_rate:.2e}")

    # USR1/USR2
    def melk(*a,**k):
        if trainer.global_rank==0:
            trainer.save_checkpoint(os.path.join(ckptdir,"last.ckpt"))
    def divein(*a,**k):
        if trainer.global_rank==0: import pudb; pudb.set_trace()
    import signal
    signal.signal(signal.SIGUSR1,melk); signal.signal(signal.SIGUSR2,divein)

    if opt.train: trainer.fit(model,data)
    if not opt.no_test and not trainer.interrupted: trainer.test(model,data)

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
if __name__=="__main__":
    q = multiprocessing.Queue()
    p1= multiprocessing.Process(target=muse_main,      args=(q,))
    p2= multiprocessing.Process(target=lightning_main, args=(q,))
    p1.start(); p2.start()
    p1.join();  p2.join()
