import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy
from .kac_ta import KACLayer


# Our method!
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4] #[0,1,2,3,4,5,6,7,8,9,10,11] 

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block, train=False, task_id=None, return_sim=False):
        sim_info = None

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
                # print(K.shape, A.shape, p.shape)
                # exit()
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)

            
            if return_sim and (not train):
                max_sim, max_idx = aq_k.max(dim=1)   # [B], [B]
                sim_info = {"max_sim": max_sim, "max_idx": max_idx, "cos_sim": aq_k}

            
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        if return_sim:
            return p_return, loss, x_block, sim_info
        else:
            return p_return, loss, x_block



def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p  



class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, clf_type = "linear"):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.num_tasks = prompt_param[0]
        print('num_tasks',self.num_tasks)

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                        num_heads=12, ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            from timm.models import vit_base_patch16_224
            load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict)

        # classifier
        if clf_type == "linear":
            self.last = nn.Linear(768, num_classes)
        elif clf_type == "kac":
            self.last = KACLayer(768, num_classes, num_tasks=self.num_tasks)
        
        else:
            raise ValueError(
                f"Invalid classifier type. Expected one of: ['kac', 'linear'], but got '{clf_type}' instead."
            )


        
        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'taskspec':
            self.prompt = TaskSpecificPrompt(768, prompt_param[0], prompt_param[1])            
        else:
            self.prompt = None
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        
    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False, return_prompt_sims=False):

        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.feat(x)  
                q = q[:, 0, :]
        
            if (not train):
                out, prompt_loss, prompt_sims = self.feat(
                    x, prompt=self.prompt, q=q, train=train, task_id=self.task_id,
                    return_prompt_sims=True
                )

                # NEW: infer task_id for this batch from prompt_sims
                inferred = self._infer_task_id_from_prompt_sims(prompt_sims)
                if inferred is None:
                    raise RuntimeError("Task inference returned None")
            
            else:
                out, prompt_loss = self.feat(
                    x, prompt=self.prompt, q=q, train=train, task_id=self.task_id,
                    return_prompt_sims=False
                )
                prompt_sims = None
                 
        else:
            out, _ = self.feat(x)
            
        out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            if hasattr(self.last, "set_task_id"):
                if train:
                    self.last.set_task_id(self.task_id)
                else:
                    self.last.set_task_id(inferred)
            out = self.last(out)
    

        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            if return_prompt_sims:
                return out, prompt_sims
            return out


    
    
    def _idx_to_task(self, max_idx: torch.Tensor) -> torch.Tensor:
        if self.prompt is None:
            return max_idx
        
        e_pool = int(getattr(self.prompt, "e_pool_size", 0) or 0)
        if e_pool > 0 and self.num_tasks > 0:
            pt = max(1, e_pool // self.num_tasks)
            return (max_idx // pt).clamp(min=0, max=self.num_tasks - 1)
        return max_idx  # fallback
    
    def _infer_task_id_from_prompt_sims(self, prompt_sims: dict) -> int | None:
        if not prompt_sims:
            return None
    
        per_layer_tasks = []
        for _, info in prompt_sims.items():
            if info is None:
                continue
            max_idx = info.get("max_idx", None)
            if max_idx is None:
                continue
            per_layer_tasks.append(self._idx_to_task(max_idx).long())  # [B]
    
        if len(per_layer_tasks) == 0:
            return None
    
        # [L, B]
        tlb = torch.stack(per_layer_tasks, dim=0)
    
        # per-sample mode over layers: [B]
        # torch.mode returns (values, indices)
        sample_task = torch.mode(tlb, dim=0).values
    
        # batch mode over samples -> scalar
        batch_task = torch.mode(sample_task, dim=0).values.item()
        return int(batch_task)
        

def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None, clf_type = 'linear'):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, clf_type = clf_type)

