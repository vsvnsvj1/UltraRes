import numpy as np
import torch
from torch.nn import functional as F
import cv2


class RESRGANinf:
    
    def __init__(self,
                 scale,
                 model=None,
                 model_path=None,
                 device=None,
                 pad=10
                 ) -> None:
        
        self.scale = scale
        self.pad = pad
        self.mod_scale = None
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"{device}")
        
        model_loader = torch.load(model_path, map_location= torch.device("cpu"), weights_only=True)
        keyname = 'params_ema' if 'params_ema' in model_loader else 'params'
        model.load_state_dict(model_loader[keyname], strict=True)
        
        model.eval()
        self.model = model.to(self.device)
    
    
    def pre_process(self, img):
        
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        
        if self.pad != 0:
            self.img = F.pad(self.img, (0, self.pad, 0, self.pad), 'reflect')
            
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
            
        return self.img
    

    def inference(self):
        self.output = self.model(self.img)
        
    def post_process(self):
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pad * self.scale, 0:w - self.pad * self.scale]
        return self.output
    
    @torch.no_grad()
    def upgrade_resolution(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input_img, w_input_img = img.shape[0:2]
        
        img = img.astype(np.float32)
        if np.max(img) > 256:
            max_range = 65536
        else:
            max_range = 256
        img = img / max_range
        if len(img.shape) == 2:
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        self.pre_process(img)

        self.inference()
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                self.inference()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)
            self.img = None
            self.output = None
            import gc
            gc.collect() 
            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha
            
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input_img * outscale),
                    int(h_input_img * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode
        
        

    