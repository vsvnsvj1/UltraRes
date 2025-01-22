import numpy as np
import torch
from torch.nn import functional as F
import cv2
from app.model.memory_manager import MemoryManager

class RESRGANinf:
    
    def __init__(self,
                 scale,
                 model=None,
                 model_path=None,
                 device=None,
                 calc_tiles = False,
                 tile_pad = 10,
                 pad=10,
                 pixel_size_kb = 50
                 ) -> None:
        self.calc_tiles = calc_tiles
        self.tile_pad = tile_pad
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
        
        self.memory_manager = MemoryManager(pixel_cost_kb=pixel_size_kb, device=self.device) if calc_tiles else None
    
    
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
    
    def tile_inference(self, tile_size):
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)
        
        # Calculate count of tiles, depends on available memory
        #tile_size = self.memory_manager.calculate_tile_count(batch, channel, height, width)
        print(f'tile count {tile_size}')

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = int(np.ceil(width / tile_size))
        tiles_y = int(np.ceil(height / tile_size))
        print(tiles_x, tiles_y)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                if tile_idx % 1000 == 0:
                    print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

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
        
        batch, channel, height, width = self.img.shape
        tile_size = self.memory_manager.calculate_tile_count(batch, channel, height, width)
        
        if self.calc_tiles == True and tile_size > 1:
            
            self.tile_inference(tile_size * 2 if tile_size > 5 else 10)
        else:   
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
        
        

    