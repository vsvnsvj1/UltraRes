import logging

import cv2
import numpy as np
import torch
from model.memory_manager import MemoryManager
from torch.nn import functional

logger = logging.getLogger(__name__)


class RESRGANinf:
    def __init__(
        self,
        scale,
        model=None,
        model_path=None,
        device=None,
        calc_tiles=False,
        tile_pad=10,
        pad=10,
        pixel_size_kb=50,
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

        logger.debug(f"Loading model from {model_path}...")
        model_loader = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        keyname = "params_ema" if "params_ema" in model_loader else "params"
        model.load_state_dict(model_loader[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)

        self.memory_manager = (
            MemoryManager(
                pixel_cost_kb=pixel_size_kb,
                device=self.device,
            )
            if calc_tiles
            else None
        )

        logger.warning(
            f"Initialized RESRGANinf with scale={self.scale}, device={self.device}, "
            f"calc_tiles={self.calc_tiles}, tile_pad={self.tile_pad}, pad={self.pad}",
        )

    def pre_process(self, img):
        logger.debug(f"Pre-processing image with shape: {img.shape}")
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)

        if self.pad != 0:
            self.img = functional.pad(self.img, (0, self.pad, 0, self.pad), "reflect")

        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, height, width = self.img.size()
            if height % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - height % self.mod_scale
            if width % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - width % self.mod_scale
            self.img = functional.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")
            logger.debug(
                f"Image dimensions adjusted with padding: "
                f"mod_pad_h={self.mod_pad_h}, mod_pad_w={self.mod_pad_w}",
            )

        return self.img

    def tile_inference(self, tile_size):
        logger.debug(f"Starting tiled inference with tile size: {tile_size}")
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = int(np.ceil(width / tile_size))
        tiles_y = int(np.ceil(height / tile_size))
        # loop over all tiles
        for tile_row_index in range(tiles_y):
            for tile_column_index in range(tiles_x):
                # extract tile from input image
                offset_x = tile_column_index * tile_size
                offset_y = tile_row_index * tile_size
                # input tile area on total image
                input_start_x = offset_x
                input_end_x = min(offset_x + tile_size, width)
                input_start_y = offset_y
                input_end_y = min(offset_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                input_tile = self.img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print("Error", error)

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
                self.output[
                    :,
                    :,
                    output_start_y:output_end_y,
                    output_start_x:output_end_x,
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]
        logger.debug("Tiled inference completed.")

    def inference(self):
        logger.debug("Starting inference on the whole image.")
        self.output = self.model(self.img)
        logger.debug("Inference completed.")

    def post_process(self):
        logger.debug("Post-processing output image.")
        if self.mod_scale is not None:
            _, _, height, width = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : height - self.mod_pad_h * self.scale,
                0 : width - self.mod_pad_w * self.scale,
            ]
        # remove prepad
        if self.pad != 0:
            _, _, height, width = self.output.size()
            self.output = self.output[
                :, :,
                0 : height - self.pad * self.scale, 0 : width - self.pad * self.scale,
            ]

        logger.debug(f"Post-processing completed. Final output shape: {self.output.shape}")
        return self.output

    def _prepare_image(self, img, alpha_upsampler):
        img = img.astype(np.float32)
        if np.max(img) > 256:
            max_range = 65536
        else:
            max_range = 256

        img = img / max_range
        alpha = None  # Инициализация alpha по умолчанию

        if len(img.shape) == 2:
            img_mode = "L"
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_mode = "RGBA"
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == "realesrgan":
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = "RGB"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.pre_process(img)

        return img, img_mode, alpha, max_range

    def _process_image(self):
        batch, channel, height, width = self.img.shape
        if self.calc_tiles:
            tile_size = self.memory_manager.calculate_tile_count(batch, channel, height, width)
            if tile_size > 1:
                self.tile_inference(tile_size * 2 if tile_size > 5 else 10)
            else:
                self.inference()
        else:
            self.inference()


    def _finalize_image(self, img_mode, max_range, alpha, alpha_upsampler):
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        # Обработка для режимов 'L' и 'RGBA'
        if img_mode == "L":
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        if img_mode == "RGBA":
            output_alpha = self._process_alpha(alpha, alpha_upsampler)
            output_img = self._merge_alpha(output_img, output_alpha)

        # Приведение к исходному диапазону
        if max_range == 65536:  # 16-bit изображение
            output_img = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output_img = (output_img * 255.0).round().astype(np.uint8)

        return output_img

    def _process_alpha(self, alpha, alpha_upsampler):
        if alpha_upsampler == "realesrgan":
            self.pre_process(alpha)
            self.inference()
            output_alpha = self.post_process()
            output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
            output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
        else:
            height, width = alpha.shape[0:2]
            output_alpha = cv2.resize(
                alpha,
                (width * self.scale, height * self.scale),
                interpolation=cv2.INTER_LINEAR,
            )
        return output_alpha

    def _merge_alpha(self, output_img, output_alpha):
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha
        return output_img

    def _rescale_output(self, output, original_shape, outscale):
        h_input_img, w_input_img = original_shape
        return cv2.resize(
            output,
            (
                int(w_input_img * outscale),
                int(h_input_img * outscale),
            ),
            interpolation=cv2.INTER_LANCZOS4,
        )

    @torch.no_grad()
    def upgrade_resolution(self, img, outscale=None, alpha_upsampler="realesrgan"):
        logger.debug(f"Upgrading resolution for image with shape: {img.shape}")

        img, img_mode, alpha, max_range = self._prepare_image(img, alpha_upsampler)

        self._process_image()

        output_img = self._finalize_image(img_mode, max_range, alpha, alpha_upsampler)

        if outscale is not None and outscale != float(self.scale):
            output_img = self._rescale_output(output_img, img.shape[:2], outscale)

        logger.debug("Resolution upgrade completed.")
        return output_img, img_mode
