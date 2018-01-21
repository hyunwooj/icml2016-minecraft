import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from top_down_viewer import TopDownViewer


class MemTopDownViewer(TopDownViewer):
    def update_frame_with_mem(self, pos_x, pos_y, facing, cur_frame, mem_img, dbg):
        full_img, td_img, cur_img = self.update_frame(pos_x, pos_y, facing, cur_frame)

        font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
        font_size = 20
        font = ImageFont.truetype(font_path, font_size)

        h = full_img.shape[0] + mem_img.shape[0]
        if dbg:
            h += font_size * len(dbg)
        w = max(full_img.shape[1], mem_img.shape[1])
        side_by_side = Image.new("RGB", (w, h))

        side_by_side.paste(Image.fromarray(full_img), (0, 0))

        mem_img = Image.fromarray(mem_img)
        enhancer = ImageEnhance.Brightness(mem_img)
        mem_img = enhancer.enhance(2)
        side_by_side.paste(mem_img, (0, full_img.shape[0]))

        if dbg:
            self._draw_dbg(side_by_side, cur_frame, dbg, font, font_size)

        full_mem_img = np.asarray(side_by_side)
        return full_mem_img, td_img, cur_img

    def _draw_dbg(self, side_by_side, cur_frame, dbg, font, font_size):
        draw = ImageDraw.Draw(side_by_side)

        frame_w, frame_h, _ = cur_frame.shape

        for i, (atten, reten) in enumerate(zip(dbg['atten'], dbg['reten'])):
            # Attention
            text = 'Atten: %.3f' % atten
            pos = (i * frame_w + 10, 2*frame_h + 0*font_size)
            draw.text(pos, text, fill='white', font=font)

            # Retention
            text = 'Reten: %.3f' % reten
            pos = (i * frame_w + 10, 2*frame_h + 1*font_size)
            draw.text(pos, text, fill='white', font=font)

def create_mem_td_viewer():
    return MemTopDownViewer()
