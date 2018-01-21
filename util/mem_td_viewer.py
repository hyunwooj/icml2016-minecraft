import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from top_down_viewer import TopDownViewer


class MemTopDownViewer(TopDownViewer):
    def update_frame_with_mem(self, pos_x, pos_y, facing, cur_frame, mem_img, dbg, mem_dbg):
        full_img, td_img, cur_img = self.update_frame(pos_x, pos_y, facing, cur_frame)

        font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
        font_size = 20
        font = ImageFont.truetype(font_path, font_size)

        h = full_img.shape[0] + mem_img.shape[0]
        if mem_dbg:
            h += font_size * len(mem_dbg)
        w = max(full_img.shape[1], mem_img.shape[1])
        side_by_side = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(side_by_side)

        side_by_side.paste(Image.fromarray(full_img), (0, 0))

        mem_img = Image.fromarray(mem_img)
        enhancer = ImageEnhance.Brightness(mem_img)
        mem_img = enhancer.enhance(2)
        side_by_side.paste(mem_img, (0, full_img.shape[0]))

        if mem_dbg:
            self._draw_mem_dbg(side_by_side, cur_frame, mem_dbg, font, font_size, draw)

        draw.text((5, 5), 'Time:   %d' % dbg['time'], fill='white', font=font)
        draw.text((5, 25), 'Reward: %.2f' % dbg['reward'], fill='white', font=font)

        full_mem_img = np.asarray(side_by_side)
        return full_mem_img, td_img, cur_img

    def _draw_mem_dbg(self, side_by_side, cur_frame, mem_dbg, font, font_size, draw):

        frame_w, frame_h, _ = cur_frame.shape

        for i, (atten, reten, time) in enumerate(zip(mem_dbg['atten'],
                                                     mem_dbg['reten'],
                                                     mem_dbg['times'])):
            # Attention
            text = 'Atten: %.3f' % atten
            pos = (i * frame_w + 10, 2*frame_h + 0*font_size)
            draw.text(pos, text, fill='white', font=font)

            # Retention
            text = 'Reten: %.3f' % reten
            pos = (i * frame_w + 10, 2*frame_h + 1*font_size)
            draw.text(pos, text, fill='white', font=font)

            # Retention
            text = 'Time : %d' % time
            pos = (i * frame_w + 10, 2*frame_h + 2*font_size)
            draw.text(pos, text, fill='white', font=font)

def create_mem_td_viewer():
    return MemTopDownViewer()
