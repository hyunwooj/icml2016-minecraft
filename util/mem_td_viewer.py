import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from top_down_viewer import TopDownViewer


class MemTopDownViewer(TopDownViewer):
    def update_frame_with_mem(self, pos_x, pos_y, facing, cur_frame, mem_img, dbg, mem_dbg):
        full_img, td_img, cur_img = self.update_frame(pos_x, pos_y, facing, cur_frame)

        font_path = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
        font_size = 20
        font = ImageFont.truetype(font_path, font_size)

        font_bold_path = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
        font_bold = ImageFont.truetype(font_bold_path, font_size)

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

        draw.text((5, 5), 'Time:   %.0f' % dbg['time'], fill='white', font=font)
        draw.text((5, 25), 'Reward: %.2f' % dbg['reward'], fill='white', font=font)

        if dbg.get('beh_q') is not None:
            pos = (5 + full_img.shape[1], 5)
            self._draw_beh_q_values(dbg['beh_q'], pos, font, font_bold, font_size, draw)

        if dbg.get('mem_q') is not None:
            pos = (5 + full_img.shape[1], 5 + 7 * font_size)
            self._draw_mem_q_values(dbg['mem_q'], pos, font, font_bold, font_size, draw)

        full_mem_img = np.asarray(side_by_side)
        return full_mem_img, td_img, cur_img

    def _draw_beh_q_values(self, beh_q, pos, font, font_bold, font_size, draw):
        x, y = pos
        labels = ['LR', 'LL', 'LU', 'LD', 'FD', 'BD']

        assert(len(labels) == len(beh_q))
        max_beh = np.argmax(beh_q)

        for i, (label, value) in enumerate(zip(labels, beh_q)):
            f = font_bold if i == max_beh else font
            draw.text((x, y + i * font_size), '%s: %.3f' % (label, value),
                      fill='white', font=f)

    def _draw_mem_q_values(self, mem_q, pos, font, font_bold, font_size, draw):
        x, y = pos
        max_mem = np.argmax(mem_q)

        for i, value in enumerate(mem_q):
            f = font_bold if i == max_mem else font
            draw.text((x, y + i * font_size), '%d: %.3f' % (i, value),
                      fill='white', font=f)


    def _draw_mem_dbg(self, side_by_side, cur_frame, mem_dbg, font, font_size, draw):

        frame_w, frame_h, _ = cur_frame.shape
        mem_size = len(mem_dbg['atten'])
        pos_y = 0

        for i in range(mem_size):
            # Attention
            text = 'Atten: %.3f' % mem_dbg['atten'][i]
            pos = (i * frame_w + 10, 2*frame_h + pos_y*font_size)
            draw.text(pos, text, fill='white', font=font)
            pos_y += 1

            # Retention
            text = 'Reten: %.3f' % mem_dbg['reten'][i]
            pos = (i * frame_w + 10, 2*frame_h + pos_y*font_size)
            draw.text(pos, text, fill='white', font=font)
            pos_y += 1

            if mem_dbg.get('stren') is not None:
                # Strength
                text = 'Stren: %.3f' % mem_dbg['stren'][i]
                pos = (i * frame_w + 10, 2*frame_h + pos_y*font_size)
                draw.text(pos, text, fill='white', font=font)
                pos_y += 1

            if mem_dbg.get('sigma') is not None:
                # Strength
                text = 'Sigma: %.3f' % mem_dbg['sigma'][i]
                pos = (i * frame_w + 10, 2*frame_h + pos_y*font_size)
                draw.text(pos, text, fill='white', font=font)
                pos_y += 1

            if mem_dbg.get('comps') is not None:
                # Compound Strength
                text = 'CompS: %.3f' % mem_dbg['comps'][i]
                pos = (i * frame_w + 10, 2*frame_h + pos_y*font_size)
                draw.text(pos, text, fill='white', font=font)
                pos_y += 1

            # Time
            text = 'Time : %.0f' % mem_dbg['times'][i]
            pos = (i * frame_w + 10, 2*frame_h + pos_y*font_size)
            draw.text(pos, text, fill='white', font=font)
            pos_y += 1

            pos_y = 0

def create_mem_td_viewer():
    return MemTopDownViewer()
