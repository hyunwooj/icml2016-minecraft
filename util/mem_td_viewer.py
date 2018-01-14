import numpy as np
from PIL import Image
from PIL import ImageEnhance

from top_down_viewer import TopDownViewer


class MemTopDownViewer(TopDownViewer):
    def update_frame_with_mem(self, pos_x, pos_y, facing, cur_frame, mem_img):
        full_img, td_img, cur_img = self.update_frame(pos_x, pos_y, facing, cur_frame)

        h = full_img.shape[0] + mem_img.shape[0]
        w = max(full_img.shape[1], mem_img.shape[1])
        side_by_side = Image.new("RGB", (w, h))

        side_by_side.paste(Image.fromarray(full_img), (0, 0))

        mem_img = Image.fromarray(mem_img)
        enhancer = ImageEnhance.Brightness(mem_img)
        mem_img = enhancer.enhance(2)
        side_by_side.paste(mem_img, (0, full_img.shape[0]))

        full_mem_img = np.asarray(side_by_side)
        return full_mem_img, td_img, cur_img

def create_mem_td_viewer():
    return MemTopDownViewer()
