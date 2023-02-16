import omni.ext
import omni.ui
import numpy as np
import os

def add_separator(height=4, width=2, color=0xff202020):
    omni.ui.Spacer(height=height)
    omni.ui.Line(style={"border_width":width, "color":color})
    omni.ui.Spacer(height=height)

def add_label(text):
    # with omni.ui.HStack(width=0):
    omni.ui.Spacer(width=8)
    label = omni.ui.Label(text)
    label.alignment = omni.ui._ui.Alignment.H_CENTER
    omni.ui.Spacer(width=4)
    return label

def add_btn(text, enabled=True, scale=1):
    omni.ui.Spacer(height=4)
    btn = omni.ui.Button(text)
    btn.height *= scale
    omni.ui.Spacer(height=4)
    
    btn.enabled = enabled
    return btn

# ----------------------------------------------------------.
class WidgetsExtension(omni.ext.IExt):
    # ------------------------------------------------.
    # Init window.
    # ------------------------------------------------.
    def init_window (self, env=None, image_folder=None):
        self.env = env
        self.image_folder = image_folder

        # Create new window.
        self._window = omni.ui.Window("Widgets Window", width=340, height=600)

        # ------------------------------------------.
        with self._window.frame:
            # Create window UI.
            with omni.ui.VStack(height=0):
                self.btn = add_btn(" START ", scale=3)
                self.btn.set_clicked_fn(self.onButtonClicked)
                add_separator(8)

                self.target_label = add_btn("Target", False)
                self.target_image = omni.ui.Image("", width=280, height=180)
                # Separator.
                add_separator(8)
                self.source_label = add_btn("Source", False)
                self.source_image = omni.ui.Image("", width=280, height=180)

                # Separator.
                add_separator(8)
                self.final_label = add_btn("Final", False)
                self.final_image = omni.ui.Image("", width=280, height=180)
    
    def onButtonClicked(self):
        if self.env is not None:
            self.env.is_start = True
            self.btn.enabled = False

        # index = np.random.random_integers(0,5)
        # folder = f"E:/workspace/visual_match/images/sc_{index}.png"
        # if index == 4:
        #     folder = f"E:/workspace/visual_match/images/sc_rgb.png"
        # print( type(self.source_label.alignment) )
        # self.source_image.source_url = folder

    def show_source_img(self):
        self.source_image.source_url = os.path.join(self.image_folder, "sc_rgb.png")

    def show_target_img(self):
        self.target_image.source_url = os.path.join(self.image_folder, "tg_rgb.png")

    def show_final_img(self):
        self.final_image.source_url = os.path.join(self.image_folder, "fin_rgb.png")

    # ------------------------------------------------.
    # Term window.
    # ------------------------------------------------.
    def term_window (self):
        if self._window != None:
            self._window = None

    # ------------------------------------------------.
    # Startup.
    # ------------------------------------------------.
    def on_startup(self, ext_id=0):
        self.init_window()

    # ------------------------------------------------.
    # Shutdown.
    # ------------------------------------------------.
    def on_shutdown(self):
        self.term_window()