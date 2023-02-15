
import numpy as np
import os
import torch
import itertools
import matplotlib
import matplotlib.figure
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Tensorboard:
    def __init__(self, logdir):
        self.writer = None
        self.logdir = logdir

    def close(self):
        self.writer.close()

    def make_dir(self):  # lazy making to avoid empty dir generation.

        if self.writer == None:
            self.writer = tf.summary.create_file_writer(self.logdir)

    def log_text(self, tag, string, step):
        self.make_dir()

        with tf.device("CPU:0"), self.writer.as_default():
            tf.summary.text(tag, tf.convert_to_tensor(str(string)), step)
        self.writer.flush()

    def get_scalar(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return value

    def log_scalar(self, tag, value, global_step):

        self.make_dir()

        value = self.get_scalar(value)
        with tf.device("CPU:0"), self.writer.as_default():
            tf.summary.scalar(tag, value, global_step)
        self.writer.flush()

    def log_confusion_matrix(self, tag, cm, label_names, global_step, normalize=False):
        '''
        Parameters:
            correct_labels                  : These are your true classification categories.
            predict_labels                  : These are you predicted classification categories
            label_names                     : This is a list of labels which will be used to display the axix labels
            title='Confusion matrix'        : Title for your matrix
            tensor_name = 'MyFigure/image'  : Name for the output summay tensor

        Returns:
            summary: TensorFlow summary

        Other itema to note:
            - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
            - Currently, some of the ticks dont line up due to rotations.
        '''

        # make confusion matrix

        value_font_size = 140 // len(label_names)

        if normalize:
            cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm = cm.astype('int')

        np.set_printoptions(precision=2)
        ###fig, ax = matplotlib.figure.Figure()

        fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, cmap='Oranges')

        # classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in label_names]
        # classes = ['\n'.join(wrap(l, 40)) for l in classes]
        classes = label_names

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=25)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=value_font_size, rotation=0, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=25)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=value_font_size, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center",
                    fontsize=value_font_size,
                    verticalalignment='center', color="black")
        fig.set_tight_layout(True)

        # write confusion matrix in tensorboard
        self.make_dir()


        # self.writer.add_summary(summary, global_step)

        cm_image = ax.get_figure()
        import io
        buf = io.BytesIO()
        cm_image.savefig(buf, format='png')
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        with tf.device("CPU:0"), self.writer.as_default():
            image = tf.expand_dims(image, 0) #should be inside tf.device("CPU:0")
            tf.summary.image(tag, image, global_step)

        self.writer.flush()


if __name__ == '__main__':
    pass
