import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
sys.path.append('/mnt/d/pancheng/Project/IDR-Jittor/code')
import jittor as jt
import utils.general as utils
import utils.plots as plt
class IDRTrainRunner():
    def __init__(self,**kwargs):
        # jt.set_default_dtype(jt.float32)
        # jt.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.train_cameras = kwargs['train_cameras']
        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        # self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(self.train_cameras,
        #                                                                                   **dataset_conf)

        print('Finish loading data ...')

        self.train_dataloader = utils.get_class(self.conf.get_string('train.dataset_class'))(self.train_cameras,
                                                            **dataset_conf,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            # collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = utils.get_class(self.conf.get_string('train.dataset_class'))(self.train_cameras,
                                                            **dataset_conf,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            # collate_fn=self.train_dataset.collate_fn
                                                            )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        # if jt.cuda.is_available():
        #     self.model
        print("this is :", __file__, '   line',sys._getframe().f_lineno) 
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = jt.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = jt.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        # settings for camera optimization
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = jt.nn.Embedding(num_images, 7, sparse=True)
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

            self.optimizer_cam = jt.optim.SparseAdam(self.pose_vecs.parameters(), self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = jt.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = jt.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = jt.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.train_cameras:
                data = jt.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                data = jt.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataloader.total_pixels
        self.img_res = self.train_dataloader.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor
                
        print("this is :", __file__, '   line',sys._getframe().f_lineno) 
    def save_checkpoints(self, epoch):
        jt.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        jt.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        jt.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        jt.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        # jt.save(
        #     {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
        #     os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        # jt.save(
        #     {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
        #     os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            jt.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            jt.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            jt.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            jt.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    def run(self):
        print("training...")

        for epoch in range(self.start_epoch, self.nepochs + 1):

            if epoch in self.alpha_milestones:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

            if epoch % 100 == 0:
                self.save_checkpoints(epoch)

            if epoch % self.plot_freq == 0:
                self.model.eval()
                if self.train_cameras:
                    self.pose_vecs.eval()
                self.train_dataloader.change_sampling_idx(-1)
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                model_input["intrinsics"] = model_input["intrinsics"]
                model_input["uv"] = model_input["uv"]
                model_input["object_mask"] = model_input["object_mask"]

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices)
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose']

                split = utils.split_input(model_input, self.total_pixels)
                res = []
                cnt = 0
                for s in split:
                    out = self.model(s)
                    print("cnt :", cnt, " len: ", len(split))
                    cnt += 1
                    res.append({
                        'points': out['points'].detach(),
                        'rgb_values': out['rgb_values'].detach(),
                        'network_object_mask': out['network_object_mask'].detach(),
                        'object_mask': out['object_mask'].detach()
                    })

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

                plt.plot(self.model,
                         indices,
                         model_outputs,
                         model_input['pose'],
                         ground_truth['rgb'],
                         self.plots_dir,
                         epoch,
                         self.img_res,
                         **self.plot_conf
                         )

                self.model.train()
                if self.train_cameras:
                    self.pose_vecs.train()

            self.train_dataloader.change_sampling_idx(self.num_pixels)
            # self.train_dataloader.change_sampling_idx(-1)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                model_input["intrinsics"] = model_input["intrinsics"]
                model_input["uv"] = model_input["uv"]
                model_input["object_mask"] = model_input["object_mask"]
                # print("idx: ", indices)
                # print("object_mask: ", model_input['object_mask'].size())
                # print("uv: ", model_input['uv'].size())
                # print("intrinsics: ", model_input['intrinsics'].size())
                
                if self.train_cameras:
                    pose_input = self.pose_vecs(indices)
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose']
                
                # print("idx: ", indices)
                # # print("object_mask: ", model_input['object_mask'].size())
                # # print("uv: ", model_input['uv'].size())
                # # print("intrinsics: ", model_input['intrinsics'].size())
                # print("pose: ", model_input['pose'].size())
                # print("pos = ", model_input['pose'])
                # print("model_input.size()", model_input.shape)
                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.optimizer.zero_grad()
                if self.train_cameras:
                    self.optimizer_cam.zero_grad()

                self.optimizer.backward(loss)

                self.optimizer.step()
                if self.train_cameras:
                    self.optimizer_cam.step()

                # print("type(self.scheduler) : ", type(self.scheduler))
                print(
                    '{0} [{1}] ({2}/{3}): loss = {4}, rgb_loss = {5}, eikonal_loss = {6}, mask_loss = {7}, alpha = {8}, lr = {9}'
                        .format(self.expname, epoch, data_index, self.n_batches, loss.item(),
                                loss_output['rgb_loss'],
                                loss_output['eikonal_loss'],
                                loss_output['mask_loss'],
                                self.loss.alpha,
                                self.scheduler.get_lr()))

            self.scheduler.step()