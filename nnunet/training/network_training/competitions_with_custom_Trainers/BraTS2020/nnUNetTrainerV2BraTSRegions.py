#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from multiprocessing import Queue, Process
from threading import Thread
from time import sleep
from typing import Tuple, Union
import matplotlib
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from nnunet.configuration import default_num_threads
from nnunet.evaluation.region_based_evaluation import evaluate_regions, get_brats_regions
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_BCE_loss, get_tp_fp_fn_tn, SoftDiceLoss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.nnUNetTrainerV2_DDP import nnUNetTrainerV2_DDP
from nnunet.utilities.distributed import awesome_allgather_function
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

import sys
sys.path.append('/home/bruno-pacheco/brats/')
from brats.utils import dice

eval_every = 2


class nnUNetTrainerV2BraTSRegions_BN(nnUNetTrainerV2):
    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = torch.nn.Softmax(1)


def postprocess_softmax_return_dice(segmentation_softmax: Union[str, np.ndarray],
                                    properties_dict: dict, order: int = 1,
                                    region_class_order: Tuple[Tuple[int]] = None,
                                    force_separate_z: bool = None,
                                    interpolation_order_z: int = 0,
                                    gt_fpath: str = None):
    """THIS IS A COPY OF `save_segmentation_nifti_from_softmax`.

    I just stripped it of the saving part so it returns the pred image. And added dice calculation.
    """
    if isinstance(segmentation_softmax, str):
        assert isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        segmentation_softmax = np.load(segmentation_softmax)
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z, cval=0,
                                               order_z=interpolation_order_z)
    else:
        seg_old_spacing = segmentation_softmax

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])

    pred = sitk.GetArrayFromImage(seg_resized_itk).transpose()
    label = nib.load(gt_fpath).get_fdata()

    scores = list()
    for c in range(1,3+1):
        scores.append(dice(pred, label, c))

    return scores


class nnUNetTrainerV2BraTSRegions_BN_Our(nnUNetTrainerV2BraTSRegions_BN):
    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)

        Modified to run only every `eval_every` epochs.
        :return:
        """
        if self.epoch % eval_every == 0:
            if self.val_eval_criterion_MA is None:
                if len(self.all_val_eval_metrics) == 0:
                    self.val_eval_criterion_MA = - self.all_val_losses[-1]
                else:
                    all_MAs = [self.all_val_eval_metrics[0]]
                    for i in range(1, len(self.all_val_eval_metrics)):
                        all_MAs.append(
                            self.val_eval_criterion_alpha * all_MAs[i-1] +
                            (1 - self.val_eval_criterion_alpha) * self.all_val_eval_metrics[i]
                        )
                    self.all_val_eval_criterion_MAs = all_MAs
                    self.val_eval_criterion_MA = all_MAs[-1]
            else:
                if len(self.all_val_eval_metrics) == 0:
                    """
                    We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                    is better, so we need to negate it.
                    """
                    self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                            1 - self.val_eval_criterion_alpha) * \
                                                self.all_val_losses[-1]
                else:
                    self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                            1 - self.val_eval_criterion_alpha) * \
                                                self.all_val_eval_metrics[-1]
                    self.all_val_eval_criterion_MAs.append(self.val_eval_criterion_MA)

    def plot_progress(self):
        """
        Should probably be improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")

            x_values_eval = list(range(0, self.epoch + 1, eval_every))
            if len(self.all_val_eval_metrics) == len(x_values_eval):
                ax2.plot(x_values_eval, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")
                ax2.plot(x_values_eval, self.all_val_eval_criterion_MAs, color='g', label=f"(alpha={self.val_eval_criterion_alpha})")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax.grid()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def finish_online_evaluation(self):
        # leaving this here just so I'm sure I won't mess up with anything
        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        if self.epoch % eval_every == 0:
            # from validate():
            self.print_to_log_file("WARNING: the evaluation was modified to perform"
                                " exactly as in the validation. This will take a while.")

            # input args
            do_mirroring = True
            use_sliding_window = True
            step_size = 0.5
            save_softmax = False
            use_gaussian = True
            overwrite = True
            all_in_gpu = False

            ds = self.network.do_ds
            self.network.do_ds = False
            current_mode = self.network.training
            self.network.eval()

            assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
            if self.dataset_val is None:
                self.load_dataset()
                self.do_split()

            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0

            # predictions as they come from the network go here
            output_folder = "/home/bruno-pacheco/brats-generalization/.tmpdir/"
            maybe_mkdir_p(output_folder)

            if do_mirroring:
                if not self.data_aug_params['do_mirror']:
                    raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
                mirror_axes = self.data_aug_params['mirror_axes']
            else:
                mirror_axes = ()

            # export_pool = Pool(default_num_threads)
            results = []

            def load(ks_queue, data_queue):
                try:
                    while True:
                        k = ks_queue.get()

                        if k is None:
                            ks_queue.put(None)
                            break

                        properties = load_pickle(self.dataset[k]['properties_file'])
                        fname = properties['list_of_data_files'][0].split("/")[-1][:-12]

                        data = np.load(self.dataset[k]['data_file'])['data']

                        data[-1][data[-1] == -1] = 0

                        q_size = data_queue.qsize()
                        print(k, data.shape, self.dataset[k]['data_file'], q_size)

                        data_queue.put((data[:-1], fname, properties))
                except Exception as e:
                    print('LOADER EXCEPTION:')
                    print(e)
                finally:
                    data_queue.put(None)

            def predict(data_queue, pred_queue):
                while True:
                    item = data_queue.get()

                    if item is None:
                        # propagate ending signal to other consumers
                        data_queue.put(None)
                        break
                    else:
                        data, fname, properties = item

                    ret = self.predict_preprocessed_data_return_seg_and_softmax(
                        # data[:-1],
                        data,
                        do_mirroring=do_mirroring,
                        mirror_axes=mirror_axes,
                        use_sliding_window=use_sliding_window,
                        step_size=step_size,
                        use_gaussian=use_gaussian,
                        all_in_gpu=all_in_gpu,
                        mixed_precision=self.fp16,
                        verbose=False
                    )

                    softmax_pred = ret[1]

                    softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    """There is a problem with python process communication that prevents us from communicating obejcts
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    # if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    #     np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    #     softmax_pred = join(output_folder, fname + ".npy")

                    # results.append(export_pool.starmap_async(postprocess_softmax_return_dice,
                    #                                         ((softmax_pred,
                    #                                         properties, interpolation_order, self.regions_class_order,
                    #                                         force_separate_z, interpolation_order_z,
                    #                                         join(self.gt_niftis_folder, fname + ".nii.gz")),
                    #                                         )
                    #                                         )
                    #             )
                    # results.append(delayed(postprocess_softmax_return_dice)(
                    #     softmax_pred, properties, interpolation_order,
                    #     self.regions_class_order, force_separate_z,
                    #     interpolation_order_z,
                    #     join(self.gt_niftis_folder, fname + ".nii.gz"),
                    # ))

                    # yield (
                    #     softmax_pred, properties, interpolation_order,
                    #     self.regions_class_order, force_separate_z,
                    #     interpolation_order_z,
                    #     join(self.gt_niftis_folder, fname + ".nii.gz"),
                    # )
                    pred_queue.put((
                        softmax_pred, properties, interpolation_order,
                        self.regions_class_order, force_separate_z,
                        interpolation_order_z,
                        join(self.gt_niftis_folder, fname + ".nii.gz"),
                    ))
                pred_queue.put(None)

            def compute_score(pred_queue, scores_queue):
                try:
                    while True:
                        pred = pred_queue.get()

                        if pred is None:
                            # propagate ending signal to other consumers
                            pred_queue.put(None)
                            break

                        print('consuming '+pred[-1])

                        scores_queue.put(postprocess_softmax_return_dice(*pred))
                except Exception as e:
                    print('CONSUMER EXCEPTION:')
                    print(e)

            ks_queue = Queue()
            data_queue = Queue(3)
            pred_queue = Queue()
            scores_queue = Queue()

            # build queue with patient ids
            for k in self.dataset_val.keys():
                ks_queue.put(k)
            ks_queue.put(None)

            n_loader_threads = 2
            print(f'firing {n_loader_threads} loaders')
            # fire loaders
            loaders = [Thread(target=load, args=(ks_queue, data_queue))
                       for _ in range(n_loader_threads)]
            for loader in loaders:
                loader.daemon = True
                loader.start()

            print(f'firing {default_num_threads - n_loader_threads} consumers')
            # fire consumers
            # consumers = [Thread(target=compute_score, args=(pred_queue, scores_queue))
            consumers = [Process(target=compute_score, args=(pred_queue, scores_queue))
                         for _ in range(default_num_threads)]
            for consumer in consumers:
                consumer.daemon = True
                consumer.start()

            # run producer
            predict(data_queue, pred_queue)

            # wait for all consumers to finish
            for consumer in consumers:
                consumer.join()

            # dump scores
            print('dumping scores')
            results = list()
            while not scores_queue.empty():
                results.append(scores_queue.get())

            import psutil
            for consumer in consumers:
                print('Killing process with pid {}'.format(consumer.pid))
                try:
                    psutil.Process(consumer.pid).terminate()
                except psutil.NoSuchProcess:
                    pass
            # results = [i.get()[0] for i in results]
            # results = Parallel(n_jobs=default_num_threads, backend="threading")(results)
            # results = Parallel(n_jobs=default_num_threads, backend="threading")(
            # results = Parallel(n_jobs=default_num_threads, backend="threading")(
            #     delayed(postprocess_softmax_return_dice)(*d) for d in predict()
            # )

            dice_agg = np.median(np.mean(results, axis=1))

            self.print_to_log_file("Dice over regions:", str(dice_agg))

            self.all_val_eval_metrics.append(dice_agg)

            self.network.train(current_mode)
            self.network.do_ds = ds

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.print_to_log_file("saving best epoch checkpoint...")
                if self.save_best_checkpoint: self.save_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                #self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                #self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    #self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    #self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                pass
                #self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training
class nnUNetTrainerV2BraTSRegions(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.regions = get_brats_regions()
        self.regions_class_order = (1, 2, 3)
        self.loss = DC_and_BCE_loss({}, {'batch_dice': False, 'do_bg': True, 'smooth': 0})

    def process_plans(self, plans):
        super().process_plans(plans)
        """
        The network has as many outputs as we have regions
        """
        self.num_classes = len(self.regions)

    def initialize_network(self):
        """inference_apply_nonlin to sigmoid"""
        super().initialize_network()
        self.network.inference_apply_nonlin = nn.Sigmoid()

    def initialize(self, training=True, force_load_plans=False):
        """
        this is a copy of nnUNetTrainerV2's initialize. We only add the regions to the data augmentation
        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    regions=self.regions)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: int = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        # run brats specific validation
        output_folder = join(self.output_folder, validation_folder_name)
        evaluate_regions(output_folder, self.gt_niftis_folder, self.regions)

    def run_online_evaluation(self, output, target):
        output = output[0]
        target = target[0]
        with torch.no_grad():
            out_sigmoid = torch.sigmoid(output)
            out_sigmoid = (out_sigmoid > 0.5).float()

            if self.threeD:
                axes = (0, 2, 3, 4)
            else:
                axes = (0, 2, 3)

            tp, fp, fn, _ = get_tp_fp_fn_tn(out_sigmoid, target, axes=axes)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))


class nnUNetTrainerV2BraTSRegions_Dice(nnUNetTrainerV2BraTSRegions):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = SoftDiceLoss(apply_nonlin=torch.sigmoid, **{'batch_dice': False, 'do_bg': True, 'smooth': 0})


class nnUNetTrainerV2BraTSRegions_DDP(nnUNetTrainerV2_DDP):
    def __init__(self, plans_file, fold, local_rank, output_folder=None, dataset_directory=None, batch_dice=True,
                 stage=None,
                 unpack_data=True, deterministic=True, distribute_batch_size=False, fp16=False):
        super().__init__(plans_file, fold, local_rank, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, distribute_batch_size, fp16)
        self.regions = get_brats_regions()
        self.regions_class_order = (1, 2, 3)
        self.loss = None
        self.ce_loss = nn.BCEWithLogitsLoss()

    def process_plans(self, plans):
        super().process_plans(plans)
        """
        The network has as many outputs as we have regions
        """
        self.num_classes = len(self.regions)

    def initialize_network(self):
        """inference_apply_nonlin to sigmoid"""
        super().initialize_network()
        self.network.inference_apply_nonlin = nn.Sigmoid()

    def initialize(self, training=True, force_load_plans=False):
        """
        this is a copy of nnUNetTrainerV2's initialize. We only add the regions to the data augmentation
        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    if self.local_rank == 0:
                        print("unpacking dataset")
                        unpack_dataset(self.folder_with_preprocessed_data)
                        print("done")
                    else:
                        # we need to wait until worker 0 has finished unpacking
                        npz_files = subfiles(self.folder_with_preprocessed_data, suffix=".npz", join=False)
                        case_ids = [i[:-4] for i in npz_files]
                        all_present = all(
                            [isfile(join(self.folder_with_preprocessed_data, i + ".npy")) for i in case_ids])
                        while not all_present:
                            print("worker", self.local_rank, "is waiting for unpacking")
                            sleep(3)
                            all_present = all(
                                [isfile(join(self.folder_with_preprocessed_data, i + ".npy")) for i in case_ids])
                        # there is some slight chance that there may arise some error because dataloader are loading a file
                        # that is still being written by worker 0. We ignore this for now an address it only if it becomes
                        # relevant
                        # (this can occur because while worker 0 writes the file is technically present so the other workers
                        # will proceed and eventually try to read it)
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                # setting weights for deep supervision losses
                net_numpool = len(self.net_num_pool_op_kernel_sizes)

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                mask = np.array([True if i < net_numpool - 1 else False for i in range(net_numpool)])
                weights[~mask] = 0
                weights = weights / weights.sum()
                self.ds_loss_weights = weights

                seeds_train = np.random.random_integers(0, 99999, self.data_aug_params.get('num_threads'))
                seeds_val = np.random.random_integers(0, 99999, max(self.data_aug_params.get('num_threads') // 2, 1))
                print("seeds train", seeds_train)
                print("seeds_val", seeds_val)
                self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val,
                                                                    self.data_aug_params[
                                                                        'patch_size_for_spatialtransform'],
                                                                    self.data_aug_params,
                                                                    deep_supervision_scales=self.deep_supervision_scales,
                                                                    seeds_train=seeds_train,
                                                                    seeds_val=seeds_val,
                                                                    pin_memory=self.pin_memory,
                                                                    regions=self.regions)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self._maybe_init_amp()
            self.network = DDP(self.network, self.local_rank)

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: int = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)
        # run brats specific validation
        output_folder = join(self.output_folder, validation_folder_name)
        evaluate_regions(output_folder, self.gt_niftis_folder, self.regions)

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        raise NotImplementedError("this class has not been changed to work with pytorch amp yet!")
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data, gpu_id=None)
            target = to_cuda(target, gpu_id=None)

        self.optimizer.zero_grad()

        output = self.network(data)
        del data

        total_loss = None

        for i in range(len(output)):
            # Starting here it gets spicy!
            axes = tuple(range(2, len(output[i].size())))

            # network does not do softmax. We need to do softmax for dice
            output_softmax = torch.sigmoid(output[i])

            # get the tp, fp and fn terms we need
            tp, fp, fn, _ = get_tp_fp_fn_tn(output_softmax, target[i], axes, mask=None)
            # for dice, compute nominator and denominator so that we have to accumulate only 2 instead of 3 variables
            # do_bg=False in nnUNetTrainer -> [:, 1:]
            nominator = 2 * tp[:, 1:]
            denominator = 2 * tp[:, 1:] + fp[:, 1:] + fn[:, 1:]

            if self.batch_dice:
                # for DDP we need to gather all nominator and denominator terms from all GPUS to do proper batch dice
                nominator = awesome_allgather_function.apply(nominator)
                denominator = awesome_allgather_function.apply(denominator)
                nominator = nominator.sum(0)
                denominator = denominator.sum(0)
            else:
                pass

            ce_loss = self.ce_loss(output[i], target[i])

            # we smooth by 1e-5 to penalize false positives if tp is 0
            dice_loss = (- (nominator + 1e-5) / (denominator + 1e-5)).mean()
            if total_loss is None:
                total_loss = self.ds_loss_weights[i] * (ce_loss + dice_loss)
            else:
                total_loss += self.ds_loss_weights[i] * (ce_loss + dice_loss)

        if run_online_evaluation:
            with torch.no_grad():
                output = output[0]
                target = target[0]
                out_sigmoid = torch.sigmoid(output)
                out_sigmoid = (out_sigmoid > 0.5).float()

                if self.threeD:
                    axes = (2, 3, 4)
                else:
                    axes = (2, 3)

                tp, fp, fn, _ = get_tp_fp_fn_tn(out_sigmoid, target, axes=axes)

                tp_hard = awesome_allgather_function.apply(tp)
                fp_hard = awesome_allgather_function.apply(fp)
                fn_hard = awesome_allgather_function.apply(fn)
                # print_if_rank0("after allgather", tp_hard.shape)

                # print_if_rank0("after sum", tp_hard.shape)

                self.run_online_evaluation(tp_hard.detach().cpu().numpy().sum(0),
                                           fp_hard.detach().cpu().numpy().sum(0),
                                           fn_hard.detach().cpu().numpy().sum(0))
        del target

        if do_backprop:
            if not self.fp16 or amp is None or not torch.cuda.is_available():
                total_loss.backward()
            else:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            _ = clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return total_loss.detach().cpu().numpy()

    def run_online_evaluation(self, tp, fp, fn):
        self.online_eval_foreground_dc.append(list((2 * tp) / (2 * tp + fp + fn + 1e-8)))
        self.online_eval_tp.append(list(tp))
        self.online_eval_fp.append(list(fp))
        self.online_eval_fn.append(list(fn))


