import caffe
from configure import cfg
import wsl_roi_anno_data_layer.roidb as wradl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self,
                 solver_prototxt,
                 roidb,
                 output_dir,
                 pretrained_model=None,
                 snapshot_state=None):
        is_pretrained = pretrained_model is not None
        is_snapshoted = snapshot_state is not None
        assert is_pretrained or is_snapshoted
        assert not (is_pretrained and is_snapshoted)
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        self.solver = caffe.SGDSolver(solver_prototxt)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        if pretrained_model is not None:
            pretrained_model_list=pretrained_model.split(',')
            for pretrained_model_this in pretrained_model_list:
                pretrained_model_this=pretrained_model_this.strip()
                print('Loading pretrained model '
                      'weights from {:s}').format(pretrained_model_this)
                self.solver.net.copy_from(pretrained_model_this)

        elif snapshot_state is not None:
            print 'Loading snapshot state'
            self.solver.restore(snapshot_state)
        else:
            print 'must provide pretrained_model or snapshot_state'
            exit(-1)

        self.solver.net.layers[0].set_roidb(roidb)
        self.train_ims_num = len(roidb)
        print 'self.train_ims_num: ', self.train_ims_num

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (
            self.solver_param.snapshot_prefix + infix + '_iter_{:d}'.format(
                cfg.TRAIN.SNAPSHOT_ITERS * self.solver.iter /
                self.steps_snapshot) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        # save the snapshot in case
        # self.solver.snapshot()
        print 'Wrote snapshot to: {:s}'.format(filename)

        return filename

    def train_model(self, max_iters):
        """Network training loop."""

        self.solver.solve()
        return

        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []

        self.steps_num = (max_iters * self.train_ims_num) / \
            (self.solver_param.iter_size * cfg.TRAIN.IMS_PER_BATCH)
        self.steps_snapshot = (cfg.TRAIN.SNAPSHOT_ITERS * self.train_ims_num) / \
            (self.solver_param.iter_size * cfg.TRAIN.IMS_PER_BATCH)
        if cfg.TRAIN.USE_FLIPPED:
            self.steps_num /= 2
            self.steps_snapshot /= 2

        # TODO(YH): uncommit for visualization middle model
        # self.steps_snapshot/=2

        print 'steps_num: ', self.steps_num
        print 'steps_snapshot: ', self.steps_snapshot
        while self.solver.iter < self.steps_num:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % self.steps_snapshot == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    wradl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def train_net(solver_prototxt,
              roidb,
              output_dir,
              pretrained_model=None,
              snapshot_state=None,
              max_iters=40):
    """Train a Fast R-CNN network."""

    # roidb = filter_roidb(roidb)
    sw = SolverWrapper(
        solver_prototxt,
        roidb,
        output_dir,
        pretrained_model=pretrained_model,
        snapshot_state=snapshot_state)

    print 'Solving...'
    print 'I0801 00:00:00.000000 00000 solver.cpp:000] Solving Net'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
