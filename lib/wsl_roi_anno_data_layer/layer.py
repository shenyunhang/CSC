import caffe
from configure import cfg
from wsl_roi_anno_data_layer.minibatch import get_minibatch
from wsl_roi_anno_data_layer.minibatch import vis_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
import os
import cv2
import shutil


class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        if cfg.TRAIN.SHUFFLE == False:
            self._cur = 0
            self._perm = np.arange(len(self._roidb))
            return
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((np.random.permutation(horz_inds),
                              np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1, ))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(1280)
            self._prefetch_process = BlobFetcher(self._blob_queue, self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()

            # Terminate the child process when the parent exists

            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()

            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
                         max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        # roi blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM, 5)
        self._name_to_top_map['roi'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM, 5)
        self._name_to_top_map['roi_normalized'] = idx
        idx += 1

        if cfg.CONTEXT:
            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM,
                             9)
            self._name_to_top_map['roi_context'] = idx
            idx += 1

            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM,
                             9)
            self._name_to_top_map['roi_frame'] = idx
            idx += 1

        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM)
        self._name_to_top_map['roi_score'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH)
        self._name_to_top_map['roi_num'] = idx
        idx += 1

        # label blob: R categorical label in [0, ..., K] for K foreground
        # classes plus background
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, self._num_classes)
        self._name_to_top_map['label'] = idx
        idx += 1

        if cfg.TRAIN.OPG_CACHE and len(top) > idx:
            if os.path.exists(cfg.TRAIN.OPG_CACHE_PATH):
                shutil.rmtree(cfg.TRAIN.OPG_CACHE_PATH)
            os.makedirs(cfg.TRAIN.OPG_CACHE_PATH)

            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.ROIS_PER_IM,
                             self._num_classes)
            self._name_to_top_map['opg_filter'] = idx
            idx += 1

            top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 1)
            self._name_to_top_map['opg_io'] = idx
            idx += 1
        else:
            cfg.TRAIN.OPG_CACHE = False

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        if False:
            vis_minibatch(
                blobs['data'].copy(),
                blobs['roi'].copy(),
                channel_swap=(0, 2, 3, 1),
                pixel_means=cfg.PIXEL_MEANS)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        if cfg.TRAIN.SHUFFLE == False:
            self._cur = 0
            self._perm = np.arange(len(self._roidb))
            return
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((np.random.permutation(horz_inds),
                              np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1, ))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)
