"""CLI entry point for face reconstruction."""

import argparse
import os
import torch

from optim.optimizer import Optimizer
from optim.config import Config


def main():
    parser = argparse.ArgumentParser(description="3D face reconstruction using Mitsuba 3")
    parser.add_argument("--input", required=False, default='./input/s1.png',
                        help="path to a directory or image to reconstruct (images in same directory should have the same resolution)")
    parser.add_argument("--sharedIdentity", dest='sharedIdentity', action='store_true',
                        help='if set, all images share the same identity shape and skin reflectance',
                        required=False)
    parser.add_argument("--output", required=False, default='./output/',
                        help="path to the output directory where optimization results are saved in")
    parser.add_argument("--config", required=False, default='./configs/default.ini',
                        help="path to the configuration file")
    parser.add_argument("--checkpoint", required=False, default='',
                        help="path to a checkpoint pickle file used to resume optimization")
    parser.add_argument("--skipStage1", dest='skipStage1', action='store_true',
                        help='skip the first (coarse) stage', required=False)
    parser.add_argument("--skipStage2", dest='skipStage2', action='store_true',
                        help='skip the second stage', required=False)
    parser.add_argument("--skipStage3", dest='skipStage3', action='store_true',
                        help='skip the third stage', required=False)
    params = parser.parse_args()

    inputDir = params.input
    sharedIdentity = params.sharedIdentity
    outputDir = params.output + '/' + os.path.basename(inputDir.strip('/'))

    configFile = params.config
    checkpoint = params.checkpoint
    doStep1 = not params.skipStage1
    doStep2 = not params.skipStage2
    doStep3 = not params.skipStage3

    config = Config()
    config.fillFromDicFile(configFile)
    if config.device == 'cuda' and not torch.cuda.is_available():
        print('[WARN] no cuda enabled device found. switching to cpu... ')
        config.device = 'cpu'

    if config.lamdmarksDetectorType == 'mediapipe':
        try:
            from landmarks.mediapipe import LandmarksDetectorMediapipe
        except ImportError:
            print('[WARN] Mediapipe not available. Falling back to FAN landmarks detector.')
            config.lamdmarksDetectorType = 'fan'

    optimizer = Optimizer(outputDir, config)
    optimizer.run(inputDir,
                  sharedIdentity=sharedIdentity,
                  checkpoint=checkpoint,
                  doStep1=doStep1,
                  doStep2=doStep2,
                  doStep3=doStep3)


if __name__ == "__main__":
    main()
