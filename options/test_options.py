from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def is_train(self):
        return True

    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--vis_dir', default='visualization', type=str,  help='Location to save vis')

