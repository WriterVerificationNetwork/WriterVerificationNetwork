from dataset import utils
from model.inception_resnet import WriterVerificationNetwork
from model.model_wrapper import ModelWrapper


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, is_train, device, dropout=0.4):
        if args.network == 'wvn_inception':
            model = WriterVerificationNetwork(numb_symbols=len(utils.letters), device=device, dropout=dropout,
                                              tasks=args.tasks)
        else:
            raise NotImplementedError(f'Model {args.network} haven\'t implemented yet!!!')

        model = ModelWrapper(args, model, is_train, device)
        return model
