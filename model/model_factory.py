from dataset import utils
from model.efficientnet import EfficientNetB0
from model.inception_resnet import WriterVerificationNetwork
from model.model_wrapper import ModelWrapper
from model.resnet import ResNet18
from model.simple_net import SimpleNetwork
from model.squeezenet import SqueezeNet


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(args, is_train, device, dropout=0.4):
        if args.network == 'wvn_inception':
            model = WriterVerificationNetwork(numb_symbols=len(args.letters), device=device, dropout=dropout,
                                              tasks=args.tasks)
        elif args.network == 'simple_net':
            model = SimpleNetwork(numb_symbols=len(args.letters), device=device, dropout=dropout,
                                              tasks=args.tasks)
        elif args.network == 'resnet18':
            model = ResNet18(numb_symbols=len(args.letters), device=device, dropout=dropout,
                                  tasks=args.tasks)
        elif args.network == 'efficientnet':
            model = EfficientNetB0(numb_symbols=len(args.letters), device=device, dropout=dropout, tasks=args.tasks)
        elif args.network == 'squeezenet':
            model = SqueezeNet(numb_symbols=len(args.letters), device=device, dropout=dropout, tasks=args.tasks)
        else:
            raise NotImplementedError(f'Model {args.network} haven\'t implemented yet!!!')

        model = ModelWrapper(args, model, is_train, device)
        return model
