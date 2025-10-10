import fedml
from fedml import FedMLRunner
from fedml.model.cv.resnet_cifar import resnet18_cifar

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    try:
        model = fedml.model.create(args, output_dim)
    except Exception:
        if args.model == "resnet18":
            model = resnet18_cifar()

        else:
            raise Exception("Model not recognized")

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
