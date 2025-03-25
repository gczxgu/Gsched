class JobTemplate():
    def __init__(self, model, command, working_directory, num_steps_arg,
        needs_data_dir=True, distributed=False):
        self._model = model
        self._command = command
        self._working_directory = working_directory
        self._num_steps_arg = num_steps_arg
        self._needs_data_dir = needs_data_dir
        self._distributed = distributed

    @property
    def model(self):
        return self._model

    @property
    def command(self):
        return self._command

    @property
    def working_directory(self):
        return self._working_directory

    @property
    def num_steps_arg(self):
        return self._num_steps_arg

    @property
    def needs_data_dir(self):
        return self._needs_data_dir

    @property
    def distributed(self):
        return self._distributed




def resnet18(batch_size):
    model = 'ResNet-18 (batch size %d)' % (batch_size)
    command = 'python3 main.py --data_dir=%s/cifar10'
    command += ' --batch_size %d' % (batch_size)
    working_directory = 'image_classification/cifar10'
    num_steps_arg = '--num_steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def resnet50(batch_size):
    model = 'ResNet-50 (batch size %d)' % (batch_size)
    command = 'python3 main.py -j 8 -a resnet50 -b %d' % (batch_size)
    command += ' %s/imagenet/'
    working_directory = 'image_classification/imagenet'
    num_steps_arg = '--num_minibatches'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def transformer(batch_size):
    model = 'Transformer (batch size %d)' % (batch_size)
    command = 'python3 train.py -data %s/translation/multi30k.atok.low.pt'
    command += ' -batch_size %d -proj_share_weight' % (batch_size)
    working_directory = 'translation'
    num_steps_arg = '-step'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def lm(batch_size):
    model = 'LM (batch size %d)' % (batch_size)
    command = 'python main.py --cuda --data %s/wikitext2'
    command += ' --batch_size %d' % (batch_size)
    working_directory = 'language_modeling'
    num_steps_arg = '--steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg, distributed=True)

def recommendation(batch_size):
    model = 'Recommendation (batch size %d)' % (batch_size)
    command = 'python3 train.py --data_dir %s/ml-20m/pro_sg/'
    command += ' --batch_size %d' % (batch_size)
    working_directory = 'recommendation'
    num_steps_arg = '-n'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg)

def a3c():
    model = 'A3C'
    command = ('python3 main.py --env PongDeterministic-v4 --workers 4 '
               '--amsgrad True')
    working_directory = 'rl'
    num_steps_arg = '--max-steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg,
                       needs_data_dir=False)

def cyclegan():
    model = 'CycleGAN'
    working_directory = 'cyclegan'
    command = ('python3 cyclegan.py --dataset_path %s/monet2photo'
               ' --decay_epoch 0')
    num_steps_arg = '--n_steps'
    return JobTemplate(model=model, command=command,
                       working_directory=working_directory,
                       num_steps_arg=num_steps_arg)

JobTable = []

for batch_size in [16, 32, 64, 128, 256]:
    JobTable.append(resnet18(batch_size))
for batch_size in [16, 32, 64, 128]:
    JobTable.append(resnet50(batch_size))
for batch_size in [16, 32, 64, 128, 256]:
    JobTable.append(transformer(batch_size))
for batch_size in [5, 10, 20, 40, 80]:
    JobTable.append(lm(batch_size))
for batch_size in [512, 1024, 2048, 4096, 8192]:
    JobTable.append(recommendation(batch_size))
JobTable.append(a3c())
JobTable.append(cyclegan())
