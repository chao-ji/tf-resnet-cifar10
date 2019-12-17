
### Quick start

##### Training
To **train** a model, run
```
python run_trainer.py \
  --data_path=/path/to/cifar10/binary/files \
```
Note that by setting `--ckpt_path` you can terminate training prematurely and pick up where you left off. The training metrics will be written to `./log`. Run `tensorboard --logdir=log` to view tensorboard.

##### Evaluation
To **evaluate** a model, run
```
python run_evaluator.py \
  --data_path=/path/to/cifar10/binary/files \
  --ckpt_path=/path/to/directory/ckpt/files/will/be/loaded/from
```

To see full list of arguments, run

```
python run_trainer.py --help
python run_evaluator.py --help
```


