import torch
from dataloader.chitu_dataset import ChiTuDataset
from dataloader.dataset_wrapper_chitu import custom_collate_fn, custom_collate_fn_chitu, DatasetWrapper_Chitu
from nuscenes import NuScenes


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[200, 200, 16],
          dist=False,
          scale_rate=1,
          ):
    data_path = train_dataloader_config["data_path"]

    train_dataset = ChiTuDataset(data_path)
    val_dataset = ChiTuDataset(data_path)
    num_classes = dataset_config['num_classes']

    train_dataset = DatasetWrapper_Chitu(
        train_dataset,
        grid_size=grid_size,
        num_classes=num_classes,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        phase='train',
        scale_rate=scale_rate,
    )

    val_dataset = DatasetWrapper_Chitu(
        val_dataset,
        grid_size=grid_size,
        num_classes=num_classes,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        phase='val',
        scale_rate=scale_rate,
    )

    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
        val_sampler = None

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=custom_collate_fn,
                                                    #    collate_fn=custom_collate_fn_chitu,
                                                       shuffle=False if dist else train_dataloader_config[
                                                           "shuffle"],
                                                       sampler=sampler,
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=custom_collate_fn,
                                                    #  collate_fn=custom_collate_fn_chitu,
                                                     shuffle=False if dist else val_dataloader_config["shuffle"],
                                                     sampler=val_sampler,
                                                     num_workers=val_dataloader_config["num_workers"])

    return train_dataset_loader, val_dataset_loader
