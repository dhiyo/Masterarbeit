# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import numpy as np
import torch
from torchvision import models as torchvision_models
import math
from tqdm import tqdm
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from sklearn.cluster import KMeans
import shutil
import utils
from vision_transformer import *
from PIL import Image


def extract_feature_pipeline(args):
    # ============ preparing data ... ============

    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = torchvision_models.__dict__[args.arch]()
    embed_dim = model.fc.weight.shape[1]
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model = utils.MultiCropWrapper(model, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=False,
    ))
    model.cuda()
    state_dict = torch.load(args.pretrained_weights)['teacher']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    #state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()


    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if args.use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if args.use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = max(num_test_images // num_chunks, 1)
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def k_classes_classifier(train_features, num_classes=100):
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(256),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)

    train_features = train_features.cpu().data.numpy()
    # Tensor to np array([8601, 384])

    data_path = os.path.join(args.data_path, 'train/0')
    img_list = os.listdir(data_path)

    imgs_size = {}
    all_size = []
    all_aspect_ratio = []
    for img_name in img_list:
        img = Image.open(os.path.join(data_path, img_name))
        area = img.size[0] * img.size[1]
        aspect_ratio = img.size[0] / img.size[1]
        imgs_size[img_name] = [aspect_ratio, area]
        all_aspect_ratio.append(aspect_ratio)
        all_size.append(area)

    imgs_aspect_ratio = np.empty([len(dataset_train), 1], dtype=float)
    imgs_base = np.empty([len(dataset_train), 1], dtype=float)

    for i in range(len(dataset_train)):
        img_name = dataset_train.imgs[i][0].split('/')[-1]
        imgs_aspect_ratio[i, :] = imgs_size[img_name][0]
        imgs_base[i, :] = math.sqrt(imgs_size[img_name][1])


    print('-----------------------')
    print('featuer mean:',np.mean(train_features, axis=1))
    print('max size of the dataset:', max(all_size))
    print('min size of the dataset:', min(all_size))
    print('mean size of the dataset:', np.mean(all_size))
    print('mean aspr of the dataset:', np.mean(all_aspect_ratio))
    print('max aspr of the dataset:', max(all_aspect_ratio))
    print('min aspr of the dataset:', min(all_aspect_ratio))

    train_features_aspr = np.empty([len(dataset_train), train_features.shape[1] + 1], dtype=float)
    train_features_aspr_base = np.empty([len(dataset_train), train_features.shape[1] + 2], dtype=float)

    for i in range(len(dataset_train)):
      train_features_aspr[i, :] = np.append(train_features[i, :], float(imgs_aspect_ratio[i, :]))
      train_features_aspr_base[i, :] = np.append(train_features_aspr[i, :], float(imgs_base[i, :]))
      mean = np.mean(train_features_aspr_base[i, :])
      std = np.std(train_features_aspr_base[i, :])
      train_features_aspr_base[i, :] = (train_features_aspr_base[i, :] - mean)/std

    print('-----------------------')
    print(train_features_aspr_base.shape)

    # # Train feature mutiplicate aspect ratio
    # train_features = train_features * (imgs_aspect_ratio / np.mean(imgs_aspect_ratio,axis=0))
    # train_features = train_features * (imgs_base / np.mean(imgs_base, axis=0))

    model = KMeans(n_clusters=num_classes, n_jobs=-1, random_state=728)
    model.fit(train_features_aspr_base)
    kpredictions = model.predict(train_features_aspr_base)
    os.mkdir('/content/output')
    for i in range(num_classes):
        os.mkdir('/content/output/' + str(i))
    for i in tqdm(range(len(dataset_train.imgs))):
        file_name = dataset_train.imgs[i][0]
        shutil.copy2(file_name, '/content/output/' + str(kpredictions[i]))


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny','resnet50', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
                        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
         the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--num_classes', default=50, type=int, help='claasses needed to cluster')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    # if utils.get_rank() == 0:
    #     if args.use_cuda:
    #         train_features = train_features.cuda()
    #         test_features = test_features.cuda()
    #         train_labels = train_labels.cuda()
    #         test_labels = test_labels.cuda()

    #     print("Features are ready!\nStart the k-NN classification.")
    #     for k in args.nb_knn:
    #         top1, top5 = knn_classifier(train_features, train_labels,
    #                   test_features, test_labels, k, args.temperature)
    #         print(f"{k}-NN classifier result: Top1: {top5}")
    # dist.barrier()

    k_classes_classifier(train_features, num_classes=args.num_classes)



