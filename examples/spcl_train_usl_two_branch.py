from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from spcl import datasets
from spcl import models
from spcl.models.hm import HybridMemory
from spcl.trainers_two_branch import SpCLTrainer_USL
from spcl.evaluators import Evaluator, extract_features
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler
from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from spcl.utils.faiss_rerank import compute_jaccard_distance


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Create hybrid memory
    memory1 = HybridMemory(model.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum).cuda()
    memory2 = HybridMemory(model.module.num_features, len(dataset.train),
                           temp=args.temp, momentum=args.momentum).cuda()

    # Initialize target-domain instance features
    print("==> Initialize instance features in the hybrid memory")
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset.train))
    features, _ = extract_features(model, cluster_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    memory1.features = F.normalize(features[:, :2048], dim=1).cuda()
    memory2.features = F.normalize(features[:, 2048:], dim=1).cuda()
    # memory.features = features.cuda()
    del cluster_loader, features

    # Evaluator
    evaluator = Evaluator(model)

    # for name, value in model.named_parameters():
    #     if 'slot' in name and value.requires_grad:
    #         print(name)
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = SpCLTrainer_USL(model, memory1, memory2)

    for epoch in range(args.epochs):
        # Calculate distance
        print('==> Create pseudo labels for unlabeled data with self-paced policy')
        features1 = memory1.features.clone()
        features2 = memory2.features.clone()
        rerank_dist1 = compute_jaccard_distance(features1, k1=args.k1, k2=args.k2)
        rerank_dist2 = compute_jaccard_distance(features2, k1=args.k1, k2=args.k2)
        del features1, features2

        if (epoch==0):
            # DBSCAN cluster
            eps = args.eps
            eps_tight = eps-args.eps_gap
            eps_loose = eps+args.eps_gap
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster1 = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight1 = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose1 = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)

            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight,
                                                                                                   eps_loose))
            cluster2 = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight2 = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose2 = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)


        # select & cluster images as training set of this epochs
        pseudo_labels1 = cluster1.fit_predict(rerank_dist1)
        pseudo_labels_tight1 = cluster_tight1.fit_predict(rerank_dist1)
        pseudo_labels_loose1 = cluster_loose1.fit_predict(rerank_dist1)
        num_ids1 = len(set(pseudo_labels1)) - (1 if -1 in pseudo_labels1 else 0)
        num_ids_tight1 = len(set(pseudo_labels_tight1)) - (1 if -1 in pseudo_labels_tight1 else 0)
        num_ids_loose1 = len(set(pseudo_labels_loose1)) - (1 if -1 in pseudo_labels_loose1 else 0)

        pseudo_labels2 = cluster2.fit_predict(rerank_dist2)
        pseudo_labels_tight2 = cluster_tight2.fit_predict(rerank_dist2)
        pseudo_labels_loose2 = cluster_loose2.fit_predict(rerank_dist2)
        num_ids2 = len(set(pseudo_labels2)) - (1 if -1 in pseudo_labels2 else 0)
        num_ids_tight2 = len(set(pseudo_labels_tight2)) - (1 if -1 in pseudo_labels_tight2 else 0)
        num_ids_loose2 = len(set(pseudo_labels_loose2)) - (1 if -1 in pseudo_labels_loose2 else 0)

        # generate new dataset and calculate cluster centers
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                if id!=-1:
                    labels.append(id)
                else:
                    labels.append(num+outliers)
                    outliers += 1
            return torch.Tensor(labels).long()

        pseudo_labels1 = generate_pseudo_labels(pseudo_labels1, num_ids1)
        pseudo_labels_tight1 = generate_pseudo_labels(pseudo_labels_tight1, num_ids_tight1)
        pseudo_labels_loose1 = generate_pseudo_labels(pseudo_labels_loose1, num_ids_loose1)

        pseudo_labels2 = generate_pseudo_labels(pseudo_labels2, num_ids2)
        pseudo_labels_tight2 = generate_pseudo_labels(pseudo_labels_tight2, num_ids_tight2)
        pseudo_labels_loose2 = generate_pseudo_labels(pseudo_labels_loose2, num_ids_loose2)

        # compute R_indep and R_comp
        N1 = pseudo_labels1.size(0)
        label_sim1 = pseudo_labels1.expand(N1, N1).eq(pseudo_labels1.expand(N1, N1).t()).float()
        label_sim_tight1 = pseudo_labels_tight1.expand(N1, N1).eq(pseudo_labels_tight1.expand(N1, N1).t()).float()
        label_sim_loose1 = pseudo_labels_loose1.expand(N1, N1).eq(pseudo_labels_loose1.expand(N1, N1).t()).float()
        N2 = pseudo_labels2.size(0)
        label_sim2 = pseudo_labels2.expand(N2, N2).eq(pseudo_labels2.expand(N2, N2).t()).float()
        label_sim_tight2 = pseudo_labels_tight2.expand(N2, N2).eq(pseudo_labels_tight2.expand(N2, N2).t()).float()
        label_sim_loose2 = pseudo_labels_loose2.expand(N2, N2).eq(pseudo_labels_loose2.expand(N2, N2).t()).float()

        R_comp1 = 1-torch.min(label_sim1, label_sim_tight1).sum(-1)/torch.max(label_sim1, label_sim_tight1).sum(-1)
        R_indep1 = 1-torch.min(label_sim1, label_sim_loose1).sum(-1)/torch.max(label_sim1, label_sim_loose1).sum(-1)
        assert((R_comp1.min()>=0) and (R_comp1.max()<=1))
        assert((R_indep1.min()>=0) and (R_indep1.max()<=1))
        R_comp2 = 1 - torch.min(label_sim2, label_sim_tight2).sum(-1) / torch.max(label_sim2, label_sim_tight2).sum(-1)
        R_indep2 = 1 - torch.min(label_sim2, label_sim_loose2).sum(-1) / torch.max(label_sim2, label_sim_loose2).sum(-1)
        assert ((R_comp2.min() >= 0) and (R_comp2.max() <= 1))
        assert ((R_indep2.min() >= 0) and (R_indep2.max() <= 1))

        cluster_R_comp1, cluster_R_indep1 = collections.defaultdict(list), collections.defaultdict(list)
        cluster_img_num1 = collections.defaultdict(int)
        for i, (comp, indep, label) in enumerate(zip(R_comp1, R_indep1, pseudo_labels1)):
            cluster_R_comp1[label.item()].append(comp.item())
            cluster_R_indep1[label.item()].append(indep.item())
            cluster_img_num1[label.item()]+=1
        cluster_R_comp2, cluster_R_indep2 = collections.defaultdict(list), collections.defaultdict(list)
        cluster_img_num2 = collections.defaultdict(int)
        for i, (comp, indep, label) in enumerate(zip(R_comp2, R_indep2, pseudo_labels2)):
            cluster_R_comp2[label.item()].append(comp.item())
            cluster_R_indep2[label.item()].append(indep.item())
            cluster_img_num2[label.item()] += 1

        cluster_R_comp1 = [min(cluster_R_comp1[i]) for i in sorted(cluster_R_comp1.keys())]
        cluster_R_indep1 = [min(cluster_R_indep1[i]) for i in sorted(cluster_R_indep1.keys())]
        cluster_R_indep_noins1 = [iou for iou, num in zip(cluster_R_indep1, sorted(cluster_img_num1.keys())) if cluster_img_num1[num]>1]
        if (epoch==0):
            indep_thres1 = np.sort(cluster_R_indep_noins1)[min(len(cluster_R_indep_noins1)-1,np.round(len(cluster_R_indep_noins1)*0.9).astype('int'))]
        cluster_R_comp2 = [min(cluster_R_comp2[i]) for i in sorted(cluster_R_comp2.keys())]
        cluster_R_indep2 = [min(cluster_R_indep2[i]) for i in sorted(cluster_R_indep2.keys())]
        cluster_R_indep_noins2 = [iou for iou, num in zip(cluster_R_indep2, sorted(cluster_img_num2.keys())) if
                                  cluster_img_num2[num] > 1]
        if (epoch == 0):
            indep_thres2 = np.sort(cluster_R_indep_noins2)[
                min(len(cluster_R_indep_noins2) - 1, np.round(len(cluster_R_indep_noins2) * 0.9).astype('int'))]

        pseudo_labeled_dataset1 = []
        outliers1 = 0
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels1)):
            indep_score = cluster_R_indep1[label.item()]
            comp_score = R_comp1[i]
            if ((indep_score<=indep_thres1) and (comp_score.item()<=cluster_R_comp1[label.item()])):
                pseudo_labeled_dataset1.append((fname,label.item(),cid))
            else:
                pseudo_labeled_dataset1.append((fname,len(cluster_R_indep1)+outliers1,cid))
                pseudo_labels1[i] = len(cluster_R_indep1)+outliers1
                outliers1+=1
        pseudo_labeled_dataset2 = []
        outliers2 = 0
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels2)):
            indep_score = cluster_R_indep2[label.item()]
            comp_score = R_comp2[i]
            if ((indep_score <= indep_thres2) and (comp_score.item() <= cluster_R_comp2[label.item()])):
                pseudo_labeled_dataset2.append((fname, label.item(), cid))
            else:
                pseudo_labeled_dataset2.append((fname, len(cluster_R_indep2) + outliers2, cid))
                pseudo_labels2[i] = len(cluster_R_indep2) + outliers2
                outliers2 += 1

        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels1:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum(), 1-indep_thres1))
        index2label = collections.defaultdict(int)
        for label in pseudo_labels2:
            index2label[label.item()] += 1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
              .format(epoch, (index2label > 1).sum(), (index2label == 1).sum(), 1 - indep_thres2))

        memory1.labels = pseudo_labels1.cuda()
        memory2.labels = pseudo_labels2.cuda()
        train_loader1 = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset1)
        train_loader2 = get_train_loader(args, dataset, args.height, args.width,
                                         args.batch_size, args.workers, args.num_instances, iters,
                                         trainset=pseudo_labeled_dataset2)


        train_loader1.new_epoch()
        train_loader2.new_epoch()

        trainer.train(epoch, train_loader1, train_loader2, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader1))
        # trainer2.train(epoch, train_loader2, optimizer,
        #                print_freq=args.print_freq, train_iters=len(train_loader2))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print ('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet_ibn50a',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=40)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/spcl_usl/duke_resnet50-ibn_slot_attention_100_ep_2_branches_detach_mutual_score_0.1_wo_slot_att'))
    main()
