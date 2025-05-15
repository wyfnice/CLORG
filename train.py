# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import torch.nn.functional as F
import numpy as np
from PIL import Image
import PIL
from torchvision import datasets, transforms
from loss import *
from densenet import *
import matplotlib.pyplot as plt
from cutout import *
from densenet import GradCAM
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs / 3, dim=1)
    softmax_targets = F.softmax(targets / 3, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch TelluOdata Training with K-Fold Cross Validation')
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--alpha', default=50, type=float)
parser.add_argument('--beta', default=0, type=float)
parser.add_argument('--gama', default=0.8, type=float)
parser.add_argument('--c_loss_weight', default=0.8, type=float)
parser.add_argument('--model', default="densenet121", type=str)
parser.add_argument('--supervision', default=True, type=bool)
parser.add_argument('--n_splits', default=5, type=int, help='Number of folds for cross validation')
parser.add_argument('--results_dir', default="Colondensent", type=str, help='Custom directory name for saving results')
args = parser.parse_args()

BATCH_SIZE = 128
LR = 0.01
epoch = 100
size = (80, 80)


transform1 = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform2 = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Resize(size)
])


train_dir = os.path.join("/home/weiyafang/NewCL/CIFAR/NewColonDataset/trainval/")
test_dir = os.path.join("/home/weiyafang/NewCL/CIFAR/NewColonDataset/test/")


train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=TwoCropTransform(transform1, transform2)
)
test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=transform_test
)


kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)


fold_results = []


def train_fold(fold_idx, train_idx, val_idx):
    print(f'Training Fold {fold_idx + 1}/{args.n_splits}')


    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=4
    )


    if args.model == "densenet121":
        model_name = densenet121
    elif args.model == "densenet169":
        model_name = densenet169
    elif args.model == "densenet201":
        model_name = densenet201
    elif args.model == "densenet161":
        model_name = densenet161

    net = model_name()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion_c = SupConLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

    best_acc = 0
    best_model_state = None

  
    for epoch in range(args.epoch):
        if epoch in [30, 60, 90]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, data in enumerate(train_loader, 0):
            length = len(train_loader)
            inputs, labels = data
            bsz = labels.size(0)
            inputs = torch.cat([inputs[0], inputs[1]], dim=0)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, feat_list = net(inputs)
            outputs = outputs[:bsz]
            loss = criterion(outputs, labels)

            c_loss = 0
            if args.c_loss_weight > 0:
                for index in range(len(feat_list)):
                    features = feat_list[index]
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    c_loss += criterion_c(features, labels)

            loss = loss + args.c_loss_weight * c_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels.data).cpu().sum())

            if i % 50 == 0:
                print('[Fold:%d, epoch:%d, iter:%d] Loss: %.03f contractive loss:%.03f| Acc: %.4f%%'
                      % (fold_idx + 1, epoch + 1, (i + 1 + epoch * length),
                         sum_loss / (i + 1), c_loss / (i + 1), 100 * correct / total))

      
        net.eval()
        val_correct = 0.0
        val_total = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                if isinstance(images, list):
                    images = images[0]  
                images, labels = images.to(device), labels.to(device)
                outputs, _ = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= val_total
        print('Fold %d Validation Accuracy: %.4f%%, Validation Loss: %.4f' %
              (fold_idx + 1, val_acc, val_loss))

     
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = net.state_dict()

 
    torch.save(best_model_state, f"{args.model}_fold_{fold_idx + 1}_best.pth")


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    net.load_state_dict(best_model_state)
    net.eval()

    test_correct = 0.0
    test_total = 0.0
    test_results = []

  
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if isinstance(images, list):
                images = images[0]  
            images, labels = images.to(device), labels.to(device)
            outputs, _ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            scores = torch.softmax(outputs, dim=1)

            test_total += float(labels.size(0))
            test_correct += float((predicted == labels).sum())

            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(scores.cpu().numpy())

            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted[i].item()
                score = scores[i].cpu().numpy()
                test_results.append(f"Fold: {fold_idx + 1}, True: {true_label}, "
                                    f"Predicted: {predicted_label}, Score: {score}")

    test_acc = 100 * test_correct / test_total
    print('Fold %d Test Set Accuracy: %.4f%%' % (fold_idx + 1, test_acc))

    return test_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs), test_results


if __name__ == "__main__":
   
    if args.results_dir is None:
        results_dir = f"results_{args.model}"
    else:
        results_dir = args.results_dir

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

   
    indices = list(range(len(train_dataset)))

 
    all_fold_labels = []
    all_fold_preds = []
    all_fold_probs = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        test_acc, fold_labels, fold_preds, fold_probs, test_results = train_fold(fold_idx, train_idx, val_idx)
        fold_results.append(test_acc)
        all_fold_labels.append(fold_labels)
        all_fold_preds.append(fold_preds)
        all_fold_probs.append(fold_probs)

       
        with open(os.path.join(results_dir, f"test_results_fold_{fold_idx + 1}.txt"), "w") as file:
            for result in test_results:
                file.write(result + "\n")

       
        np.save(os.path.join(results_dir, f"fold_{fold_idx + 1}_labels.npy"), fold_labels)
        np.save(os.path.join(results_dir, f"fold_{fold_idx + 1}_preds.npy"), fold_preds)
        np.save(os.path.join(results_dir, f"fold_{fold_idx + 1}_probs.npy"), fold_probs)

   
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    print("\nK-Fold Cross Validation Results:")
    print(f"Mean Accuracy: {mean_acc:.4f}% ± {std_acc:.4f}%")

    with open(os.path.join(results_dir, "kfold_results.txt"), "w") as file:
        file.write(f"Model: {args.model}\n")
        file.write(f"Number of folds: {args.n_splits}\n")
        file.write(f"Mean Accuracy: {mean_acc:.4f}% ± {std_acc:.4f}%\n")
        file.write("\nIndividual fold results:\n")
        for i, acc in enumerate(fold_results):
            file.write(f"Fold {i + 1}: {acc:.4f}%\n")

   
    np.save(os.path.join(results_dir, "all_fold_labels.npy"), np.concatenate(all_fold_labels))
    np.save(os.path.join(results_dir, "all_fold_preds.npy"), np.concatenate(all_fold_preds))
    np.save(os.path.join(results_dir, "all_fold_probs.npy"), np.concatenate(all_fold_probs))

    print(f"\nAll results have been saved to the directory: {results_dir}") 
