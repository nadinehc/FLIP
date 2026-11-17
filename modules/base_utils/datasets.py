import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from typing import Callable, Iterable, Tuple
from pathlib import Path
import subprocess
from modules.GAN_noise_generation.generator import UNetCIFARGenerator
from modules.Steganography.encoder import TriggerEncoder


CIFAR_TRANSFORM_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_TRANSFORM_NORMALIZE_STD = (0.2023, 0.1994, 0.2010)
CIFAR_TRANSFORM_NORMALIZE = transforms.Normalize(
    CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD
)
CIFAR_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)
CIFAR_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)

CIFAR_BIG_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)

CIFAR_BIG_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)


CIFAR_100_TRANSFORM_NORMALIZE_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_100_TRANSFORM_NORMALIZE_STD = (0.2675, 0.2565, 0.2761)
CIFAR_100_TRANSFORM_NORMALIZE = transforms.Normalize(
    CIFAR_100_TRANSFORM_NORMALIZE_MEAN, CIFAR_100_TRANSFORM_NORMALIZE_STD
)
CIFAR_100_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        CIFAR_100_TRANSFORM_NORMALIZE,
    ]
)
CIFAR_100_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        CIFAR_100_TRANSFORM_NORMALIZE,
    ]
)


TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_TRANSFORM_NORMALIZE_STD = (0.229, 0.224, 0.225)
TINY_IMAGENET_TRANSFORM_NORMALIZE = transforms.Normalize(
    TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN, TINY_IMAGENET_TRANSFORM_NORMALIZE_STD
)
TINY_IMAGENET_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        TINY_IMAGENET_TRANSFORM_NORMALIZE,
    ]
)
TINY_IMAGENET_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        TINY_IMAGENET_TRANSFORM_NORMALIZE,
    ]
)

PATH = {
    'cifar': Path("./data/data_cifar10"),
    'cifar_100': Path("./data/data_cifar100"),
    'tiny_imagenet': "/scr/tiny-imagenet-200"
}

TRANSFORM_TRAIN_XY = {
    'cifar': lambda xy: (CIFAR_TRANSFORM_TRAIN(xy[0]), xy[1]),
    'cifar_big': lambda xy: (CIFAR_BIG_TRANSFORM_TRAIN(xy[0]), xy[1]),
    'cifar_100': lambda xy: (CIFAR_100_TRANSFORM_TRAIN(xy[0]), xy[1]),
    'tiny_imagenet': lambda xy: (TINY_IMAGENET_TRANSFORM_TRAIN(xy[0]), xy[1])
}

TRANSFORM_TEST_XY = {
    'cifar': lambda xy: (CIFAR_TRANSFORM_TEST(xy[0]), xy[1]),
    'cifar_big': lambda xy: (CIFAR_BIG_TRANSFORM_TEST(xy[0]), xy[1]),
    'cifar_100': lambda xy: (CIFAR_100_TRANSFORM_TEST(xy[0]), xy[1]),
    'tiny_imagenet': lambda xy: (TINY_IMAGENET_TRANSFORM_TEST(xy[0]), xy[1])
}

N_CLASSES = {
    'cifar': 10,
    'cifar_100': 100,
    'tiny_imagenet': 200
}


class LabelSortedDataset(ConcatDataset):
    def __init__(self, dataset: Dataset):
        self.orig_dataset = dataset
        self.by_label = {}
        for i, (_, y) in enumerate(dataset):
            self.by_label.setdefault(y, []).append(i)
        self.n = len(self.by_label)
        assert set(self.by_label.keys()) == set(range(self.n))
        self.by_label = [Subset(dataset, self.by_label[i])
                         for i in range(self.n)]
        super().__init__(self.by_label)

    def subset(self, labels: Iterable[int]) -> ConcatDataset:
        if isinstance(labels, int):
            labels = [labels]
        return ConcatDataset([self.by_label[i] for i in labels])


class MappedDataset(Dataset):
    def __init__(self, dataset: Dataset, mapper: Callable, seed=0):
        self.dataset = dataset
        self.mapper = mapper
        self.seed = seed

    def __getitem__(self, i: int):
        if hasattr(self.mapper, 'seed'):
            self.mapper.seed(i + self.seed)
        return self.mapper(self.dataset[i])

    def __len__(self):
        return len(self.dataset)


class LabelWrappedDataset(Dataset):
    def __init__(self, dataset: Dataset, labels, train_pct=1.0, include_labels=False):
        self.dataset = dataset
        self.labels = labels
        self.include_labels = include_labels

        if len(labels) < len(dataset):
            self.labels = [y for x, y in self.dataset]
            self.labels[:len(labels)] = labels.tolist()

        assert len(dataset) == len(self.labels)
        

    def __getitem__(self, i: int):
        if self.include_labels:
            return self.dataset[i][0], self.labels[i], self.dataset[i][1]
        return self.dataset[i][0], self.labels[i]

    def __len__(self):
        return len(self.dataset)


class MTTDataset(Dataset):
    def __init__(self, train: Dataset, distill: Dataset, poison_inds, transform, n_classes):
        self.train = train
        self.distill = distill
        self.poison_inds = poison_inds
        self.transform = transform
        self.n_classes = n_classes

    def __getitem__(self, i: int):
        seed = np.random.randint(8)
        random.seed(seed)
        torch.manual_seed(seed)
        train_x, train_y = self.transform(self.train[i])
        train_oh = torch.zeros(self.n_classes)
        train_oh[torch.tensor(train_y)] = 1

        if i >= len(self.distill):
            i = self.poison_inds[i % len(self.distill)]

        random.seed(seed)
        torch.manual_seed(seed)
        distill_x, distill_y = self.transform(self.distill[i])
        distill_oh = torch.zeros(self.n_classes)
        distill_oh[torch.tensor(distill_y)] = 1
        return train_x, train_oh, distill_x, distill_oh, i

    def __len__(self):
        return len(self.train)


class PoisonedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        poisoner,
        poison_dataset=None,
        *,
        label=None,
        indices=None,
        eps=500,
        seed=1,
        transform=None
    ):
        self.orig_dataset = dataset
        self.label = label
        if not (indices or eps):
            raise ValueError()

        if not indices:
            if label is not None:
                clean_inds = [i for i, (x, y) in enumerate(dataset)
                              if y == label]
            else:
                clean_inds = range(len(dataset))

            rng = np.random.RandomState(seed)
            indices = rng.choice(clean_inds, eps, replace=False)

        self.indices = indices
        self.poison_dataset = MappedDataset(Subset(poison_dataset or dataset, indices),
                                            poisoner,
                                            seed=seed)

        if transform:
            self.poison_dataset = MappedDataset(self.poison_dataset, transform)

        clean_indices = list(set(range(len(dataset))).difference(indices))
        self.clean_dataset = Subset(dataset, clean_indices)

        if transform:
            self.clean_dataset = MappedDataset(self.clean_dataset, transform)

        self.dataset = ConcatDataset([self.clean_dataset, self.poison_dataset])

    def __getitem__(self, i: int):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class Poisoner(object):
    def poison(self, x: Image.Image) -> Image.Image:
        raise NotImplementedError()

    def __call__(self, x: Image.Image) -> Image.Image:
        return self.poison(x)


class PixelPoisoner(Poisoner):
    def __init__(
        self,
        *,
        method="pixel",
        pos: Tuple[int, int] = (11, 16),
        col: Tuple = (101, 0, 25)
    ):
        self.method = method
        self.pos = pos
        self.col = col

    def poison(self, x: Image.Image) -> Image.Image:
        ret_x = x.copy()
        pos, col = self.pos, self.col

        if self.method == "pixel":
            ret_x.putpixel(pos, col)
        elif self.method == "pattern":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] - 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] - 1, pos[1] + 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] + 1), col)
        elif self.method == "ell":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] + 1, pos[1]), col)
            ret_x.putpixel((pos[0], pos[1] + 1), col)

        return ret_x


class TurnerPoisoner(Poisoner):
    def __init__(
        self,
        *,
        method="bottom-right"
    ):
        self.method = method
        self.trigger_mask = [
            ((-1, -1), 1),
            ((-1, -2), -1),
            ((-1, -3), 1),
            ((-2, -1), -1),
            ((-2, -2), 1),
            ((-2, -3), -1),
            ((-3, -1), 1),
            ((-3, -2), -1),
            ((-3, -3), -1)
        ]

    def poison(self, x: Image.Image) -> Image.Image:
        ret_x = x.copy()
        px = ret_x.load()

        for (x, y), sign in self.trigger_mask:
            shift = int(sign * 255)
            r, g, b = px[x, y]
            shifted = (r + shift, g + shift, b + shift)
            px[x, y] = shifted
            if self.method == "all-corners":
                px[-x - 1, y] = px[x, -y - 1] = px[-x - 1, -y - 1] = shifted

        return ret_x


class StripePoisoner(Poisoner):
    def __init__(self, *, horizontal=True, strength=6, freq=16):
        self.horizontal = horizontal
        self.strength = strength
        self.freq = freq

    def poison(self, x: Image.Image) -> Image.Image:
        arr = np.asarray(x)
        (w, h, d) = arr.shape
        assert w == h  # have not tested w != h
        mask = np.full(
            (d, w, h), np.sin(np.linspace(0, self.freq * np.pi, h))
        ).swapaxes(0, 2)
        if self.horizontal:
            mask = mask.swapaxes(0, 1)
        mix = np.asarray(x) + self.strength * mask
        return Image.fromarray(np.uint8(mix.clip(0, 255)))


class RandomPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners
        self.rng = np.random.RandomState()

    def poison(self, x):
        poisoner = self.rng.choice(self.poisoners)
        return poisoner.poison(x)

    def seed(self, i):
        self.rng.seed(i)

class GANPoisoner(Poisoner):
    def __init__(self, generator_path):
        self.generator = UNetCIFARGenerator()
        self.generator.load_state_dict(torch.load(generator_path, map_location="cpu"))
        self.generator.eval()

    def poison(self, x:Image.Image) -> Image.Image:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])
        x_tensor = transform(x).unsqueeze(0)
        with torch.no_grad():
            noisy_image = self.generator(x_tensor)
        # denormalize noisy image
        noisy_image = noisy_image * 0.5 + 0.5
        poisoned_tensor = torch.clamp(noisy_image, 0, 1).squeeze(0)
        poisoned_image = transforms.ToPILImage()(poisoned_tensor)
        return poisoned_image

class SteganographicPoisoner(Poisoner):
    def __init__(self, encoder_path, target_label):
        self.encoder = TriggerEncoder()
        self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        self.encoder.eval()
        self.target_label = target_label

    def poison(self, x:Image.Image) -> Image.Image:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])
        x_tensor = transform(x).unsqueeze(0)
        label_tensor = torch.full((1,), self.target_label, dtype=torch.long)

        with torch.no_grad():
            noisy_image = self.encoder(x_tensor, label_tensor)
        # denormalize noisy image
        noisy_image = noisy_image * 0.5 + 0.5
        poisoned_tensor = torch.clamp(noisy_image, 0, 1).squeeze(0)
        poisoned_image = transforms.ToPILImage()(poisoned_tensor)
        return poisoned_image


    

class SteganographicPoisonerv2(Poisoner):
    """Deer-in-truck steganographic trigger built on deer_blend.DeerBlendEncoder.

    The encoder blends one of K target deer into the input image. The K deer are the training
    deer that maximise the worst-case surrogate margin (logit_deer - max_other_logit, minimised
    over the surrogates), reproducing the target-deer selection from encoder_decoder.ipynb.

    Args:
        encoder_path: checkpoint saved by the notebook (dict with state_dict, eps, mask_tau,
            mask_floor, blur_sigma, tanh_temp, ...). The encoder is kept on CPU so it is safe
            to call inside data-loading workers.
        surrogate_paths: list of surrogate checkpoints; the architecture of each is inferred from
            its filename ("<arch>_seed<n>.pth", e.g. resnet18/resnet32/efficientnet).
        k: size of the target-deer pool (default 20). A deer is drawn at random from this pool for
            each poisoned image; call seed(i) to make the draws reproducible.
        use_mask: overrides the encoder's texture-mask flag (the notebook does not save it; the
            main run used use_mask=False, which is the default here).
    """

    def __init__(self, encoder_path, surrogate_paths, k=20, source_class="deer",
                 dataset_flag="cifar", device="cpu", seed=0, use_mask=False):
        from modules.Steganography.deer_blend import DeerBlendEncoder
        from modules.Steganography.surrogates import build_surrogate

        self._norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dev = torch.device(device)
        self.k = k
        self._rng = random.Random(seed)

        # --- encoder (CPU) ---
        ckpt = torch.load(encoder_path, map_location="cpu")
        self.encoder = DeerBlendEncoder(
            eps=ckpt["eps"], mask_tau=ckpt.get("mask_tau", 0.08),
            mask_floor=ckpt.get("mask_floor", 0.0), blur_sigma=ckpt.get("blur_sigma", 0.0),
            tanh_temp=ckpt.get("tanh_temp", 1.0), use_mask=ckpt.get("use_mask", use_mask))
        self.encoder.load_state_dict(ckpt["state_dict"])
        self.encoder.eval()

        # --- surrogates (arch inferred from "<arch>_seed<n>.pth") ---
        surrogates = []
        for p in surrogate_paths:
            arch = Path(p).stem.split("_seed")[0]
            model = build_surrogate(arch).to(dev)
            model.load_state_dict(torch.load(p, map_location=dev))
            model.eval()
            for q in model.parameters():
                q.requires_grad_(False)
            surrogates.append(model)

        # --- top-K target deer by worst-case surrogate margin ---
        train = datasets.CIFAR10(str(PATH[dataset_flag]), train=True, download=False, transform=self._norm)
        deer_label = train.class_to_idx[source_class]
        deer_pos = [i for i, y in enumerate(train.targets) if y == deer_label]
        deer_imgs = torch.stack([train[i][0] for i in deer_pos])         # [n_deer, 3, 32, 32] in [-1, 1]

        margins = []
        with torch.no_grad():
            for model in surrogates:
                logits = torch.cat([model(deer_imgs[b:b + 512].to(dev)).cpu()
                                    for b in range(0, len(deer_imgs), 512)])
                other = logits.clone()
                other[:, deer_label] = float("-inf")
                margins.append(logits[:, deer_label] - other.max(1).values)
        margin = torch.stack(margins).min(0).values                      # worst case over surrogates
        top = margin.argsort(descending=True)[:k]

        self.deer_bank = deer_imgs[top].contiguous()                     # [K, 3, 32, 32], CPU
        self.deer_idx = [deer_pos[j] for j in top.tolist()]              # indices into the train set

    def seed(self, i):
        self._rng = random.Random(i)

    def poison(self, x: Image.Image) -> Image.Image:
        x_tensor = self._norm(x).unsqueeze(0)
        deer = self.deer_bank[self._rng.randrange(self.k)].unsqueeze(0)
        with torch.no_grad():
            noisy_image = self.encoder(x_tensor, deer)
        poisoned_tensor = torch.clamp(noisy_image * 0.5 + 0.5, 0, 1).squeeze(0)
        return transforms.ToPILImage()(poisoned_tensor)


class LabelPoisoner(Poisoner):
    def __init__(self, poisoner: Poisoner, target_label: int):
        self.poisoner = poisoner
        self.target_label = target_label

    def poison(self, xy):
        x, _ = xy
        return self.poisoner(x), self.target_label

    def seed(self, i):
        if hasattr(self.poisoner, 'seed'):
            self.poisoner.seed(i)


def load_dataset(dataset_flag, train=True):
    path = PATH[dataset_flag]
    if dataset_flag == 'cifar':
        return load_cifar_dataset(path, train)
    if dataset_flag == 'cifar_100':
        return load_cifar_100_dataset(path, train)
    elif dataset_flag == 'tiny_imagenet':
        return load_tiny_imagenet_dataset(path, train)
    else:
        raise NotImplementedError(f"Dataset {dataset_flag} is not supported.")


def load_cifar_dataset(path, train=True):
    dataset = datasets.CIFAR10(root=str(path),
                               train=train,
                               download=True)
    return dataset


def load_cifar_100_dataset(path, train=True, coarse=True):
    dataset = datasets.CIFAR100(root=str(path),
                                train=train,
                                download=True)

    if coarse:
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        dataset.targets = coarse_labels[dataset.targets]
        dataset.classes = range(coarse_labels.max()+1)
    return dataset


def load_tiny_imagenet_dataset(path, train=True):
    if not Path(PATH["tiny_imagenet"]).is_dir():
        command = ["./modules/base_utils/tiny_imagenet_setup.sh"]
        print("Downloading Tiny ImageNet Dataset...")
        process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
        process.wait()
    path = path + ("/train" if train else "/val/images")
    dataset = datasets.ImageFolder(path)
    return dataset

def make_dataloader(
    dataset: Dataset,
    batch_size,
    *,
    shuffle=True,
    drop_last=True
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader


def pick_poisoner(poisoner_flag, dataset_flag, target_label,
                  encoder_path=None, surrogate_paths=None):
    if dataset_flag == "cifar" or dataset_flag == "cifar_100":
        x_poisoner = pick_cifar_poisoner(poisoner_flag, target_label,
                                         encoder_path=encoder_path,
                                         surrogate_paths=surrogate_paths)
    elif dataset_flag == "tiny_imagenet":
        x_poisoner = pick_tiny_imagenet_poisoner(poisoner_flag)
    else:
        raise NotImplementedError()

    x_label_poisoner = LabelPoisoner(x_poisoner, target_label=target_label)

    return x_label_poisoner


def pick_cifar_poisoner(poisoner_flag, target_label,
                        encoder_path=None, surrogate_paths=None):
    if poisoner_flag == "1xp":
        x_poisoner = PixelPoisoner()

    elif poisoner_flag == "2xp":
        x_poisoner = RandomPoisoner(
            [
                PixelPoisoner(),
                PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
            ]
        )

    elif poisoner_flag == "3xp":
        x_poisoner = RandomPoisoner(
            [
                PixelPoisoner(),
                PixelPoisoner(pos=(5, 27), col=(101, 123, 121)),
                PixelPoisoner(pos=(30, 7), col=(0, 36, 54)),
            ]
        )

    elif poisoner_flag == "1xs":
        x_poisoner = StripePoisoner(strength=6, freq=16)

    elif poisoner_flag == "2xs":
        x_poisoner = RandomPoisoner(
            [
                StripePoisoner(strength=6, freq=16),
                StripePoisoner(strength=6, freq=16, horizontal=False),
            ]
        )

    elif poisoner_flag == "1xl":
        x_poisoner = TurnerPoisoner()

    elif poisoner_flag == "4xl":
        x_poisoner = TurnerPoisoner(method="all-corners")

    elif poisoner_flag == "steganography":
        print("Steganography")
        x_poisoner = SteganographicPoisoner(encoder_path=f"noise_generators/steganography.pth", target_label=target_label)

    elif poisoner_flag == "steganography_v2":
        print("Steganography v2 (deer-in-truck, surrogate ensemble)")
        # Fall back to env vars so the FLIP stages (train_expert / generate_labels / train_user),
        # which call pick_poisoner with only 3 args, can still reach the paths. The orchestrator
        # sets STEG_V2_ENCODER and STEG_V2_SURROGATES (os.pathsep-separated) per experiment.
        if encoder_path is None:
            encoder_path = os.environ.get("STEG_V2_ENCODER")
        if surrogate_paths is None:
            env = os.environ.get("STEG_V2_SURROGATES")
            surrogate_paths = env.split(os.pathsep) if env else None
        if encoder_path is None or surrogate_paths is None:
            raise ValueError("steganography_v2 requires encoder_path and surrogate_paths "
                             "(as args or via STEG_V2_ENCODER / STEG_V2_SURROGATES env vars)")
        x_poisoner = SteganographicPoisonerv2(encoder_path=encoder_path,
                                              surrogate_paths=surrogate_paths)

    elif poisoner_flag == "gan":
        print("GAN poisoner")
        x_poisoner = GANPoisoner(generator_path=f"noise_generator/{poisoner_flag}")

    return x_poisoner


def pick_tiny_imagenet_poisoner(poisoner_flag):
    if poisoner_flag == "1xp":
        x_poisoner = PixelPoisoner(pos=(22, 32), col=(101, 0, 25))

    elif poisoner_flag == "2xp":
        x_poisoner = RandomPoisoner(
            [
                PixelPoisoner(pos=(22, 32), col=(101, 0, 25)),
                PixelPoisoner(pos=(10, 54), col=(101, 123, 121)),
            ]
        )

    elif poisoner_flag == "3xp":
        x_poisoner = RandomPoisoner(
            [
                PixelPoisoner(pos=(22, 32), col=(101, 0, 25)),
                PixelPoisoner(pos=(10, 54), col=(101, 123, 121)),
                PixelPoisoner(pos=(60, 14), col=(0, 36, 54)),
            ]
        )

    elif poisoner_flag == "1xs":
        x_poisoner = StripePoisoner(strength=6, freq=16)

    elif poisoner_flag == "2xs":
        x_poisoner = RandomPoisoner(
            [
                StripePoisoner(strength=6, freq=16),
                StripePoisoner(strength=6, freq=16, horizontal=False),
            ]
        )

    elif poisoner_flag == "1xl":
        x_poisoner = TurnerPoisoner()

    elif poisoner_flag == "4xl":
        x_poisoner = TurnerPoisoner(method="all-corners")

    else:
        raise NotImplementedError()

    return x_poisoner


def get_matching_datasets(
    dataset_flag,
    poisoner,
    label,
    seed=1,
    train_pct=1.0,
    big=False
):
    train_transform = TRANSFORM_TRAIN_XY[dataset_flag + ('_big' if big else '')]
    test_transform = TRANSFORM_TEST_XY[dataset_flag + ('_big' if big else '')]

    train_data = load_dataset(dataset_flag, train=True)
    test_data = load_dataset(dataset_flag, train=False)

    n_classes = len(train_data.classes)
    train_labels = np.array([y for _, y in train_data])

    train_labels = train_labels[:int(len(train_labels) * train_pct)]

    n_poisons_train = int((len(train_data) // n_classes) * train_pct)
    n_poisons_test = len(test_data) // n_classes

    if label == -1:
        poison_inds = np.where(train_labels != poisoner.target_label)[0][-n_poisons_train:]
    else:
        poison_inds = np.where(train_labels == label)[0][-n_poisons_train:]

    mtt_distill_dataset = distill_dataset = Subset(train_data, np.arange(len(train_data)))
    poison_dataset = MappedDataset(Subset(train_data, poison_inds),
                                   poisoner,
                                   seed=seed)

    train_dataset = Subset(train_data, np.arange(int(len(train_data) * train_pct)))
    dataset_list = [train_dataset, poison_dataset]
    if dataset_flag == 'tiny_imagenet':   # Oversample poisons for expert training
        dataset_list.extend([poison_dataset] * 9)
    train_dataset = ConcatDataset(dataset_list)

    if train_pct < 1.0:
        mtt_distill_dataset = Subset(distill_dataset, np.arange(int(len(distill_dataset) * train_pct)))

    mtt_dataset = MTTDataset(train_dataset, mtt_distill_dataset, poison_inds,
                             train_transform, n_classes)

    distill_dataset = MappedDataset(distill_dataset, train_transform)
    train_dataset = MappedDataset(train_dataset, train_transform)
    test_dataset = MappedDataset(test_data, test_transform)
    poison_test_dataset = PoisonedDataset(
        test_data,
        poisoner,
        eps=n_poisons_test,
        label=label if label != -1 else None,
        transform=test_transform,
    )

    return train_dataset, distill_dataset, test_dataset, poison_test_dataset, mtt_dataset


def construct_user_dataset(distill_dataset, labels, mask=None, target_label=None, include_labels=False):
    dataset = LabelWrappedDataset(distill_dataset, labels, include_labels)
    return dataset

def get_n_classes(dataset_flag):
    return N_CLASSES[dataset_flag]
