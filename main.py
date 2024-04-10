import numpy as np
import torch
import torch.nn as nn
import sys
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
from tqdm import tqdm, trange
from load import CustomDataset

np.random.seed(0)
torch.manual_seed(0)

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches**2, h * w * c // n_patches**2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class MyViT(nn.Module):
    def __init__(self, chw, n_patches=14, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert (
            chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
            chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches**2 + 1, hidden_d),
            persistent=False,
        )

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5) Classification MLPk
        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)  # Map to output dimension, output category distribution


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


def train(pixelFrom, pixelTo, batchSize, nPatches, nBlocks, hiddenD, nHeads, outC, learningRate, nEpoch):
    pixelF = pixelFrom
    pixelT = pixelTo
    batchSize = batchSize
    nPatches = nPatches
    nBlocks = nBlocks
    hiddenD = hiddenD
    nHeads = nHeads
    outC = outC
    learningRate = learningRate
    nEpoch = nEpoch

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    trainTransform = transforms.Compose([
        transforms.RandomCrop(int(pixelT)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    testTransform = transforms.Compose([
        transforms.Resize(int(pixelT)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    print("Loading data...")
    print()

    train_set = CustomDataset(csv_file='jute-pest-classification/train_'+pixelF+'_preprocessed_encoded.csv', image_dir='jute-pest-classification/train_images_'+pixelF+'_preprocessed/', transform=trainTransform)
    test_set = CustomDataset(csv_file='jute-pest-classification/test.csv', image_dir='jute-pest-classification/test_images_'+pixelF+'_preprocessed/', transform=testTransform)

    print("Data loaded")
    print("Train set size: ", len(train_set))
    print("Test set size: ", len(test_set))
    print()

    # Split training dataset into training and validation sets
    train_size = int(0.7 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batchSize)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batchSize)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batchSize)

    print("Data loaders created")
    print("Train loader size: ", len(train_loader))
    print("Validation loader size: ", len(val_loader))
    print("Test loader size: ", len(test_loader))
    print()

    # Defining model and training options
    print("Checking available GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "Not using CUDA",
    )
    print()

    model = MyViT(
        (3, pixelT, pixelT), n_patches=nPatches, n_blocks=nBlocks, hidden_d=hiddenD, n_heads=nHeads, out_d=outC
    ).to(device)

    N_EPOCHS = nEpoch
    LR = learningRate

    print("Model created")
    print("Number of epochs: ", N_EPOCHS)
    print("Learning rate: ", LR)
    print()

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):

            if isinstance(batch, tuple):
                x, y = batch
            elif isinstance(batch, dict):
                x, y = batch['image'], batch['label']
            else:
                raise TypeError("Unsupported batch format")

            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

        # Validation loop
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            for batch in tqdm(val_loader, desc="Validation"):
                if isinstance(batch, tuple):
                    x, y = batch
                elif isinstance(batch, dict):
                    x, y = batch['image'], batch['label']
                else:
                    raise TypeError("Unsupported batch format")

                x, y = x.to(device), y.to(device)

                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(val_loader)

                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                total += len(x)
            print(f"Val loss: {test_loss:.2f}")
            print(f"Val accuracy: {correct / total * 100:.2f}%")

    torch.save(model, 'ViT-'+pixelT+'-1st.pth')

def contTrain(pixelFrom, pixelTo, batchSize, learningRate, nEpoch, fileName):
    pixelF = pixelFrom
    pixelT = pixelTo
    batchSize = batchSize
    learningRate = learningRate
    nEpoch = nEpoch
    fileName = fileName

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    trainTransform = transforms.Compose([
        transforms.RandomCrop(int(pixelT)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    testTransform = transforms.Compose([
        transforms.Resize(int(pixelT)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    print("Loading data...")
    print()

    train_set = CustomDataset(csv_file='jute-pest-classification/train_'+pixelF+'_preprocessed_encoded.csv', image_dir='jute-pest-classification/train_images_'+pixelF+'_preprocessed/', transform=trainTransform)
    test_set = CustomDataset(csv_file='jute-pest-classification/test.csv', image_dir='jute-pest-classification/test_images_'+pixelF+'_preprocessed/', transform=testTransform)

    print("Data loaded")
    print("Train set size: ", len(train_set))
    print("Test set size: ", len(test_set))
    print()

    # Split training dataset into training and validation sets
    train_size = int(0.7 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batchSize)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batchSize)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batchSize)

    print("Data loaders created")
    print("Train loader size: ", len(train_loader))
    print("Validation loader size: ", len(val_loader))
    print("Test loader size: ", len(test_loader))
    print()

    # Defining model and training options
    print("Checking available GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "Not using CUDA",
    )
    print()

    model = torch.load(fileName)

    N_EPOCHS = nEpoch
    LR = learningRate

    print("Model loaded")
    print(f"Continuing training for {N_EPOCHS} epochs...")
    print("Learning rate: ", LR)
    print()

    count = 0

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        count+= 1
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):

            if isinstance(batch, tuple):
                x, y = batch
            elif isinstance(batch, dict):
                x, y = batch['image'], batch['label']
            else:
                raise TypeError("Unsupported batch format")

            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

        # Validation loop
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            for batch in tqdm(val_loader, desc="Validation"):
                if isinstance(batch, tuple):
                    x, y = batch
                elif isinstance(batch, dict):
                    x, y = batch['image'], batch['label']
                else:
                    raise TypeError("Unsupported batch format")

                x, y = x.to(device), y.to(device)

                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(val_loader)

                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                total += len(x)
            print(f"Val loss: {test_loss:.2f}")
            print(f"Val accuracy: {correct / total * 100:.2f}%")

        if count % 5 == 0:
            print("Saving model...")
            torch.save(model, 'ViT-'+pixelT+'-'+str(count)+'.pth')

    torch.save(model, 'ViT-'+pixelT+'-1st.pth')

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py")
        sys.exit(1)
    elif sys.argv[1] == "startTrain":
        if len(sys.argv) < 11:
            print("Usage: python main.py startTrain pixelFrom pixelTo batchSize nPatches nBlocks hiddenD nHeads outC learningRate nEpoch")
            sys.exit(1)
        train(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]), float(sys.argv[10]), int(sys.argv[11]))
        pass
    elif sys.argv[1] == "startTest":
        pass
    elif sys.argv[1] == "startPredict":
        pass
    elif sys.argv[1] == "startPreprocess":
        pass
    elif sys.argv[1] == "continueTraining":
        if len(sys.argv) < 8:
            print("Usage: python main.py continueTraining pixelFrom pixelTo batchSize learningRate nEpoch fileName")
            sys.exit(1)
        contTrain(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
        pass
    elif sys.argv[1] == "startEncodeCategory":
        pass
    elif sys.argv[1] == "startDecodeCategory":
        pass
    else:
        pass

if __name__ == "__main__":
    main()