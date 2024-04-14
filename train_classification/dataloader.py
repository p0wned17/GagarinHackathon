from torch.utils.data import Dataset, DataLoader
import pandas as pd
from turbojpeg import TurboJPEG
from augmentations import train_transform, val_transform
import cv2

turbo_jpeg = TurboJPEG()


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # self.root_dir = root_dir
        self.dataframe = pd.read_csv(csv_file, delimiter=",")
        # self.dataframe.reset_index(inplace=True)
        # print(self.dataframe)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        
        class_id = int(self.dataframe.iloc[idx, 1])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        return image, class_id


# Создание экземпляра Dataset
train_dataset = CustomDataset(csv_file="train_angle.csv", transform=train_transform)

val_dataset = CustomDataset(csv_file="val_angle.csv", transform=val_transform)
print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}")
# exit(1)


def get_dataloaders():

    
    # Создание DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"train_dataloader: {len(train_dataloader)}, val_dataloader: {len(val_dataloader)}"
    )

    return train_dataloader, val_dataloader
