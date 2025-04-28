import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler        # mixed-precision-training
from Generator import Generator
from Discriminator import Discriminator
import multiprocessing
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name)
          
        if self.transform:
            image = self.transform(image)

        return image


def save_models():
    
    model_saving_path_generator = "Your path here"
    model_saving_path_discriminator = "Your path here"
    
    torch.save({
    'model_state_dict': generator.state_dict(),
    }, model_saving_path_generator)
    print("Generator successfully saved.")
    
    torch.save({
        'model_state_dict': discriminator.state_dict(),   
    }, model_saving_path_discriminator)
    print("Discriminator successfully saved.")
    
    

def load_models():
    
    model_loading_path_generator = "Your path here"
    model_loading_path_discriminator = "Your path here"
    
    checkpoint_generator = torch.load(model_loading_path_generator, weights_only=True)
    generator.load_state_dict(checkpoint_generator['model_state_dict'])
    print("Generator successfully loaded.")
    
    checkpoint_discriminator = torch.load(model_loading_path_discriminator, weights_only=True)
    discriminator.load_state_dict(checkpoint_discriminator['model_state_dict'])
    print("Discriminator successfully loaded.")
    
    return generator, discriminator
    
    
    
# Hyperparameter 
img_channels = 3
features_d = 3
latent_dim = 256
epochs = 4000               # Exceptionally high, because GANs need a lot of training, so the training loop won't stop too early

batch_size = 45
lr_generator = 1e-4
lr_discriminator = 2e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

continue_training = True        # Continue Training (load weights) or new initialization? - (True == load existing model, False == start all over)


# Trainingscript
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # initializing (or loading) both models
    generator = Generator(latent_dim=latent_dim, style_dim=latent_dim, features_d=features_d)
    discriminator = Discriminator(img_channels = img_channels, features_d=features_d)
    
    if continue_training:
        print("Continuing Training.")
        generator, discriminator = load_models()
        
        

    p1 = sum(p1.numel() for p1 in discriminator.parameters())
    p2 = sum(p2.numel() for p2 in generator.parameters())
    
    print(f"Thats the parametercount: Discriminator: {p1} - Generator: {p2}.")

    criterion = nn.BCEWithLogitsLoss()    # Binary Cross-Entropy for Mixed-Precision. You  could change it back to BCE, but then you also need the sigmoid-activation in the Discriminator after the last Layer.
    scaler = GradScaler(device=device)    # FP16 Scaler 
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(0.8, 0.999))

    # Move both models to the GPU
    generator.to(device=device)
    discriminator.to(device=device)
    
    # Move both optimizers to the GPU
    for state in optimizer_G.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(generator)

    for state in optimizer_D.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(discriminator)
      
    
    # Preprocessing the images (resizing, create a tensor, Normalize (/255))                
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ])
                
     
    
    path = "Your dataset path here"          # You have to navigate to the folder, the dataset-class creates a dataset based on the images inside that folder.
                
    dataset = CustomImageDataset(img_dir = path, transform = transform)   
     
    # !Important! - You might need to readjust 'num_workers, prefetch_factor' depending on your hardware 
    # My setup: Ryzen 5 3500x, 32GB RAM, NVIDIA RTX 4060 8GB  
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=7)   # My GPU Usage: 100% (GPU is not waiting for Batches)
    
    num_batches = len(dataloader)
    print(f"The DataLoader contains {num_batches} batches.")
    
    # set both models to trainingmode
    generator.train()
    discriminator.train()
    
    # The discriminator is not always updated.
    train_discriminator = False               
    
    for epoch in range(epochs):
        
        counter = 0
        
        for batch in dataloader:
            batch_len, _, _, _ = batch.shape
            
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            
            
            real_batch = batch.to(device)
            real_labels = torch.ones(batch_len, 1).to(device)  # Label 1 for real data
            
            # ==========================================================================================================================================
            # controlls the discriminator-training        || Currently it has to be manually set. - In the future, there will be a heuristic algorithm!
            # ==========================================================================================================================================
            
            if (epoch+1) %2 == 0 and (counter+1) %5 == 0:
                train_discriminator=True    # set Trainingvariable for Discriminator on True
            
            else:
                train_discriminator = False
            
            
            # 1. Discriminator-Update
            if train_discriminator == True:
                with autocast(device_type=device, dtype=torch.float16):
                    latent_matrix = torch.randn(batch_len, latent_dim).to(device)
                    
                    with torch.no_grad():
                        fake_data = generator(latent_matrix, batch_len)  # Generated images
                        
                    fake_labels = torch.zeros(batch_len, 1).to(device)  # Label 0 for generated images
                    real_loss = criterion(discriminator(real_batch), real_labels)
                    fake_loss = criterion(discriminator(fake_data), fake_labels)
                    
                    loss_D = (real_loss + fake_loss) / 2
                    
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()
                
                
                
                        
            # 2. Generator-Update
            with autocast(device_type=device, dtype=torch.float16):
                latent_matrix = torch.randn(batch_len, latent_dim).to(device)
                fake_data = generator(latent_matrix, batch_len)
            
                loss_G = criterion(discriminator(fake_data), real_labels)  
            
            
            
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            
            # You might not need these LOGs, but they can be pretty useful to track actual losses during an epoch
            if(counter+1) %5 == 0 and train_discriminator == True:
                print(f"Loss D: {loss_D.item():.4f} - Loss G: {loss_G.item():.4f}")
                       
            
            counter += 1
            
        # Logging final epoch-progress and saving one image from the last batch
        if (epoch+1) %1 == 0 and train_discriminator == True:
            print(f"Epoch [{epoch+1}] - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
            
            matrix = fake_data.detach().cpu().numpy()
            image = np.transpose(matrix[0], (1, 2, 0))
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(f"Your path here/{epoch+1}.png")    # Thats the final path to save an image-example.
        
        else: 
            print(f"Epoch [{epoch+1}] - Loss G: {loss_G.item():.4f}")
            
            matrix = fake_data.detach().cpu().numpy()
            image = np.transpose(matrix[0], (1, 2, 0))
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(f"Your path here/{epoch+1}.png")    # Thats the final path to save an image-example.
            
            
            
        # save the models    
        if (epoch+1) %10 == 0:
            save_models()
            torch.cuda.empty_cache()
            
            
        train_discriminator = False                # reset discriminator-training-flag
