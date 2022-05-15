import numpy as np
import matplotlib.pyplot as plt

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def plot_images(images, labels, classes, normalize=True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 20))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title("Class: " + str(classes[labels[i]]))
        ax.axis('off')
        
        
# iterator = iter(test_loader)
# for i in range(0, 101): #(len(test_loader)):
#     data = next(iterator)
#     if i == 100:
#       plot_images(data, classes, classes)
#     # while i < 100: # k would be the batch index where to resume training
#     #     continue
#     # plot_images(data, classes, classes)

# def img_convert(tensor):
#     image = tensor.clone().detach().numpy()
#     image = image.transpose(1, 2, 0) # we reverse the pixels
#     # image = image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5]) # denormalization
#     image = image.clip(0, 1)
#     return image

# data_iter = iter(train_raw_loader) # it creates an object that allows us to throw the ittrable training loader one element at a time
# images, labels = data_iter.next() # it will grab the first batch of out training data
# fig = plt.figure(figsize=(25, 20)) # width & height of the figure ...

# for idx in np.arange(32):
#     ax = fig.add_subplot(8, 4, idx + 1, xticks=[], yticks=[]) # 2 rows & 10 columns!
#     plt.imshow(img_convert(images[idx]).squeeze(), cmap=plt.get_cmap('gray'))
#     ax.set_title("Class : " + str(labels[idx].item()), size=15)




import scipy.ndimage as ndimage


def plot_ae_outputs_den(my_model, my_loader, n=1):
    plt.figure(figsize=(10, 20))
    my_model.eval()
    for i in range(n):
        ax = plt.subplot(5, n, i+1)
        image_noisy, org_img = next(iter(my_loader)) #test_raw_dataset[i][0].unsqueeze(0)
        org_img = org_img[i].to(device)
        image_noisy = image_noisy[i].to(device)
        with torch.no_grad():
            image_noisy = image_noisy.to(device)
            residual_img = my_model(image_noisy.view(1, 3, 360, 640))

        # new_data = ndimage.rotate(img.cpu().squeeze().numpy().T, angle, reshape=True)
        # ax.imshow((org_img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8));
        ax.imshow(normalize_image(org_img).detach().permute(1, 2, 0).cpu().numpy());
      
        # tr = transforms.Affine2D().rotate_deg(180)
        # ax.imshow(img.cpu().squeeze().numpy().T, transform=tr + ax.transData)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Original images')
        ax = plt.subplot(5, n, i + 1 + n);
        # new_data = ndimage.rotate(image_noisy.cpu().squeeze().numpy().T, angle, reshape=True)

        # ax.imshow((image_noisy.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        ax.imshow(normalize_image(image_noisy).detach().permute(1, 2, 0).cpu().numpy());
        # plt.imshow(image_noisy.cpu().reshape(256, 512, 3))
        # tr = transforms.Affine2D().rotate_deg(180)
        # ax.imshow(img.cpu().squeeze().numpy().T, transform=tr + ax.transData)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Corrupted images')
        # print(torch.max((org_img - image_noisy)))
        # print("Calc Res:=> ", torch.max(residual_img))
        ax = plt.subplot(5, n, i + 1 + n + n);
        # new_data = ndimage.rotate(rec_img.cpu().squeeze().numpy().T, angle, reshape=True)
        # plt.imshow(rec_img.cpu().reshape(360, 640, 3))
        # print(residual_img.size())
        reconstructed_img = residual_img.detach().cpu() + image_noisy.detach().cpu()

        # ax.imshow((reconstructed_img.view(3, 360, 640).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8));
        ax.imshow(normalize_image(reconstructed_img.view(3, 360, 640)).detach().permute(1, 2, 0).cpu().numpy());
        # tr = transforms.Affine2D().rotate_deg(180)
        # ax.imshow(img.cpu().squeeze().numpy().T, transform=tr + ax.transData)
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False); 
        if i == n//2:
            ax.set_title('Reconstructed images')
        
        ax = plt.subplot(5, n, i + 1 + n + n + n);

        # ax.imshow((residual_img.detach().view(3, 360, 640).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8));
        ax.imshow(normalize_image(residual_img.view(3, 360, 640)).detach().permute(1, 2, 0).cpu().numpy());
        # tr = transforms.Affine2D().rotate_deg(180)
        # ax.imshow(img.cpu().squeeze().numpy().T, transform=tr + ax.transData)
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False); 
        if i == n//2:
            ax.set_title('Model Residual images')
        
        ax = plt.subplot(5, n, i + 1 + n + n + n + n);

        # ax.imshow((torch.abs(org_img - image_noisy).detach().view(3, 360, 640).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8));
        ax.imshow(normalize_image((image_noisy - org_img).view(3, 360, 640)).detach().permute(1, 2, 0).cpu().numpy());
        # tr = transforms.Affine2D().rotate_deg(180)
        # ax.imshow(img.cpu().squeeze().numpy().T, transform=tr + ax.transData)
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False); 
        if i == n//2:
            ax.set_title('Real Residual images')
    
    plt.subplots_adjust(left=0.0,
                        bottom=0.001, 
                        right=0.9, 
                        top=0.999, 
                        wspace=0., 
                        hspace=0.1,
                        );
    plt.show();
