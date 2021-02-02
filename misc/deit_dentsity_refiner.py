import torch


def get_patches(img):
    _, im_h, im_w = img.shape

    patches1 = []
    patches1_coords = []
    patches2 = []
    patches2_coords = []
    patches3 = []
    patches3_coords = []
    patches4 = []
    patches4_coords = []

    h_pivot = 4
    while h_pivot < im_h - 4:

        w_pivot = 4
        while w_pivot < im_w - 4:
            y1 = h_pivot if h_pivot + 224 < im_h - 4 else im_h - 224 - 4
            y2 = y1 + 224
            x1 = w_pivot if w_pivot + 224 < im_w - 4 else im_w - 224 - 4
            x2 = x1 + 224
            patches1.append(img[:, y1 - 4:y2 - 4, x1 - 4:x2 - 4])
            patches1_coords.append((y1 - 4, y2 - 4, x1 - 4, x2 - 4))
            patches2.append(img[:, y1 - 4:y2 - 4, x1 + 4:x2 + 4])
            patches2_coords.append((y1 - 4, y2 - 4, x1 + 4, x2 + 4))
            patches3.append(img[:, y1 + 4:y2 + 4, x1 - 4:x2 - 4])
            patches3_coords.append((y1 + 4, y2 + 4, x1 - 4, x2 - 4))
            patches4.append(img[:, y1 + 4:y2 + 4, x1 + 4:x2 + 4])
            patches4_coords.append((y1 + 4, y2 + 4, x1 + 4, x2 + 4))
            w_pivot += 224
        h_pivot += 224

    return (patches1, patches2, patches3, patches4), \
           (patches1_coords, patches2_coords, patches3_coords, patches4_coords)


def refine_density(model, den, img):
    mask = torch.zeros((224, 224))  # Make a mask, we only want the centre 8 x 8 pixels per 16 x 16 block
    for h in range(224):
        for w in range(224):
            if 4 <= h % 16 < 12 and 4 <= w % 16 < 12:
                mask[h, w] = 1

    all_patches, all_patches_coords = get_patches(img)
    for patches, patches_coords in zip(all_patches, all_patches_coords):
        patch_stack = torch.stack(patches).cuda()

        out = model(patch_stack)
        out = out.squeeze().cpu()

        for i in range(len(patches_coords)):
            y1, y2, x1, x2 = patches_coords[i]
            pred = out[i]
            den[y1:y2, x1:x2] = den[y1:y2, x1:x2] - mask * den[y1:y2, x1:x2]  # Remove old value
            den[y1:y2, x1:x2] = den[y1:y2, x1:x2] + mask * pred  # Insert new value

    return den