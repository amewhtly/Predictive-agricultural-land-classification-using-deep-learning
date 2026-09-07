
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio as rio
#import imageio
from PIL import Image
import cv2
import imageio.v2 as imageio

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.ndimage import binary_erosion
from sklearn.metrics import confusion_matrix
# from tensorflow_addons.layers import GroupNormalization
import matplotlib.colors as mcolors
import matplotlib.patches as patches


path = 'C:/Users/milli/OneDrive/Documents/Cranfield University/Thesis/Data/'
path1 = "C:/Users/milli/OneDrive/Documents/Cranfield University/Thesis/Data/new/chip64/"
path2 = "C:/Users/milli/OneDrive/Documents/Cranfield University/Thesis/Data/new/lbl64/"
pathC = path + 'target/'
pathP = path + 'predictors/'
outpath = path + 'new/'
soil_path = outpath + "soil_hybrid.tif"

strata_path = pathC + "ALC_predict_hybrid.tif"
raster_paths = [
    #outpath + "soil_hybrid_tag.tif",
    pathP + "AAR.tif",
    pathP + "Aspect.tif",
    pathP + "ASR.tif",
    pathP + "AT0.tif",
    pathP + "ATS.tif",
    pathP + "Channel_Network_Base_Level.tif",
    pathP + "Channel_Network_Distance.tif",
    pathP + "Convergence_Index.tif",
    pathP + "Flow_Path_Length.tif",
    pathP + "LS_Factor.tif",
    pathP + "MRRTF.tif",
    pathP + "MRVBF.tif",
    pathP + "Plan_Curvature.tif",
    pathP + "Profile_Curvature.tif",
    pathP + "Relative_Slope_Position.tif",
    pathP + "Slope.tif",
    pathP + "TCI_Low.tif",
    pathP + "Topographic_Position_Index.tif",
    pathP + "Topographic_Wetness_Index.tif",
    pathP + "Total_Catchment_Area.tif",
    pathP + "Valley_Depth.tif",
    pathP + "DROCK50.tif",
    pathP + "TEXTURE50.tif",
    pathP + "SOILGROUP50.tif",
    pathP + "WETNESS50.tif"
]



Nbands = 1 + len(raster_paths)

# Custom colour map
grade_colours = {
    0: (0, 0, 0, 0),          # transparent
    1: '#0081fe',
    2: '#c2fbfe',
    3: '#018100',
    4: '#a5fe81',
    5: '#fefb69',
    6: '#b28864'
}
# colours = ['#0081fe', '#c2fbfe', '#018100', '#a5fe81', '#fefb69', '#b28864']
# keys = ['Grade 1', 'Grade 2', 'Grade 3a', 'Grade 3b', 'Grade 4', 'Grade 5']

# Dataset colours
dataset_colours = {
    "train": "#d873f6",
    "valid": "#ee5a9f",
    "test":  "#eb1111"
}


def LoadData(img_dir, mask_dir):
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    X = []
    y = []


    for img_f, mask_f in zip(img_files, mask_files):
        img = imageio.imread(os.path.join(img_dir, img_f))   # shape (64,64,27)
        mask = imageio.imread(os.path.join(mask_dir, mask_f)) # shape (64,64)
        print(img.shape)
        raw_mask = imageio.imread(os.path.join(mask_dir, mask_f))
        print("Raw mask unique values:", np.unique(raw_mask))
        
        # Ensure correct shapes
        img = img.astype(np.float32)
        mask = mask.astype(np.int32)

        # Replace nodata pixels in the image
        # img[img < -1e30] = 0
        # img = np.nan_to_num(img, nan=0)

        # Replace nodata pixels in the mask
        print("Raw mask min/max:", mask.min(), mask.max())
        print("Raw mask unique:", np.unique(mask))

        mask[mask < 0] = 0

        # Add channel dimension to mask
        mask = mask[..., np.newaxis]

        X.append(img)
        y.append(mask)

        # detect if it has class 1
        has_class1 = np.any(mask == 1)
        if has_class1:
            rot_180 = np.rot90(img, k=2)
            mask_rot_180 = np.rot90(mask, k=2)

            rot_90 = np.rot90(img, k=1)
            mask_rot_90 = np.rot90(mask, k=1)


            X.append(rot_180)
            y.append(mask_rot_180)

            X.append(rot_90)
            y.append(mask_rot_90)



    img_array = np.array(X)
    mask_array = np.array(y)

    # Train / validation / test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        img_array, mask_array,
        test_size=0.3,
        random_state=123
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=123
    )

    print(np.sum(img_array < -1e30))

    # Normalize images
    #X_train = X_train / np.max(X_train)
    mean = np.mean(X_train, axis=(0,1,2))   # shape (27,)
    std  = np.std(X_train, axis=(0,1,2))    # shape (27,)
    X_train = (X_train - mean) / (std + 1e-6)
    X_valid = (X_valid - mean) / (std + 1e-6)
    X_test = (X_test - mean) / (std + 1e-6)

    # Debug info
    print("Mask classes:", np.unique(mask_array))
    print("Image shape:", X_train.shape[1:])

    print("Min pixel value:", np.min(img_array))
    print("Count of nodata-like pixels:", np.sum(img_array < -1e30))
    print("Mask min:", mask_array.min())
    print("Mask max:", mask_array.max())
    print("unique (y_train):", np.unique(y_train))
    print("unique (y_valid):", np.unique(y_valid))
    print("unique (y_test):", np.unique(y_test))


    # num_classes = mask_array.max() + 1
    # print("Detected classes:", num_classes) 
    assert mask_array.min() >= 0
    assert mask_array.max() <= 6

    # Optional: quick visualization of first sample
    i = 0
    img_view = X_train[i]          # (64,64,27)
    mask_view = mask_array[i, :, :, 0]  # (64,64)
    shape = X_train.shape[1:]   # (H, W, channels)
    print(f"Shape:", shape)

    print(f"Image {i} shape:", img_view.shape)
    print(f"Mask {i} shape:", mask_view.shape)
    print("Mask classes:", np.unique(mask_view))

    shape = img_view.shape

    fig, arr = plt.subplots(1, 2, figsize=(10, 5))
    arr[0].imshow(np.mean(img_view, axis=-1), cmap='gray')
    arr[0].set_title(f'Image {i}')
    arr[1].imshow(mask_view, cmap=custom_cmap)
    arr[1].set_title(f'Masked Image {i}')
    plt.tight_layout()
    plt.show()


    return X_train, X_valid, X_test, y_train, y_valid, y_test, shape



def encoder_block(inputs, num_filters, dropout_prob=0.0):

    x = tf.keras.layers.Conv2D(
        num_filters, 3, padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)
        # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    #x = tf.keras.layers.BatchNormalization()(x) #, training=False

    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(
        num_filters, 3, padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
        # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    #x = tf.keras.layers.BatchNormalization()(x) #, training=False

    x = tf.keras.layers.Activation('relu')(x)


    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:
        x = tf.keras.layers.Dropout(dropout_prob)(x) # x = tf.keras.layers.SpatialDropout2D(dropout_prob)(x)

    # skip connection
    skip_connection = x

    #Pooling reduces the size of the image while keeping the number of channels same
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    return x, skip_connection

def decoder_block(inputs, skip_features, num_filters):

    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same',
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)

    skip_features = tf.keras.layers.Resizing(x.shape[1], x.shape[2])(skip_features)

    x = tf.keras.layers.Concatenate()([x, skip_features])

    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Activation('relu')(x)
    

    return x

def unet_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Contracting Path (Encoder)
    c1, s1 = encoder_block(inputs, 64)
    c2, s2 = encoder_block(c1, 128)
    c3, s3 = encoder_block(c2, 256)
    c4, s4 = encoder_block(c3, 512)
    
    # Bottleneck
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(c4)
    b1 = tf.keras.layers.Activation('relu')(b1)
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))(b1)
    b1 = tf.keras.layers.Activation('relu')(b1)
    
    # Expansive Path (Decoder)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # conv = tf.keras.layers.Conv2D(64, 3, padding='same')(d4)
    # conv = tf.keras.layers.Activation('relu')(conv)
    print("end of unet model")
    
    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(d4)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='U-Net')
    return model


def predict_large_image(path, model, patch_size=(64, 64), stride=2, debug_one_patch=False):
    """
    Predicts every pixel of a large image using overlapping patches.
    Uses batch inference for speed.
    Handles borders and nodata (NaN).
    """

    ph, pw = patch_size
    num_classes = model.output_shape[-1]

    mean = np.mean(X_train, axis=(0,1,2))   # shape (27,)
    std  = np.std(X_train, axis=(0,1,2))    # shape (27,)

    debug_results = []

    with rio.open(path) as src:
        H, W = src.height, src.width
        C = src.count

        out_mask = np.zeros((H, W, num_classes), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        for y in range(0, H, stride):
            for x in range(0, W, stride):

                window = rio.windows.Window(x, y, pw, ph)
                sub_raw = src.read(window=window).transpose(1,2,0)
                h, w = sub_raw.shape[:2]

                # 1. nodata mask
                nodata_mask = np.isnan(sub_raw).any(axis=-1)

                # 2. replace NaN for model input
                for c in range(C):
                    sub_raw[..., c] = np.nan_to_num(sub_raw[..., c], nan=mean[c])

                # 3. pad FIRST (so padding gets normalized too)
                patch = np.zeros((ph, pw, C), dtype=np.float32)
                patch[:h, :w] = sub_raw

                # 4. normalize entire patch
                patch = (patch - mean) / (std + 1e-6)

                # 5. predict
                pred = model.predict(patch[None])[0]

                # 6. apply nodata mask only to real region
                pred[:h, :w][nodata_mask] = 0

                # return first valid patch for debugging
                # debug mode: collect first N valid patches
                if debug_one_patch:
                    N = 30

                    if not nodata_mask.all():   # skip fully-nodata patches
                        eps = 1e-9
                        entropy = -np.sum(pred * np.log(pred + eps), axis=-1)

                        sorted_probs = np.sort(pred, axis=-1)
                        margin = sorted_probs[..., -1] - sorted_probs[..., -2]
                        margin_uncertainty = 1 - margin

                        class_map = np.argmax(pred, axis=-1)

                        debug_results.append((class_map, entropy, margin_uncertainty))

                        # once we have 20 patches, return them
                        if len(debug_results) >= N:
                            return debug_results
                    continue

                # 6. stitch back
                yy2 = min(y + ph, H)
                xx2 = min(x + pw, W)
                out_mask[y:yy2, x:xx2] += pred[:yy2-y, :xx2-x]
                count_map[y:yy2, x:xx2] += 1

    # If debug mode was on but fewer than N valid patches were found
    if debug_one_patch:
        return debug_results

    # Average overlapping predictions
    out_mask /= np.maximum(count_map[..., None], 1e-9)

    # uncertainty map
    eps = 1e-9
    entropy_map = -np.sum(out_mask * np.log(out_mask + eps), axis=-1)

    sorted_probs = np.sort(out_mask, axis=-1)
    margin_map = sorted_probs[..., -1] - sorted_probs[..., -2]
    margin_uncertainty = 1 - margin_map # so high uncertainty is high value?

    class_map = np.argmax(out_mask, axis=-1)

    return class_map, entropy_map, margin_uncertainty



def sparse_focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    # y_true: (B,H,W,1)
    y_true = tf.squeeze(y_true, axis=-1)  # (B,H,W)

    # mask out nodata (class 0)
    valid = tf.cast(tf.not_equal(y_true, 0), tf.float32)

    # convert to int
    y_true_int = tf.cast(y_true, tf.int32)

    # get per-pixel probabilities
    # y_pred_soft shape: (B, H, W, C)
    y_pred_soft = tf.nn.softmax(y_pred, axis=-1)

    # Gather the probability of the true class for each pixel.
    # pt = p(y_true_class)
    # batch_dims=3 means gather along the last dimension (class axis).
    pt = tf.gather(y_pred_soft, y_true_int, batch_dims=3)

    # Focal scaling term: (1 - pt)^gamma
    # Down-weights easy examples, focuses on hard ones.
    focal_term = tf.pow(1.0 - pt, gamma)

    # standard sparse categorical crossentropy.
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Optional alpha weighting (per-class weighting).
    # alpha is a vector of length C.
    if alpha is not None:
        # Gather alpha for each pixel's true class.
        alpha_t = tf.gather(alpha, y_true_int)
        # Apply class weights.
        ce = ce * alpha_t

    # apply focal scaling + mask
    loss = focal_term * ce * valid

    # Normalize by number of valid pixels to avoid bias from nodata.
    return tf.reduce_sum(loss) / (tf.reduce_sum(valid) + 1e-6)

def focal_loss_fixed(y_true, y_pred):
    return sparse_focal_loss(y_true, y_pred, gamma=2.0, alpha=alpha)


def compute_iou_per_class(y_true, y_pred, num_classes):
    ious = []
    for cls in range(num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)

        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()

        if union == 0:
            iou = np.nan  # class not present
        else:
            iou = intersection / union

        ious.append(iou)

    return ious, np.nanmean(ious)


def compute_boundary(mask):
    eroded = binary_erosion(mask)
    boundary = mask ^ eroded
    return boundary

def compute_boundary_iou(y_true, y_pred, num_classes):
    bious = []
    for cls in range(num_classes):
        true_mask = (y_true == cls)
        pred_mask = (y_pred == cls)

        true_boundary = compute_boundary(true_mask)
        pred_boundary = compute_boundary(pred_mask)

        intersection = np.logical_and(true_boundary, pred_boundary).sum()
        union = np.logical_or(true_boundary, pred_boundary).sum()

        if union == 0:
            biou = np.nan
        else:
            biou = intersection / union

        bious.append(biou)

    return bious, np.nanmean(bious)

def create_colour_map():
    """Create a ListedColormap from the grade_colors dictionary."""
    rgba_colours = []
    for k in sorted(grade_colours.keys()):
        col = grade_colours[k]
        if isinstance(col, str):
            rgba_colours.append(mcolors.to_rgba(col))
        else:
            rgba_colours.append(col)
    return mcolors.ListedColormap(rgba_colours)

def normalize_to_rgba(col):
    """Convert hex, RGB, or RGBA to a 4‑channel RGBA tuple."""
    if isinstance(col, str):
        return mcolors.to_rgba(col)  # already RGBA
    col = np.array(col, dtype=float)
    if col.shape[0] == 3:
        return (*col, 1.0)          # add alpha = 1
    return tuple(col)               # already RGBA

def apply_colours_rgba(class_map, colours):
    H, W = class_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)

    for cls, col in colours.items():
        rgba[class_map == cls] = normalize_to_rgba(col)

    return rgba

def create_rgba_cmap(colour_dict):
    rgba_list = [normalize_to_rgba(colour_dict[k]) for k in sorted(colour_dict.keys())]
    return mcolors.ListedColormap(rgba_list)

custom_cmap = create_rgba_cmap(grade_colours)

def stitch_debug_patches(debug_results, cols=5):
    """
    Stitches the first N debug patches into mosaics:
    - class mosaic
    - entropy mosaic
    - margin uncertainty mosaic
    """

    N = len(debug_results)
    rows = int(np.ceil(N / cols))

    class_maps = [r[0] for r in debug_results]
    entropies = [r[1] for r in debug_results]
    margins = [r[2] for r in debug_results]

    ph, pw = class_maps[0].shape

    class_mosaic = np.zeros((rows * ph, cols * pw))
    entropy_mosaic = np.zeros((rows * ph, cols * pw))
    margin_mosaic = np.zeros((rows * ph, cols * pw))

    for idx in range(N):
        r = idx // cols
        c = idx % cols

        y0, y1 = r * ph, (r + 1) * ph
        x0, x1 = c * pw, (c + 1) * pw

        class_mosaic[y0:y1, x0:x1] = class_maps[idx]
        entropy_mosaic[y0:y1, x0:x1] = entropies[idx]
        margin_mosaic[y0:y1, x0:x1] = margins[idx]

    return class_mosaic, entropy_mosaic, margin_mosaic


def visualize_mosaic(class_mosaic, entropy_mosaic, margin_mosaic):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(class_mosaic, cmap=custom_cmap)
    axes[0].set_title("Class Mosaic")
    axes[0].axis("off")

    axes[1].imshow(entropy_mosaic, cmap="inferno")
    axes[1].set_title("Entropy Mosaic")
    axes[1].axis("off")

    axes[2].imshow(margin_mosaic, cmap="viridis")
    axes[2].set_title("Margin Uncertainty Mosaic")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def VisualizeResults(index, X, y, model, dataset_name="valid", save_path=None):
    img = X[index]
    img_batch = img[np.newaxis, ...]

    # Predict mask probabilities
    pred_y = model.predict(img_batch)[0]   # shape: (H, W, C)

    # Predicted class map
    pred_mask = tf.argmax(pred_y, axis=-1).numpy()

    # Ground truth mask
    true_mask = y[index, :, :, 0]

    # ---- UNCERTAINTY MAPS ----
    eps = 1e-9

    # Entropy
    entropy_map = -np.sum(pred_y * np.log(pred_y + eps), axis=-1)

    # Margin uncertainty
    sorted_probs = np.sort(pred_y, axis=-1)
    margin_map = sorted_probs[..., -1] - sorted_probs[..., -2]
    margin_uncertainty = 1 - margin_map

    # ---- VISUALIZATION ----
    border_colour = dataset_colours[dataset_name]

    fig, arr = plt.subplots(1, 4, figsize=(32, 6))

    panels = [
        ("True Mask (Coloured)", true_mask, custom_cmap),
        ("Predicted Mask (Coloured)", pred_mask, custom_cmap),
        ("Entropy Uncertainty", entropy_map, "inferno"),
        ("Margin Uncertainty", margin_uncertainty, "viridis")
    ]

    for ax, (title, data, cmap) in zip(arr, panels):
        if cmap == custom_cmap:
            # categorical mask → use fixed class range
            ax.imshow(data, cmap=cmap, vmin=0, vmax=6)
        else:
            # continuous uncertainty maps
            ax.imshow(data, cmap=cmap)

        ax.set_title(
            f"{title}\n[{dataset_name.upper()} SET]",
            fontsize=12,
            backgroundcolor=border_colour,
            color="white"
        )
        ax.axis("off")

        rect = patches.Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            linewidth=6,
            edgecolor=border_colour,
            facecolor='none'
        )
        ax.add_patch(rect)


    plt.tight_layout()
    if save_path:
        save_fig(save_path)
    plt.close(fig)


def masked_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)

    # Ensure consistent dtype
    y_true = tf.cast(y_true, tf.int32)

    # Mask out nodata (class 0)
    valid = tf.not_equal(y_true, 0)

    # Predicted labels
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_pred_labels = tf.cast(y_pred_labels, tf.int32)

    # Compare only valid pixels
    matches = tf.equal(y_true, y_pred_labels)
    matches = tf.logical_and(matches, valid)

    return tf.reduce_sum(tf.cast(matches, tf.float32)) / (tf.reduce_sum(tf.cast(valid, tf.float32)) + 1e-6)


def integrated_gradients(model, input_image, baseline, target_class, steps=50):
    """
    Compute Integrated Gradients for a single image.
    input_image: (H, W, C)
    baseline: same shape, usually zeros
    target_class: integer class ID
    """
    input_image = tf.cast(input_image, tf.float32)
    baseline = tf.cast(baseline, tf.float32)

    interpolated = [
        baseline + (float(i) / steps) * (input_image - baseline)
        for i in range(steps + 1)
    ]

    grads = []
    for img in interpolated:
        img = tf.expand_dims(img, axis=0)  # add batch dimension
        with tf.GradientTape() as tape:
            tape.watch(img)
            preds = model(img)
            # preds shape: (1, H, W, num_classes)
            class_score = preds[..., target_class]
            class_score = tf.reduce_mean(class_score)  # scalar
        grad = tape.gradient(class_score, img)[0]  # remove batch dim
        grads.append(grad)

    grads = tf.stack(grads, axis=0)
    avg_grads = tf.reduce_mean(grads, axis=0)

    ig = (input_image - baseline) * avg_grads
    return ig.numpy()

def compute_channel_importance(model, X, target_class, steps=50):
    """
    Computes global channel importance over dataset X.
    X: (N, H, W, C)
    """
    N, H, W, C = X.shape
    baseline = np.zeros((H, W, C), dtype=np.float32)

    channel_scores = np.zeros(C)

    for i in range(N):
        ig = integrated_gradients(model, X[i], baseline, target_class, steps)
        print("IG shape:", ig.shape)
        # Sum over spatial dimensions → importance per channel
        channel_scores += np.sum(np.abs(ig), axis=(0, 1))

    # Normalize
    channel_scores /= np.sum(channel_scores)
    return channel_scores


def save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':

    # Preprocess + split
    X_train, X_valid, X_test, y_train, y_valid, y_test, shape = LoadData(path1, path2)

    
    # Build model
    unet = unet_model(input_shape=shape, num_classes=7)

    alpha = tf.constant([0.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # change class 1 to have weight like. 5? idk

    unet.summary()

    unet.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=focal_loss_fixed,
        metrics=[masked_accuracy]
    )


    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_masked_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    results = unet.fit(
        X_train, y_train,
        batch_size = 8,
        epochs=100,
        callbacks=[checkpoint],
        validation_data=(X_valid, y_valid),
    )

    # Load the best model for testing
    best_model = tf.keras.models.load_model(
        'best_model.keras',
        custom_objects={'masked_accuracy': masked_accuracy,
                        'focal_loss_fixed': focal_loss_fixed,
                        'sparse_focal_loss': sparse_focal_loss}
    )


    ############################################################
    # statistics (train)

    # predictions
    y_true_train = y_train.flatten()
    y_pred_train = np.argmax(best_model.predict(X_train), axis=-1).flatten()

    print(classification_report(y_true_train, y_pred_train))
    report = classification_report(y_true_train, y_pred_train)
    with open(outpath + "plots_2/classification_report_train.txt", "w") as f:
        f.write(report)

    # remove background (class 0)
    mask_train = y_true_train != 0
    y_true_cm = y_true_train[mask_train]
    y_pred_cm = y_pred_train[mask_train]

    # confusion matrix for classes 1–6
    cm = confusion_matrix(y_true_cm, y_pred_cm, labels=[1,2,3,4,5,6], normalize='true')
    print("Unique true classes:", np.unique(y_true_cm))

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix (Training)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_fig(outpath + "plots_2/confusion_matrix_train.png")
    plt.close()



    ## integrated gradients
    print(X_train.shape)
    C = X_train.shape[-1]
    # Number of raster predictors to keep (drop last 3 categorical rasters)
    num_real_rasters = len(raster_paths) - 3
    start = num_real_rasters

    # Clean raster names
    raster_channel_names = [
        os.path.basename(p).replace(".tif", "").replace("_", " ")
        for p in raster_paths[:num_real_rasters]
    ]
    # One-hot group sizes
    texture_n = 22
    wetness_n = 7
    soil_group_n = 80

    num_classes = 7
    classes_to_compute = [1,2,3,4,5,6]
    all_grouped_scores = []

    for cls in classes_to_compute:
        print(f"Computing IG for class {cls}... (TRAIN)")
        scores = compute_channel_importance(best_model, X_train[:50], cls, steps=25)

        # Group one-hot channels
        texture_scores = scores[start : start + texture_n].sum()
        wetness_scores = scores[start + texture_n : start + texture_n + wetness_n].sum()
        soil_group_scores = scores[start + texture_n + wetness_n :
                                start + texture_n + wetness_n + soil_group_n].sum()

        grouped_scores = list(scores[:start]) + [texture_scores, wetness_scores, soil_group_scores]
        all_grouped_scores.append(grouped_scores)

        grouped_names = raster_channel_names + ["Texture", "Wetness", "Soil Group"]

        plt.figure(figsize=(10, 12))
        plt.barh(grouped_names, grouped_scores)
        plt.xlabel("Importance")
        plt.title(f"Channel Importance for Class {cls} (Training)")
        plt.tight_layout()
        plt.savefig(outpath + f"plots_2/train_channel_importance_class_{cls}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    # Overall average across classes
    overall_scores = np.mean(all_grouped_scores, axis=0)

    plt.figure(figsize=(10, 12))
    plt.barh(grouped_names, overall_scores)
    plt.xlabel("Importance")
    plt.title("Overall Channel Importance (Training)")
    plt.tight_layout()
    plt.savefig(outpath + "plots_2/train_channel_importance_overall.png", dpi=300)
    plt.close()

    ###################################################

    print("Unique classes in validation:", np.unique(y_valid))
    print("Count of class 0 in validation:", np.sum(y_valid == 0))

    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    axis[0].plot(results.history["loss"], color='r', label = 'train loss')
    axis[0].plot(results.history["val_loss"], color='b', label = 'val loss')
    axis[0].set_title('Loss Comparison')
    axis[0].legend()
    axis[1].plot(results.history["masked_accuracy"], color='r', label = 'train accuracy')
    axis[1].plot(results.history["val_masked_accuracy"], color='b', label = 'val accuracy')
    axis[1].set_title('Accuracy Comparison')
    axis[1].legend()
    save_fig(outpath + "plots_2/loss_accuracy.png")


    # loss and accuracy
    val_loss, val_acc = best_model.evaluate(X_valid, y_valid)
    print("Validation loss:", val_loss)
    print("Validation accuracy:", val_acc)

    # predictions
    y_true_flat = y_valid.flatten()
    y_pred_flat = np.argmax(best_model.predict(X_valid), axis=-1).flatten()
    mask_valid = y_true_flat != 0
    y_true_valid = y_true_flat[mask_valid]
    y_pred_valid = y_pred_flat[mask_valid]
    

    print(classification_report(y_true_valid, y_pred_valid))
    report = classification_report(y_true_valid, y_pred_valid)
    with open(outpath + "plots_2/classification_report_valid.txt", "w") as f:
        f.write(report)


    #print(classification_report(y_true_flat, y_pred_flat))
    VisualizeResults(10, X_train, y_train, best_model, "train",
                 outpath + "plots_2/train_sample_10.png")
    VisualizeResults(30, X_train, y_train, best_model, "train",
                 outpath + "plots_2/train_sample_30.png")
    VisualizeResults(60, X_train, y_train, best_model, "train",
                 outpath + "plots_2/train_sample_60.png")
    VisualizeResults(0, X_valid, y_valid, best_model, "valid",
                      outpath + "plots_2/valid_sample_0.png")
    VisualizeResults(40, X_valid, y_valid, best_model, "valid",
                      outpath + "plots_2/valid_sample_40.png")
    VisualizeResults(80, X_valid, y_valid, best_model, "valid",
                      outpath + "plots_2/valid_sample_80.png")


    # IoU (intersection over union)
    ious, miou = compute_iou_per_class(y_true_valid, y_pred_valid, num_classes=6)

    print("IoU per class:", ious)
    print("Mean IoU:", miou)

    # Boundary IoU
    bious, mean_biou = compute_boundary_iou(y_true_valid, y_pred_valid, num_classes=6)

    print("Boundary IoU per class:", bious)
    print("Mean Boundary IoU:", mean_biou)

    with open(outpath + "plots_2/valid_iou_metrics.txt", "w") as f:
        f.write("IoU per class:\n")
        for i, val in enumerate(ious):
            f.write(f"Class {i}: {val:.4f}\n")

        f.write("\nMean IoU:\n")
        f.write(f"{miou:.4f}\n")

        f.write("\nBoundary IoU per class:\n")
        for i, val in enumerate(bious):
            f.write(f"Class {i}: {val:.4f}\n")

        f.write("\nMean Boundary IoU:\n")
        f.write(f"{mean_biou:.4f}\n")

    # confusion matrix
    # y_true and y_pred_labels are (H,W)

    # Confusion matrix (classes 1–6)
    # Shift classes 1–6 → 0–5
    mask = y_true_valid != 0
    y_true_cm = y_true_valid[mask]
    y_pred_cm = y_pred_valid[mask]


    cm = confusion_matrix(y_true_cm, y_pred_cm, labels=[1,2,3,4,5,6], normalize='true')
    print("Unique true classes:", np.unique(y_true_cm))

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_fig(outpath + "plots_2/confusion_matrix_valid.png")
    plt.close()

    # integrated gradients
    num_classes = 7
    classes_to_compute = [1,2,3,4,5,6]
    all_grouped_scores = []

    for cls in classes_to_compute:
        print(f"Computing IG for class {cls}... (VALID)")
        scores = compute_channel_importance(best_model, X_valid[:50], cls, steps=25)

        # Group one-hot channels
        texture_scores = scores[start : start + texture_n].sum()
        wetness_scores = scores[start + texture_n : start + texture_n + wetness_n].sum()
        soil_group_scores = scores[start + texture_n + wetness_n :
                                start + texture_n + wetness_n + soil_group_n].sum()

        grouped_scores = list(scores[:start]) + [texture_scores, wetness_scores, soil_group_scores]
        all_grouped_scores.append(grouped_scores)

        grouped_names = raster_channel_names + ["Texture", "Wetness", "Soil Group"]

        plt.figure(figsize=(10, 12))
        plt.barh(grouped_names, grouped_scores)
        plt.xlabel("Importance")
        plt.title(f"Channel Importance for Class {cls} (Validation)")
        plt.tight_layout()
        plt.savefig(outpath + f"plots_2/valid_channel_importance_class_{cls}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    # Overall average across classes
    overall_scores = np.mean(all_grouped_scores, axis=0)

    plt.figure(figsize=(10, 12))
    plt.barh(grouped_names, overall_scores)
    plt.xlabel("Importance")
    plt.title("Overall Channel Importance (Validation)")
    plt.tight_layout()
    plt.savefig(outpath + "plots_2/valid_channel_importance_overall.png", dpi=300)
    plt.close()


    #########################################################
    
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    # predictions
    y_true_flat = y_test.flatten()
    y_pred_flat = np.argmax(best_model.predict(X_test), axis=-1).flatten()

    print(classification_report(y_true_flat, y_pred_flat))
    with open(outpath + "plots_2/classification_report_test.txt", "w") as f:
        f.write(classification_report(y_true_flat, y_pred_flat))

    VisualizeResults(0, X_test, y_test, best_model, "test",
                      outpath + "plots_2/test_sample_0.png")
    VisualizeResults(10, X_test, y_test, best_model, "test",
                      outpath + "plots_2/test_sample_10.png")
    VisualizeResults(20, X_test, y_test, best_model, "test",
                      outpath + "plots_2/test_sample_10.png")
    VisualizeResults(25, X_test, y_test, best_model, "test",
                      outpath + "plots_2/test_sample_25.png")
    VisualizeResults(30, X_test, y_test, best_model, "test",
                      outpath + "plots_2/test_sample_30.png")
    VisualizeResults(40, X_test, y_test, best_model, "test",
                      outpath + "plots_2/test_sample_40.png")
    VisualizeResults(50, X_test, y_test, best_model, "test",
                      outpath + "plots_2/test_sample_50.png")


    # IoU
    y_true_test = y_test.flatten()
    y_pred_test = np.argmax(best_model.predict(X_test), axis=-1).flatten()

    ious_test, miou_test = compute_iou_per_class(y_true_test, y_pred_test, num_classes=6)
    
    # Boundary IoU
    bious_test, mean_biou_test = compute_boundary_iou(y_true_test, y_pred_test, num_classes=6)

    print("IoU per class:", ious)
    print("Mean IoU:", miou)
    with open(outpath + "plots_2/test_iou_metrics.txt", "w") as f:
        f.write("IoU per class:\n")
        for i, val in enumerate(ious_test):
            f.write(f"Class {i}: {val:.4f}\n")

        f.write("\nMean IoU:\n")
        f.write(f"{miou_test:.4f}\n")

        f.write("\nBoundary IoU per class:\n")
        for i, val in enumerate(bious_test):
            f.write(f"Class {i}: {val:.4f}\n")

        f.write("\nMean Boundary IoU:\n")
        f.write(f"{mean_biou_test:.4f}\n")

    print("Boundary IoU per class:", bious)
    print("Mean Boundary IoU:", mean_biou)


    # confusion matrix
    mask_test = y_true_test != 0
    y_true_test_cm = y_true_test[mask_test]
    y_pred_test_cm = y_pred_test[mask_test]

    cm = confusion_matrix(y_true_test_cm, y_pred_test_cm, labels=[1,2,3,4,5,6], normalize='true')
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix (Testing)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_fig(outpath + "plots_2/confusion_matrix_test.png")
    plt.close()

    ##integrated gradients
    num_classes = 7
    classes_to_compute = [1,2,3,4,5,6]
    all_grouped_scores = []

    for cls in classes_to_compute:
        print(f"Computing IG for class {cls}... (TEST)")
        scores = compute_channel_importance(best_model, X_test[:50], cls, steps=25)

        # Group one-hot channels
        texture_scores = scores[start : start + texture_n].sum()
        wetness_scores = scores[start + texture_n : start + texture_n + wetness_n].sum()
        soil_group_scores = scores[start + texture_n + wetness_n :
                                start + texture_n + wetness_n + soil_group_n].sum()

        grouped_scores = list(scores[:start]) + [texture_scores, wetness_scores, soil_group_scores]
        all_grouped_scores.append(grouped_scores)

        grouped_names = raster_channel_names + ["Texture", "Wetness", "Soil Group"]

        plt.figure(figsize=(10, 12))
        plt.barh(grouped_names, grouped_scores)
        plt.xlabel("Importance")
        plt.title(f"Channel Importance for Class {cls} (Testing)")
        plt.tight_layout()
        plt.savefig(outpath + f"plots_2/test_channel_importance_class_{cls}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    # Overall average across classes
    overall_scores = np.mean(all_grouped_scores, axis=0)

    plt.figure(figsize=(10, 12))
    plt.barh(grouped_names, overall_scores)
    plt.xlabel("Importance")
    plt.title("Overall Channel Importance (Testing)")
    plt.tight_layout()
    plt.savefig(outpath + "plots_2/test_channel_importance_overall.png", dpi=300)
    plt.close()


    ###########################################################

    # stacked_path = outpath + "stacked_multiband_64.tif"

    # #Predict full image without loading it
    # class_map, entropy_map, margin_uncertainty = predict_large_image(
    #     stacked_path,
    #     best_model,
    #     patch_size=(64, 64),
    #     stride=2,
    #     debug_one_patch=False
    # )

    # # debugging version
    # # debug_results = predict_large_image(
    # #     stacked_path,
    # #     best_model,
    # #     patch_size=(64, 64),
    # #     stride=16,
    # #     debug_one_patch=True
    # # )

    # # class_mosaic, entropy_mosaic, margin_mosaic = stitch_debug_patches(debug_results)
    # # visualize_mosaic(class_mosaic, entropy_mosaic, margin_mosaic)



    # rgba = apply_colours_rgba(class_map, grade_colours)
    # rgba_uint8 = (rgba * 255).astype(np.uint8)

    # print("Input exists:", os.path.exists(stacked_path))
    # print("Input readable:", os.access(stacked_path, os.R_OK))
    # print("Output dir exists:", os.path.exists(outpath))
    # print("Output writable:", os.access(outpath, os.W_OK))

    # with rio.open(stacked_path) as src:
    #     profile = src.profile

    # profile.update({"count": 4, "dtype": "uint8"})
    # profile.pop("nodata", None)

    # class_output_path = outpath + "classified_output.TIF"
    # with rio.open(class_output_path, "w", **profile) as dst:
    #     dst.write(np.transpose(rgba_uint8, (2, 0, 1)))

    # # Save entropy + margin uncertainty maps
    # entropy_map_path = outpath + "uncertainty_entropy.tif"
    # margin_uncertainty_path = outpath + "margin_uncertainty.tif"

    # # Read profile from the stacked input image
    # with rio.open(stacked_path) as src:
    #     profile = src.profile
    #     profile.update(count=1, dtype='float32')
    #     profile.pop("nodata", None)

    # # Save entropy map
    # with rio.open(entropy_map_path, "w", **profile) as dst:
    #     dst.write(entropy_map.astype(np.float32), 1)

    # # Save margin uncertainty map
    # with rio.open(margin_uncertainty_path, "w", **profile) as dst:
    #     dst.write(margin_uncertainty.astype(np.float32), 1)

    # #visuals
    # cmap = plt.cm.viridis
    # cmap.set_bad("white")  # nodata = white

    # plt.imshow(entropy_map, cmap=cmap, vmin=0, vmax=np.nanmax(entropy_map)) 
    # plt.colorbar(label="Entropy")
    # save_fig(outpath + "plots/entropy_map_coloured.png") # low entropy (confident) = dark, high entropy (uncertainty) = light, nodata = white
    # plt.close()

    # plt.imshow(margin_uncertainty, cmap=cmap, vmin=0, vmax=np.nanmax(margin_uncertainty))
    # plt.colorbar(label="Margin Uncertainty")
    # save_fig(outpath + "plots/margin_uncertainty_coloured.png") # high uncertainty = high values = bright
    # plt.close()
