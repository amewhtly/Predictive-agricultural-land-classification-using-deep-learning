import numpy as np
import rasterio as rio
from rasterio.transform import xy
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
from shapely.geometry import Polygon
import random
from skimage.util import random_noise
import pandas as pd
import os
import sys
import glob
import multiprocessing
import time
#import earthpy as et
from rasterio.transform import from_bounds
import csv

start = time.time()

def extract_soil(path, rasters, profile, band, outpath):
    df = pd.DataFrame()

    # loop through each soil raster
    for i in rasters:
        Name = i[:-4] # remove .tif

        with rio.open(path+i, 'r') as src:            
            band = src.read(1) # read raster band
            data = band.ravel() # flatten to 1D for dataframe

            df[Name] = data
            # replace extreme values >9000 with 65535 to represent nodata
            df[Name] = np.where(df[Name] > 9000, 65535, df[Name])
            
    # replace known nodata values with NaN so not includeddd
    df.replace([0,-9999, 65535], np.nan, inplace=True)
    
    # Conditions for selecting soil series to use
    # priority order?
    conditions = [
        ~df['S25'].isna(),      # If 'S25' is not NaN
        ~df['S50'].isna(),      # If 'S50' is not NaN
        ~df['S63'].isna()       # If 'S63' is not NaN
    ]

    # Corresponding choices for series
    choices = [
        df['S25'], 
        df['S50'], 
        df['S63']
    ]
    
    # Corresponding choices for tags
    # eg S25 = 1
    tag_choices = [
        1,2,3
    ]

    # creating soil series and tag maps
    # Use numpy.select() to create 'result' column
    df['SERIES'] = np.select(conditions, choices, default=np.nan)
    df['TAG'] = np.select(conditions, tag_choices, default=np.nan) 

    # Reshape SERIES into 2D raster

    # write output soil series map and corresponding tags
    VP = df['SERIES'].values
    P = np.reshape(VP,(band.shape[0],band.shape[1])) 
    P = np.where(np.isnan(P), 65535, P) 

    # update profile for output rasters
    # Set NoData value in the profile
    # Lempel-Ziv-Welch (LZW) compression is a lossless data compression 
    # algorithm that reduces the file size of raster data without affecting the quality or accuracy of the data.
    profile.update(dtype=rio.uint16, count=1, nodata=65535, compress='lzw')
    


    # save soil hybrid raster
    with rio.open(outpath + 'soil_hybrid.tif', 'w', **profile) as dst:
        dst.write(P.astype(rio.uint16),1)
        print('Saved Soil hybrid raster')
        
    # save TAG raster
    VP = df['TAG'].values
    P = np.reshape(VP,(band.shape[0],band.shape[1]))
    P = np.where(np.isnan(P), 65535, P)    
    with rio.open(outpath + 'soil_hybrid_tag.tif', 'w', **profile) as dst:
        dst.write(P.astype(rio.uint16),1)    
        print('Saved tag raster')  


    # create a 0/1 raster mask
    valid_mask = (P != 65535).astype(np.uint8)
    with rio.open(outpath + 'valid_mask.tif', 'w', **profile) as dst:
        dst.write(valid_mask, 1)
        print('Saved valid raster mask')

    # with rio.open(outpath + 'soil_hybrid_tag.tif') as src:
    #     arr = src.read(1)
    #     transform = src.transform

    return df


def sample_chips_soil_strata(
        soil_path,
        strata_path,
        raster_paths,
        outpath,
        chip_size=64,
        stride=2,              # test centres every [] pixels
        target_per_stratum=800  # representative sampling
    ):
    """
    Dense sliding-window chip sampler:
    - Tests chip centres frequently (stride < chip_size)
    - Ensures NO overlap between accepted chips
    - Ensures representative sampling across strata
    - Ensures chips are fully valid across all rasters
    - Outputs GIS-compatible polygons
    """

    half = chip_size // 2

    # Load STRATA raster (defines pixel grid)

    with rio.open(strata_path) as src_strata:
        strata = src_strata.read(1)
        transform = src_strata.transform
        h, w = strata.shape
        crs = src_strata.crs
        strata_nodata = src_strata.nodata

    # Load soil raster
    with rio.open(soil_path) as src_soil:
        soil = src_soil.read(1)
        soil_nodata = src_soil.nodata

    # Load other rasters

    rasters = []
    nodatas = []
    for rp in raster_paths:
        with rio.open(rp) as src:
            rasters.append(src.read(1))
            nodatas.append(src.nodata)


    # Combined validity mask
    valid_strata = (
        (strata != strata_nodata) &
        (~np.isnan(strata)) &
        (strata >= 0)  # if your strata classes are 0–5
    )

    valid_masks = [
        (soil != soil_nodata) & ~np.isnan(soil),
        valid_strata
    ]


    for arr, nd in zip(rasters, nodatas):
        valid_masks.append((arr != nd) & ~np.isnan(arr))

    valid_all = np.logical_and.reduce(valid_masks)
    print("Valid pixels:", np.sum(valid_all))

    
    # Prepare strata buckets
    
    strata_values = sorted([v for v in np.unique(strata) if v != strata_nodata])
    strata_centres = {s: [] for s in strata_values}

    accepted = []  # list of accepted chip centres
    
    
    # Dense sliding-window scan
    
    for r in range(half, h - half, stride):
        for c in range(half, w - half, stride):

            # Stop early if all strata filled
            if all(len(v) >= target_per_stratum for v in strata_centres.values()):
                break

            # Check window validity
            if not np.all(valid_all[r-half:r+half, c-half:c+half]):
                continue

            # Check stratum
            s = strata[r, c]
            if s not in strata_centres:
                continue
            if len(strata_centres[s]) >= target_per_stratum:
                continue


            # Check overlap with previously accepted chips (AABB test)
            overlap = False
            for (ar, ac, _) in accepted:
                if abs(ar - r) < chip_size and abs(ac - c) < chip_size:
                    overlap = True
                    break

            if overlap:
                continue

            # Accept chip
            accepted.append((r, c, s))
            strata_centres[s].append((r, c))

    print("Accepted:", len(accepted))
    print("Strata counts:", {s: len(v) for s, v in strata_centres.items()})


    
    #  Build chip polygons

    chip_polygons = []
    strata_list = []

    for (r, c, s) in accepted:
        r0, r1 = r - half, r + half
        c0, c1 = c - half, c + half

        x0, y0 = xy(transform, r0, c0)
        x1, y1 = xy(transform, r0, c1)
        x2, y2 = xy(transform, r1, c1)
        x3, y3 = xy(transform, r1, c0)

        poly = Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])
        chip_polygons.append(poly)
        strata_list.append(s)

    gdf = gpd.GeoDataFrame(
        {"strata": strata_list},
        geometry=chip_polygons,
        crs=crs
    )

    outfile = os.path.join(outpath, "chip_squares.geojson")
    # Write to a temp file
    tmpfile = outfile + ".tmp"
    gdf.to_file(tmpfile, driver="GeoJSON")

    # Replace
    os.replace(tmpfile, outfile)

    return gdf, accepted


def multiband_sampling(
        soil_path,
        strata_path,
        raster_paths,
        accepted,
        outpath,
        chip_size = 64,
):
    """
    Extract multiband chips for each polygon in chip_squares.geojson.
    """
    chip_dir = os.path.join(outpath, "chip64")
    os.makedirs(chip_dir, exist_ok=True)
    lbl_dir = os.path.join(outpath, "lbl64")
    os.makedirs(lbl_dir, exist_ok=True)


    # categorical classes (must be defined BEFORE stacking)
    categorical_classes = {
        "TEXTURE50.tif": range(1, 22),
        "SOILGROUP50.tif": range(22, 103),
        "WETNESS50.tif": range(1, 8)
    }
    
    files = raster_paths
    stacked_path = os.path.join(outpath, "stacked_multiband_64.tif")

    if os.path.exists(stacked_path):
        os.remove(stacked_path)

    # read metadata from first raster
    with rio.open(files[0]) as src0:
        meta = src0.meta.copy()

    # count expected bands EXACTLY like chip extraction
    num_cat_bands = sum(len(categorical_classes[os.path.basename(p)])
                        for p in files
                        if os.path.basename(p) in categorical_classes)

    num_num_bands = sum(1 for p in files
                        if os.path.basename(p) not in categorical_classes)

    expected_bands = num_cat_bands + num_num_bands

    # update metadata
    meta.update(count=expected_bands, dtype="float32", compress="lzw", nodata=np.nan)

    # write stacked raster
    with rio.open(stacked_path, "w", **meta) as dst:
        band_index = 1

        for filename in files:
            base = os.path.basename(filename)

            with rio.open(filename) as src:
                raw = src.read(1).astype("float32")

                # mask nodata
                if src.nodata is not None:
                    raw[raw == src.nodata] = np.nan

                # soil uses class0 as nodata
                # if base == os.path.basename(soil_path):
                #     raw[raw == 0] = np.nan

                # numeric band
                if base not in categorical_classes:
                    dst.write(raw, band_index)
                    band_index += 1

                # categorical → one-hot encode
                else:
                    for cls in categorical_classes[base]:
                        dst.write((raw == cls).astype("float32"), band_index)
                        band_index += 1


    with rio.open(stacked_path) as src:
        for ch in [18, 20, 25]:
            band = src.read(ch+1)
            print(f"Band {ch} global unique:", np.unique(band))

    print(f"Raster data has been written to {stacked_path}")


    csv_path = os.path.join(outpath, "chip_strata_stats_64.csv")
    print("CSV Creation...")

    excluded = []


    # Create CSV with header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chip_id",
            "strata_1","strata_2","strata_3","strata_4","strata_5","strata_6",
            "prop_1","prop_2","prop_3","prop_4","prop_5","prop_6"
        ])


        # chip extraction

        with rio.open(stacked_path) as src_img, rio.open(strata_path) as src_lbl:
            chip_id = 1

            for centre_row, centre_col, s in accepted:

                # top-left of 64×64 window, centred on the pixel
                row_off = centre_row - chip_size // 2
                col_off = centre_col - chip_size // 2

                window = rio.windows.Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=chip_size,
                    height=chip_size
                )

                # holds all numerical and one hot bands
                expanded_bands = []

                for filename in files:
                    with rio.open(filename) as src:
                        band = src.read(1, window=window)
                        base = os.path.basename(filename)

                        # numerical band
                        if base not in categorical_classes:
                            # numerical bands append directly
                            expanded_bands.append(band.astype("float32"))
                            continue

                        # categorical band -> one-hot encode
                        for cls in categorical_classes[base]:
                            # for each class create a binary mask band
                            expanded_bands.append((band == cls).astype("float32"))

                # stack all bands into 1 chip
                img_chip = np.stack(expanded_bands, axis=0)

                lbl_chip = src_lbl.read(1, window=window)       # shape: (64, 64)


                # nodata check (if needed)
                valid_classes = {1, 2, 3, 4, 5, 6}
                invalid_mask = ~np.isin(lbl_chip, list(valid_classes))

                if np.any(invalid_mask):
                    excluded.append({
                        "chip_id": chip_id,
                        "row": centre_row,
                        "col": centre_col,
                        "reason": "invalid_strata_class",
                        "values": np.unique(lbl_chip).tolist()
                    })
                    continue  # skip chip

                if (
                    np.any(img_chip == src_img.nodata) or
                    np.any(lbl_chip == src_lbl.nodata) or
                    np.isnan(img_chip).any() or
                    np.isnan(lbl_chip).any()
                ):
                    excluded.append({
                        "chip_id": chip_id,
                        "row": centre_row,
                        "col": centre_col,
                        "reason": "nodata/nan",
                        "values": np.unique(lbl_chip).tolist()
                    })
                    continue
                
                # compute class proportions
                unique, counts = np.unique(lbl_chip, return_counts=True)
                pixel_counts = dict(zip(unique, counts))

                total_pixels = lbl_chip.size  # should be 4096
                proportions = {cls: cnt / total_pixels for cls, cnt in pixel_counts.items()}
                
                row_counts = [pixel_counts.get(i, 0) for i in range(1,7)]
                row_props  = [proportions.get(i, 0) for i in range(1,7)]


                # write CSV
                writer.writerow([chip_id] + row_counts + row_props)

                # save chip + label
                chip_path = os.path.join(chip_dir, f"chip_{chip_id:05d}.tif")
                lbl_path  = os.path.join(lbl_dir,  f"lbl_{chip_id:05d}.tif")

                # update chip metadata
                chip_profile = src_img.profile.copy()
                chip_profile.update(
                    count=img_chip.shape[0],
                    height=chip_size,
                    width=chip_size,
                    transform=rio.windows.transform(window, src_img.transform)
                )

                # update label metadata
                lbl_profile = src_lbl.profile.copy()
                lbl_profile.update(
                    height=chip_size,
                    width=chip_size,
                    transform=rio.windows.transform(window, src_lbl.transform)
                )

                # write file
                with rio.open(chip_path, "w", **chip_profile) as dst:
                    dst.write(img_chip)

                with rio.open(lbl_path, "w", **lbl_profile) as dst:
                    dst.write(lbl_chip, 1)

                print(img_chip.shape, lbl_chip.shape, sum(row_counts), sum(row_props))

                chip_id += 1

    #expected_bands = 1 + len(raster_paths)   # or 2 + len(raster_paths) if you also stacked strata

    # num_cat_bands = sum(len(categorical_classes[os.path.basename(p)]) 
    #                 for p in raster_paths 
    #                 if os.path.basename(p) in categorical_classes)

    # num_num_bands = len(raster_paths) - len(categorical_classes)

    # expected_bands = 1 + num_num_bands + num_cat_bands

    for chip_file in sorted(os.listdir(chip_dir)):
        chip_path = os.path.join(chip_dir, chip_file)

        with rio.open(chip_path) as src:
            if src.count != expected_bands:
                print("Wrong band count:", chip_file)
            else:
                print("OK:", chip_file)

    print("All chips extracted.")
    with rio.open(stacked_path) as src:
        print("Stacked band count:", src.count)
    print("Chip band count:", img_chip.shape[0])

    print("Total accepted centres:", len(accepted))
    print("Total chips written:", chip_id - 1)
    print("Number of chip files:", len(os.listdir(chip_dir)))
    print("Total chips excluded:", len(excluded))
    print("Excluded chip details:")
    for e in excluded:
        print(e)

    for ch in [22, 23, 24]:
        print("Chip unique:", np.unique(img_chip[ch]))




path = 'C:/Users/milli/OneDrive/Documents/Cranfield University/Thesis/Data/'
pathC = path + 'target/'
pathP = path + 'predictors/'
pathsoil = path + 'soildata/'
outpath = path + 'new/'

print('Detailed soil data....')


S25 = 'S25.tif'
S50 = 'S50.tif'
S63 = 'S63.tif'
soil = [S25, S50,S63]

with rio.open(pathsoil + S25) as src:
    profile = src.profile
    band = src.read(1)




dfS = extract_soil(pathsoil, soil, profile, band, outpath)
dfS.to_csv(pathC + 'soil.csv')

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


gdf, accepted = sample_chips_soil_strata(
    soil_path = outpath + "soil_hybrid.tif",     # primary raster with NoData
    strata_path = strata_path,                   # ALC_predict_hybrid.tif
    raster_paths = raster_paths,                 # other rasters that must be valid
    outpath = outpath,
    chip_size=64,
    stride=2,
    target_per_stratum=800,
)


multiband_sampling(
        soil_path =  outpath + "soil_hybrid.tif",
        strata_path = strata_path,
        raster_paths = raster_paths,
        accepted = accepted,
        outpath = outpath,
        chip_size = 64,
)

print('Done!')
for i, rp in enumerate(raster_paths):
    print(i, os.path.basename(rp))

